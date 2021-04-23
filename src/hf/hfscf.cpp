#include "hf/hfmod.hpp"
#include "hf/hfdefaults.hpp"
#include "math/linalg/orthogonalizer.hpp"
#include "math/solvers/diis.hpp"
#include "math/linalg/piv_cd.hpp"

namespace hf {
	
hfmod::hfmod(dbcsr::world w, desc::shared_molecule mol, desc::options opt) 
	: m_mol(mol), 
	  m_opt(opt), 
	  m_world(w),
	  LOG(m_world.comm(),m_opt.get<int>("print", HF_PRINT_LEVEL)),
	  TIME(m_world.comm(), "Hartree Fock", LOG.global_plev()),
	  m_guess(m_opt.get<std::string>("guess", HF_GUESS)),
	  m_max_iter(m_opt.get<int>("max_iter", HF_MAX_ITER)),
	  m_scf_threshold(m_opt.get<double>("scf_thresh", HF_SCF_THRESH)),
	  m_diis(m_opt.get<bool>("diis", HF_SCF_DIIS)),
	  m_diis_beta(m_opt.get<bool>("diis_beta", HF_DIIS_BETA)),
	  m_scf_energy(0.0),
	  m_nobetaorb(false),
	  m_locc(m_opt.get<bool>("locc", HF_LOCC)),
	  m_lvir(m_opt.get<bool>("lvir", HF_LVIR))
{
	
	m_restricted = (m_mol->nele_alpha() == m_mol->nele_beta()) ? true : false; 
	
	if (m_mol->nocc_beta() == 0) m_nobetaorb = true;
	
	auto b = m_mol->dims().b();
	auto oA = m_mol->dims().oa();
	auto oB = m_mol->dims().ob();
	auto vA = m_mol->dims().va();
	auto vB = m_mol->dims().vb();
	
	vec<int> mA = oA;
	mA.insert(mA.end(), vA.begin(), vA.end());
	
	vec<int> mB = oB;
	mB.insert(mB.end(), vB.begin(), vB.end());
	
	arrvec<int,2> bb = {b,b};
	arrvec<int,2> bm_A = {b, mA};
	arrvec<int,2> bm_B = {b, mB};
	
	// create non-integral tensors
	
	// alpha
	
	m_core_bb = dbcsr::matrix<>::create()
		.set_world(m_world)
		.name("core_bb")
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::symmetric)
		.build();
		
	m_p_bb_A = dbcsr::matrix<>::create_template(*m_core_bb)
		.name("p_bb_A").build();
		
	m_f_bb_A = dbcsr::matrix<>::create_template(*m_core_bb)
		.name("f_bb_A").build();
	
	m_c_bm_A = dbcsr::matrix<>::create()
		.set_world(m_world)
		.name("c_bm_A")
		.row_blk_sizes(b)
		.col_blk_sizes(mA)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	if (!m_restricted) {
		
		m_p_bb_B = dbcsr::matrix<>::create_template(*m_core_bb)
			.name("p_bb_B").build();
		
		m_f_bb_B = dbcsr::matrix<>::create_template(*m_core_bb)
			.name("f_bb_B").build();
		
	}
	
	if (!m_nobetaorb && !m_restricted) {
		
		m_c_bm_B = dbcsr::matrix<>::create()
			.set_world(m_world)
			.name("c_bm_B")
			.row_blk_sizes(b)
			.col_blk_sizes(mB)
			.matrix_type(dbcsr::type::no_symmetry)
			.build();

	}
		
	m_eps_A = std::make_shared<std::vector<double>>(std::vector<double>(0));
	if (!m_restricted) m_eps_B = std::make_shared<std::vector<double>>(std::vector<double>(0));
	
	// basis set
	if (m_opt.present("dfbasis")) {
		 
		std::string basname = m_opt.get<std::string>("dfbasis");
		bool augmented = m_opt.get<bool>("df_augmentation", false);
		
		int nsplit = m_mol->c_basis()->nsplit();
		auto smethod = m_mol->c_basis()->split_method();
		auto atoms = m_mol->atoms();
		
		LOG.os<>("Setting df basis: ", basname, "\n\n");
		auto dfbasis = std::make_shared<desc::cluster_basis>(
			basname, atoms, smethod, nsplit, augmented);
	
		m_mol->set_cluster_dfbasis(dfbasis);
		
		auto x = m_mol->dims().x();
		if (LOG.global_plev() >= 1) {
			for (auto i : x) {
				LOG.os<1>(i, " ");
			} LOG.os<1>("\n");
		}
		
	}
	
	// integral machine
	
	std::optional<int> nbatches_b = m_opt.present("nbatches_b") ? 
		std::make_optional<int>(m_opt.get<int>("nbatches_b")) : 
		std::nullopt;
		
	std::optional<int> nbatches_x = m_opt.present("nbatches_x") ? 
		std::make_optional<int>(m_opt.get<int>("nbatches_x")) : 
		std::nullopt;
		
	std::optional<dbcsr::btype> btype_e = m_opt.present("eris") ?
		std::make_optional<dbcsr::btype>(dbcsr::get_btype(m_opt.get<std::string>("eris"))) :
		std::nullopt;
		
	std::optional<dbcsr::btype> btype_i = m_opt.present("intermeds") ?
		std::make_optional<dbcsr::btype>(dbcsr::get_btype(m_opt.get<std::string>("intermeds"))) :
		std::nullopt;
	
	m_aoloader = ints::aoloader::create()
		.set_world(m_world)
		.set_molecule(m_mol)
		.print(LOG.global_plev())
		.nbatches_b(nbatches_b)
		.nbatches_x(nbatches_x)
		.btype_eris(btype_e)
		.btype_intermeds(btype_i)
		.build();
	
}

hfmod::~hfmod() {}

void hfmod::compute_nucrep() {
	
	m_nuc_energy = 0.0;
	
	auto atoms = m_mol->atoms();
	
	for (int i = 0; i != atoms.size(); ++i) {
		for (int j = i+1; j < atoms.size(); ++j) {
			
			int Zi = atoms[i].atomic_number;
			int Zj = atoms[j].atomic_number;
			
			double dx = atoms[i].x - atoms[j].x;
			double dy = atoms[i].y - atoms[j].y;
			double dz = atoms[i].z - atoms[j].z;
			
			double R = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));
			
			m_nuc_energy += (Zi*Zj)/R;
			
		}
	}
	
	//LOG.os<>(0,"Nuclear Repulsion Energy: ", nuc_energy, '\n');
	
}

void hfmod::one_electron() {
	
	LOG.os<>("Forming one-electron integrals...\n");
	auto& TIME_1e = TIME.sub("One-Electron Integrals");
	
	TIME_1e.start();
	
	ints::aofactory int_engine(m_mol, m_world);
	
	// overlap			 
	m_s_bb = int_engine.ao_overlap();	
	
	//kinetic
	m_t_bb = int_engine.ao_kinetic();
	
	// nuclear
	m_v_bb = int_engine.ao_nuclear();
	
	// get X
	math::orthogonalizer og(m_s_bb, (LOG.global_plev() >= 2) ? true : false);
	og.compute();
	m_x_bb = og.result();
	
	//std::cout << "H1" << std::endl;
	m_core_bb->add(0.0, 1.0, *m_v_bb);
	m_core_bb->add(1.0, 1.0, *m_t_bb);
	
	TIME_1e.finish();
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_s_bb);
		LOG.os<>('\n');
		dbcsr::print(*m_t_bb);
		LOG.os<>('\n');
		dbcsr::print(*m_v_bb);
		LOG.os<>('\n');
		dbcsr::print(*m_core_bb);
		LOG.os<>('\n');
		dbcsr::print(*m_x_bb);
		LOG.os<>('\n');
	}

	m_s_bb->filter(dbcsr::global::filter_eps);
	m_t_bb->filter(dbcsr::global::filter_eps);
	m_v_bb->filter(dbcsr::global::filter_eps);
	m_x_bb->filter(dbcsr::global::filter_eps);
	m_core_bb->filter(dbcsr::global::filter_eps);
	
	LOG.os<>("Done with 1 electron integrals.\n");
	
}

void hfmod::two_electron() {
	
	LOG.os<>("Forming two-electron integrals...\n");
	auto& TIME_2e = TIME.sub("Two-Electron Integrals");
	
	TIME_2e.start();
	
	fock::jmethod jmeth = fock::str_to_jmethod(
		m_opt.get<std::string>("build_J"));
		
	fock::kmethod kmeth = fock::str_to_kmethod(
		m_opt.get<std::string>("build_K"));
		
	ints::metric metr = ints::str_to_metric(
		m_opt.get<std::string>("df_metric", HF_METRIC));
	
	fock::load_jints(jmeth, metr, *m_aoloader);
	fock::load_kints(kmeth, metr, *m_aoloader);
	
	m_aoloader->compute();
	
	std::optional<int> nocc_batches = (m_opt.present("occ_nbatches")) ?
		m_opt.get<int>("occ_nbatches") : 1;
	
	m_jbuilder = fock::create_j()
		.set_world(m_world)
		.molecule(m_mol)
		.print(LOG.global_plev())
		.aoloader(*m_aoloader)
		.method(jmeth)
		.metric(metr)
		.build();
	
	m_kbuilder = fock::create_k()
		.set_world(m_world)
		.molecule(m_mol)
		.print(LOG.global_plev())
		.aoloader(*m_aoloader)
		.method(kmeth)
		.metric(metr)
		.occ_nbatches(nocc_batches)
		.build();
		
	m_jbuilder->init();
	m_kbuilder->init();
	
	TIME_2e.finish();
	
	LOG.os<>("Done with 2 electron integrals.\n");
	
}

void hfmod::form_fock(bool SAD_iter, int rank) {
	
	LOG.os<>("Forming Fock matrix...\n");
	auto& TIME_2e = TIME.sub("Computing Fock matrix");
	
	auto pA_copy = dbcsr::matrix<>::copy(*m_p_bb_A).build();
	pA_copy->filter(dbcsr::global::filter_eps);
	
	LOG.os<1>("Ocupation of alpha density matrix: ", pA_copy->occupation(), '\n');
	pA_copy->release();
	
	m_jbuilder->set_density_alpha(m_p_bb_A);
	m_jbuilder->set_density_beta(m_p_bb_B);
	m_jbuilder->set_coeff_alpha(m_c_bm_A);
	m_jbuilder->set_coeff_beta(m_c_bm_B);
	
	m_kbuilder->set_density_alpha(m_p_bb_A);
	m_kbuilder->set_density_beta(m_p_bb_B);
	m_kbuilder->set_coeff_alpha(m_c_bm_A);
	m_kbuilder->set_coeff_beta(m_c_bm_B);
	
	m_jbuilder->set_SAD(SAD_iter,rank);
	m_kbuilder->set_SAD(SAD_iter,rank);
	
	m_jbuilder->compute_J();
	m_kbuilder->compute_K();
	
	auto j_bb = m_jbuilder->get_J();
	auto k_bb_A = m_kbuilder->get_K_A();
	auto k_bb_B = m_kbuilder->get_K_B();
	
	m_f_bb_A->clear();
	m_f_bb_A->add(1.0, 1.0, *m_core_bb);
	m_f_bb_A->add(1.0, 1.0, *j_bb);
	m_f_bb_A->add(1.0, 1.0, *k_bb_A);
	
	if (m_f_bb_B) {
		m_f_bb_B->clear();
		m_f_bb_A->add(1.0, 1.0, *m_core_bb);
		m_f_bb_B->add(1.0, 1.0, *j_bb);
		m_f_bb_B->add(1.0, 1.0, *k_bb_B);
	}
	
	LOG.os<>("Done with forming Fock matrix.\n");
	
}
	
void hfmod::compute_scf_energy() {		
	
	double e1A, e2A, e1B = 0.0, e2B = 0.0;
	
	e1A = m_core_bb->dot(*m_p_bb_A);
	e2A = m_f_bb_A->dot(*m_p_bb_A);
	
	if (!m_restricted) {
		e1B = m_core_bb->dot(*m_p_bb_B);
		e2B = m_f_bb_B->dot(*m_p_bb_B);
	} 
	
	if (m_restricted) {
		m_scf_energy = 0.5 * (2.0 * (e1A + e2A));
	} else {
		m_scf_energy = 0.5 * ((e1A + e2A) + (e1B + e2B));
	}
	
}

dbcsr::shared_matrix<double> hfmod::compute_errmat(
	dbcsr::shared_matrix<double>& F_x, 
	dbcsr::shared_matrix<double>& P_x, 
	dbcsr::shared_matrix<double>& S, std::string x) {
	
	auto e_1 = dbcsr::matrix<>::create_template(*F_x)
		.name("e_1_"+x)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto e_2 = dbcsr::matrix<>::create_template(*F_x)
		.name("e_2_"+x)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	//DO E = FPS - SPF
	
	dbcsr::multiply('N','N',1.0,*F_x,*P_x,0.0,*e_1).perform();
	dbcsr::multiply('N','N',1.0,*e_1,*S,0.0,*e_1).perform();
	dbcsr::multiply('N','N',-1.0,*S,*P_x,0.0,*e_2).perform();
	dbcsr::multiply('N','N',1.0,*e_2,*F_x,1.0,*e_1).perform();
	
	e_2->release();
	
	return e_1;
	
}

void hfmod::compute() {
	
	TIME.start();
	
	if (LOG.global_plev() >= 0) 
		LOG.banner<>("HARTREE FOCK", 50, '*');
	
	// first, get one-electron integrals...
	one_electron();
	
	// form the guess
	compute_guess();
	
	// then, get two-electron integrals
	two_electron();
	
	compute_nucrep();
	
	// Now enter loop
	int iter = 0;
	bool converged = false;
	
	int dmax = m_opt.get<int>("diis_max_vecs", HF_DIIS_MAX_VECS);
	int dmin = m_opt.get<int>("diis_min_vecs", HF_DIIS_MIN_VECS);
	int dstart = m_opt.get<int>("diis_start", HF_DIIS_START);
	
	math::diis_helper<2> diis_A(m_world.comm(),dstart, dmin, dmax, (LOG.global_plev() >= 2) ? true : false );
	math::diis_helper<2> diis_B(m_world.comm(),dstart, dmin, dmax, (LOG.global_plev() >= 2) ? true : false );
	
	// ERROR MATRICES
	dbcsr::shared_matrix<double> e_A;
	dbcsr::shared_matrix<double> e_B;
	
	size_t nbas = m_mol->c_basis()->nbf();
	
	double norm_A = 10;
	double norm_B = 10;
	
	// ---------> print info here <-------
	int width = 18;
	LOG.left();
	LOG.setw(width).os<>("Iteration Nr")
		.setw(width).os<>("Energy (Ht)")
		.setw(width).os<>("Error (Ht)")
		.setw(width).os<>("RMS alpha(Ht)")
		.setw(width).os<>("RMS beta(Ht)").os<>('\n');
		
	LOG.os<>("--------------------------------------------------------------------------------\n");
	
	auto RMS = [&](dbcsr::shared_matrix<double>& m) {
		double prod = m->dot(*m);
		return sqrt(prod/(nbas*nbas));
	};
		
	while (true) {
		
		// form fock matrix
		
		bool SAD_iter = ((iter == 0) && (m_guess == "SAD" || m_guess == "SADNO")) ? true : false;
		int rank = ((iter == 0) && (m_guess == "SAD" || m_guess == "SADNO")) ? m_SAD_rank : 0;
		
		form_fock(SAD_iter,rank);
						
		// compute error, do diis, compute energy
		
		e_A = compute_errmat(m_f_bb_A, m_p_bb_A, m_s_bb, "A");
		if (!m_restricted)
			e_B = compute_errmat(m_f_bb_B, m_p_bb_B, m_s_bb, "B");
		
		double old_energy = m_scf_energy;
		compute_scf_energy();
		
		norm_A = RMS(e_A);
		if (m_restricted) {
			norm_B = norm_A;
		} else {
			norm_B = RMS(e_B);
		}
		
		LOG.left();
		LOG.setw(width).os<>("UHF@"+std::to_string(iter));
		LOG.scientific();
		LOG.setprecision(10);
		LOG.setw(width).os<>(m_scf_energy + m_nuc_energy)
			.setw(width).os<>(old_energy - m_scf_energy)
			.setw(width).os<>(norm_A)
			.setw(width).os<>(norm_B).os<>('\n');
		LOG.reset();
		
		if (norm_A < m_scf_threshold && norm_B < m_scf_threshold && iter > 0) break;
		if (iter > m_max_iter) break;
		
		if (m_diis) {
			diis_A.compute_extrapolation_parameters(m_f_bb_A, e_A, iter);
			diis_A.extrapolate(m_f_bb_A, iter);
			if (!m_restricted && !m_nobetaorb) {
				
				if (m_diis_beta) {	
					// separate diis optimization for beta
					diis_B.compute_extrapolation_parameters(m_f_bb_B, e_B, iter);
					diis_B.extrapolate(m_f_bb_B, iter);
					
				} else {
					
					// impose the same coefficients for both alpha and beta
					auto coeffA = diis_A.coeffs();
					diis_B.compute_extrapolation_parameters(m_f_bb_B, e_B, iter);
					diis_B.extrapolate(m_f_bb_B,coeffA,iter);
					
				}
			}
		}
		
		// diag fock
		diag_fock();
		
		// loop
		 ++iter;
		
		
	} // end while
	
	e_A->release();
	if (e_B) e_B->release();
	
	if (iter > m_max_iter) throw std::runtime_error("HF did not converge.");
	
	LOG.os<>("Done with SCF cycle. Took ", iter, " iterations.\n");
	LOG.scientific();
	LOG.setprecision(12);
	LOG.os<>("Final SCF energy: ", m_scf_energy, '\n');
	LOG.os<>("Total energy: ", m_scf_energy + m_nuc_energy, '\n');
	LOG.reset();
	
	TIME.finish();
	
	TIME.print_info();
	m_aoloader->print_info();
	
	m_jbuilder->print_info();
	m_kbuilder->print_info();
	
}

} // end namespace
	
