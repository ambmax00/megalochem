#include "hf/hfmod.h"
#include "hf/hfdefaults.h"
#include "fock/fockmod.h"
#include "ints/aofactory.h"
#include "math/linalg/orthogonalizer.h"
#include "math/solvers/diis.h"

namespace hf {
	
hfmod::hfmod(dbcsr::world w, desc::smolecule mol, desc::options opt) 
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
	
	m_core_bb = dbcsr::create<double>()
		.set_world(m_world)
		.name("core_bb")
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::symmetric).get();
		
	m_p_bb_A = dbcsr::create_template<double>(m_core_bb)
		.name("p_bb_A").get();
		
	m_f_bb_A = dbcsr::create_template<double>(m_core_bb)
		.name("f_bb_A").get();	
	
	m_c_bm_A = dbcsr::create<double>()
		.set_world(m_world)
		.name("c_bm_A")
		.row_blk_sizes(b)
		.col_blk_sizes(mA)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	if (!m_restricted) {
		
		m_p_bb_B = dbcsr::create_template<double>(m_core_bb)
			.name("p_bb_B").get();
		
		m_f_bb_B = dbcsr::create_template<double>(m_core_bb)
			.name("f_bb_B").get();
		
	}
	
	if (!m_nobetaorb) {
		
		m_c_bm_B = dbcsr::create<double>()
			.set_world(m_world)
			.name("c_bm_B")
			.row_blk_sizes(b)
			.col_blk_sizes(mB)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();

	}
		
	m_eps_A = std::make_shared<std::vector<double>>(std::vector<double>(0));
	if (!m_restricted) m_eps_B = std::make_shared<std::vector<double>>(std::vector<double>(0));
	
	// basis set
	if (m_opt.present("dfbasis")) {
		 
		std::string basname = m_opt.get<std::string>("dfbasis");
		bool augmented = m_opt.get<bool>("df_augmentation", false);
		
		int nsplit =  m_opt.get<int>("df_ao_split"); //m_mol->c_basis()->nsplit();
		std::string smethod = m_opt.get<std::string>("df_ao_split_method");
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
	
	auto e_1 = dbcsr::create_template(F_x)
		.name("e_1_"+x)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto e_2 = dbcsr::create_template(F_x)
		.name("e_2_"+x)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	//DO E = FPS - SPF
	
	dbcsr::multiply('N','N',*F_x,*P_x,*e_1).perform();
	dbcsr::multiply('N','N',*e_1,*S,*e_1).perform();
	dbcsr::multiply('N','N',*S,*P_x,*e_2).alpha(-1.0).perform();
	dbcsr::multiply('N','N',*e_2,*F_x,*e_1).beta(1.0).perform();
	
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
	
	compute_nucrep();
	
	// Now enter loop
	int iter = 0;
	bool converged = false;
	
	int dmax = m_opt.get<int>("diis_max_vecs", HF_DIIS_MAX_VECS);
	int dmin = m_opt.get<int>("diis_min_vecs", HF_DIIS_MIN_VECS);
	int dstart = m_opt.get<int>("diis_start", HF_DIIS_START);
	
	fock::fockmod fbuilder(m_world, m_mol, m_opt);
	
	fbuilder.set_density_alpha(m_p_bb_A);
	fbuilder.set_density_beta(m_p_bb_B);
	fbuilder.set_coeff_alpha(m_c_bm_A);
	fbuilder.set_coeff_beta(m_c_bm_B);
	fbuilder.set_core(m_core_bb);
	
	fbuilder.init();
	
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
		
		fbuilder.compute(SAD_iter,rank);
						
		m_f_bb_A = fbuilder.get_f_A();
		m_f_bb_B = fbuilder.get_f_B();
		
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
	LOG.os<>("Final SCF energy: ", m_scf_energy, '\n');
	LOG.os<>("Total energy: ", m_scf_energy + m_nuc_energy, '\n');
	LOG.reset();
	
	TIME.finish();
	
	TIME.print_info();
	fbuilder.print_info();
		
}

} // end namespace
	
