#include "fock/fockmod.h"
#include "ints/screening.h"
#include "math/solvers/hermitian_eigen_solver.h"
#include "fock/fock_defaults.h"

namespace fock {
	
fockmod::fockmod (dbcsr::world iworld, desc::smolecule imol, desc::options iopt) :
	m_world(iworld),
	m_mol(imol),
	m_opt(iopt),
	LOG(m_world.comm(),m_opt.get<int>("print", FOCK_PRINT_LEVEL)),
	TIME(m_world.comm(), "Fock Builder", LOG.global_plev())
{
	// set up tensors
	auto b = m_mol->dims().b();
	dbcsr::mat_d fA = dbcsr::matrix<>::create().set_world(m_world).name("f_bb_A")
		.row_blk_sizes(b).col_blk_sizes(b).type(dbcsr_type_symmetric);
	
	m_f_bb_A = fA.get_smatrix();
	
	if (m_mol->nele_alpha() != m_mol->nele_beta() && m_mol->nele_beta() != 0) {
		dbcsr::mat_d fB = dbcsr::matrix<>::create_template(*m_f_bb_A).name("f_bb_B");
		m_f_bb_B = fB.get_smatrix();
	}
	
}

void fockmod::init() {
	
	std::string j_method = m_opt.get<std::string>("build_J", FOCK_BUILD_J);
	std::string k_method = m_opt.get<std::string>("build_K", FOCK_BUILD_K);
	
	bool compute_eris = false;
	bool compute_3c2e = false;
	bool compute_s_xx = false;
	bool compute_s_xx_inv = false;
	
	// set J
	if (j_method == "exact") {
		
		J* builder = new EXACT_J(m_world, m_opt);
		m_J_builder.reset(builder);
		
		compute_eris = true;
		
	} else if (j_method == "df") {
		
		J* builder = new DF_J(m_world, m_opt);
		m_J_builder.reset(builder);
		
		compute_3c2e = true;
		compute_s_xx = true;
		compute_s_xx_inv = true;
		
	}
	
	// set K
	if (k_method == "exact") {
		
		K* builder = new EXACT_K(m_world, m_opt);
		m_K_builder.reset(builder);
		
		compute_eris = true;
		
	} else if (k_method == "df") {
		
		K* builder = new DF_K(m_world,m_opt);
		m_K_builder.reset(builder);
		
		compute_3c2e = true;
		compute_s_xx = true;
		compute_s_xx_inv = true;
		
	}
	
	LOG.os<>("Setting up JK builder.\n");
	LOG.os<>("J method: ", j_method, '\n');
	LOG.os<>("K method: ", k_method, '\n');
	
	std::shared_ptr<ints::aofactory> aofac =
		std::make_shared<ints::aofactory>(m_mol,m_world);
		
	m_J_builder->set_density_alpha(m_p_A);
	m_J_builder->set_density_beta(m_p_B);
	m_J_builder->set_coeff_alpha(m_c_A);
	m_J_builder->set_coeff_beta(m_c_B);
	m_J_builder->set_factory(aofac);
	
	m_K_builder->set_density_alpha(m_p_A);
	m_K_builder->set_density_beta(m_p_B);
	m_K_builder->set_coeff_alpha(m_c_A);
	m_K_builder->set_coeff_beta(m_c_B);
	m_K_builder->set_factory(aofac);
	
	// initialize integrals depending on method combination
	
	ints::registry reg;
	
	if (compute_eris) {
		
		auto eris = aofac->ao_eri(vec<int>{0,1},vec<int>{2,3});
		reg.insert_tensor<4,double>(m_mol->name() + "_i_bbbb_(01|23)", eris);
		
	}
	
	if (compute_3c2e) {
		
		auto& t_screen = TIME.sub("3c2e screening");
		
		t_screen.start();
		
		ints::screener* scr = new ints::schwarz_screener(aofac);
		scr->compute();
		
		t_screen.finish();
		
		auto& t_eri = TIME.sub("3c2e integrals");
		t_eri.start();
		
		auto out = aofac->ao_3c2e(vec<int>{0}, vec<int>{1,2},scr);
		reg.insert_tensor<3,double>(m_mol->name() + "_i_xbb_(0|12)", out);
		
		t_eri.finish();
		
		delete scr;
		
	}
	
	if (compute_s_xx) {
		
		auto& t_eri = TIME.sub("Metric");
		
		t_eri.start();
		
		auto out = aofac->ao_3coverlap();
		reg.insert_matrix<double>(m_mol->name() + "_s_xx", out);
		
		t_eri.finish();
		
	}
	
	if (compute_s_xx_inv) {
		
		auto& t_inv = TIME.sub("Inverting metric");
		
		t_inv.start();
		
		auto s_xx = reg.get_matrix<double>(m_mol->name() + "_s_xx");
		math::hermitian_eigen_solver solver(s_xx, 'V');
		
		solver.compute();
		
		auto inv = solver.inverse();
		
		std::string name = m_mol->name() + "_s_xx_inv_(0|1)";
		
		dbcsr::pgrid<2> grid2(m_world.comm());
		arrvec<int,2> xx = {m_mol->dims().x(), m_mol->dims().x()};
		
		dbcsr::stensor2_d out = dbcsr::make_stensor<2>(
			dbcsr::tensor2_d::create().name(name).ngrid(grid2)
			.map1({0}).map2({1}).blk_sizes(xx));
			
		dbcsr::copy_matrix_to_tensor(*inv,*out);
		
		reg.insert_tensor<2,double>(name, out);
		
		t_inv.finish();
		
	}
	
	m_J_builder->init();
	m_K_builder->init();
	
	m_J_builder->init_tensors();
	m_K_builder->init_tensors();
	
	LOG.os<>("Finished setting up JK builder \n \n");
	
}

void fockmod::compute(bool SAD_iter) {
	
	TIME.start();
	
	m_J_builder->set_SAD(SAD_iter);
	m_K_builder->set_SAD(SAD_iter);
	
	auto& t_j = TIME.sub("J builder");
	auto& t_k = TIME.sub("K builder");
	
	LOG.os<1>("Computing coulomb matrix.\n");
	t_j.start();
	m_J_builder->compute_J();
	t_j.finish();
	
	LOG.os<1>("Computing exchange matrix.\n");
	t_k.start();	
	m_K_builder->compute_K();
	t_k.finish();
	
	auto Jtensor = m_J_builder->get_J();
	
	auto KtensorA = m_K_builder->get_K_A();
	auto KtensorB = m_K_builder->get_K_B();
	
	m_f_bb_A->add(0.0,1.0,*m_core);
	dbcsr::copy_tensor_to_matrix(*Jtensor,*m_f_bb_A,true);
	dbcsr::copy_tensor_to_matrix(*KtensorA,*m_f_bb_A,true);
	
	if (LOG.global_plev() >= 2) {
		dbcsr::print(*m_f_bb_A);
	}
	
	if (m_f_bb_B) {
		m_f_bb_B->add(0.0,1.0,*m_core);
		dbcsr::copy_tensor_to_matrix(*Jtensor,*m_f_bb_B,true);
		dbcsr::copy_tensor_to_matrix(*KtensorB,*m_f_bb_B,true);
		
		if (LOG.global_plev() >= 2) {
			dbcsr::print(*m_f_bb_B);
		}
	}
	
	TIME.finish();
		
}

} // end namespace
