#include "locorb/moprintmod.hpp"
#include "io/molden.hpp"
#include "math/linalg/SVD.hpp"

namespace megalochem {

namespace locorb {

moprintmod::moprintmod(moprintmod::create_pack&& p) :
		m_world(p.p_set_world),
		m_wfn(p.p_set_wfn),
		m_filename(p.p_filename),
		m_job_name(p.p_job_name),
		m_lmo_occ(p.p_lmo_occ),
		m_lmo_vir(p.p_lmo_vir),
		m_print(p.p_print ? *p.p_print : 0),
		LOG(m_world.comm(), p.p_print ? *p.p_print : 0),
		TIME(m_world.comm(), "moprintmod")
{
}

void moprintmod::compute() {
	
	LOG.banner("MO PRINT AND LOCALIZER MODULE", 50, '*');
	
	if (!m_wfn->hf_wfn) {
		throw std::runtime_error("Could not find hartree fock wave function.");
	}
	
	auto hfwfn = m_wfn->hf_wfn;
	auto mol = m_wfn->mol;
	
	auto cOA = hfwfn->c_bo_A();
	auto cOB = hfwfn->c_bo_B();
	auto cVA = hfwfn->c_bv_A();
	auto cVB = hfwfn->c_bv_B();
	
	auto eOA = hfwfn->eps_occ_A();
	auto eOB = hfwfn->eps_occ_B();
	auto eVA = hfwfn->eps_vir_A();
	auto eVB = hfwfn->eps_vir_B();
		
	if (m_job_name == job_type::cmo) {
		LOG.os<>("Writing CMOs to file\n");
		io::write_molden("cmo_" + m_filename, m_world, *mol, cOA, cVA, *eOA, *eVA, 
			cOB, cVB, eOB ? *eOB : std::vector<double>{}, 
			eVB ? *eVB : std::vector<double>{});
		return;
	}
	
	if (m_job_name == job_type::local) {
		
		locorb::mo_localizer moloc(m_world, mol);
		ints::aofactory aofac(mol, m_world);
		auto s_bb = aofac.ao_overlap();
		
		auto localize = [&] (auto ltype, auto c_bm, auto eps_m) {
			decltype(c_bm) l_br, u_rm;
			switch (ltype) {
				case lmo_type::boys: 
					std::tie(l_br, u_rm) = moloc.compute_boys(c_bm, s_bb);
					break;
				case lmo_type::pao:
					std::tie(l_br, u_rm) = moloc.compute_pao(c_bm, s_bb);
					break;
				case lmo_type::cholesky:
					std::tie(l_br, u_rm) = moloc.compute_cholesky(c_bm, s_bb);
					break;
			}
			
			// form fock matrix
			auto m = c_bm->col_blk_sizes();
			auto r = l_br->col_blk_sizes();
			
			auto f_mm = dbcsr::matrix<double>::create()
				.set_cart(m_world.dbcsr_grid())
				.name("f_mm")
				.row_blk_sizes(m)
				.col_blk_sizes(m)
				.matrix_type(dbcsr::type::symmetric)
				.build();
				
			auto f_mr = dbcsr::matrix<double>::create()
				.set_cart(m_world.dbcsr_grid())
				.name("f_mr")
				.row_blk_sizes(m)
				.col_blk_sizes(r)
				.matrix_type(dbcsr::type::no_symmetry)
				.build();
			
			auto f_rr = dbcsr::matrix<double>::create()
				.set_cart(m_world.dbcsr_grid())
				.name("f_rr")
				.row_blk_sizes(r)
				.col_blk_sizes(r)
				.matrix_type(dbcsr::type::symmetric)
				.build();
				
			f_mm->reserve_diag_blocks();
			f_mm->set_diag(*eps_m);
			
			dbcsr::multiply('N', 'T', 1.0, *f_mm, *u_rm, 0.0, *f_mr)
				.perform();
				
			dbcsr::multiply('N', 'N', 1.0, *u_rm, *f_mr, 0.0, *f_rr)
				.perform();
				
			auto eps_r = f_rr->get_diag();
			return std::make_tuple(l_br, eps_r);
			
		};
		
		auto [lcOA, leOA]  = localize(*m_lmo_occ, cOA, eOA);
		auto [lcVA, leVA]  = localize(*m_lmo_vir, cVA, eVA);
		
		LOG.os<>("Writing LMOs to file\n");
		io::write_molden("lmo_" + m_filename, m_world, *mol, lcOA, lcVA, leOA, leVA, 
			cOB, cVB, eOB ? *eOB : std::vector<double>{}, 
			eVB ? *eVB : std::vector<double>{});
		return;
			
	}
	
	if (m_job_name == job_type::nto) {
		
		auto adcwfn = m_wfn->adc_wfn;
		if (!adcwfn) {
			throw std::runtime_error("Could not find adc wave function.");
		}
		
		auto eigvecs = adcwfn->davidson_eigenvectors();
		int nstates = eigvecs.size();
		
		for (int istate = 0; istate != nstates; ++istate) {
			
			auto r_ov = dbcsr::matrix<double>::copy(*eigvecs[istate])
				.build();
				
			r_ov->scale(1.0/r_ov->dot(*r_ov));
			
			math::SVD svd(m_world, r_ov, 'V', 'V', m_print);
			svd.compute();
			int rank = svd.rank();
			
			auto r = dbcsr::split_range(rank, m_wfn->mol->mo_split());
			auto o = r_ov->row_blk_sizes();
			auto v = r_ov->col_blk_sizes();
			auto b = m_wfn->mol->dims().b();
			
			auto u_or = svd.U(o,r);
			auto vt_rv = svd.Vt(r,v);
			
			auto co_br = dbcsr::matrix<double>::create()
				.name("co_br")
				.set_cart(m_world.dbcsr_grid())
				.row_blk_sizes(b)
				.col_blk_sizes(r)
				.matrix_type(dbcsr::type::no_symmetry)
				.build();
				
			auto cv_br = dbcsr::matrix<double>::create()
				.name("cv_br")
				.set_cart(m_world.dbcsr_grid())
				.row_blk_sizes(b)
				.col_blk_sizes(r)
				.matrix_type(dbcsr::type::no_symmetry)
				.build();
				
			dbcsr::multiply('N', 'N', 1.0, *cOA, *u_or, 0.0, *co_br).perform();
			dbcsr::multiply('N', 'T', 1.0, *cVA, *vt_rv, 0.0, *cv_br).perform();
			
			auto epso_r = svd.s();
			auto epsv_r = svd.s();
			
			std::for_each(epso_r.begin(), epso_r.end(), [](double& d) { d = -d; });
			
			std::string filename = "nto_" + std::to_string(istate) + "_" + m_filename;
			LOG.os<>("Writing NTOs to file\n");
			io::write_molden(filename, m_world, *mol, co_br, cv_br, epso_r, epsv_r, 
				nullptr, nullptr, std::vector<double>{}, std::vector<double>{});
			return;
			
		}
	}
		
		

}

}

}
