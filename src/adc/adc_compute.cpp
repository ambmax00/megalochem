#include "adc/adcmod.h"
#include "adc/adc_mvp.h"
#include "math/solvers/davidson.h"
#include "math/linalg/piv_cd.h"
#include "locorb/locorb.h"

#include <dbcsr_conversions.hpp>

namespace adc {

void adcmod::compute() {
	
		// BEFORE: init tensors base (ints, metrics, etc..., put into m_reg)
		
		init_ao_tensors();
		
		// AFTER: init tensors (2) (mo-ints, diags, amplitudes ...) 
		
		init_mo_tensors();
				
		// SECOND: Generate guesses
		
		compute_diag();
		
		LOG.os<>("--- Starting Computation ---\n\n");
				
		int nocc = m_hfwfn->mol()->nocc_alpha();
		int nvir = m_hfwfn->mol()->nvir_alpha();
		auto epso = m_hfwfn->eps_occ_A();
		auto epsv = m_hfwfn->eps_vir_A();
		
		LOG.os<>("Computing guess vectors...\n");
		// now order it : there is probably a better way to do it
		auto eigen_ia = dbcsr::matrix_to_eigen(m_d_ov);
		
		//std::cout << eigen_ia << std::endl;
		
		std::vector<int> index(eigen_ia.size(), 0);
		for (int i = 0; i!= index.size(); ++i) {
			index[i] = i;
		}
		
		std::sort(index.begin(), index.end(), 
			[&](const int& a, const int& b) {
				return (eigen_ia.data()[a] < eigen_ia.data()[b]);
		});
		
		std::vector<dbcsr::shared_matrix<double>> dav_guess(m_nroots);
			
		// generate the guesses
		
		auto o = m_hfwfn->mol()->dims().oa();
		auto v = m_hfwfn->mol()->dims().va();
		auto b = m_hfwfn->mol()->dims().b();
		
		for (int i = 0; i != m_nroots; ++i) {
			
			LOG.os<>("Guess ", i, '\n');
			
			Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(nocc,nvir);
			mat.data()[index[i]] = 1.0;
			
			std::string name = "guess_" + std::to_string(i);
			
			auto guessmat = dbcsr::eigen_to_matrix(mat, m_world, name,
				o, v, dbcsr::type::no_symmetry);
			
			dav_guess[i] = guessmat;
			
			//dbcsr::print(*guessmat);
			
		}
		
		MVP* mvfacptr = new MVP_ao_ri_adc1(m_world, m_hfwfn->mol(), 
			m_opt, m_reg, epso, epsv);
		std::shared_ptr<MVP> mvfac(mvfacptr);
		
		mvfac->init();
		//auto out = mvfac->compute(dav_guess[0], 0.3); 
		
		//double t = out->dot(*dav_guess[0]);
		//std::cout << "DOT: " << t << std::endl;
		
		//dbcsr::print(*dav_guess[0]);
		
		//exit(0); 
		
		math::davidson<MVP> dav(m_world.comm(), LOG.global_plev());
		
		dav.set_factory(mvfac);
		dav.set_diag(m_d_ov);
		dav.pseudo(false);
		dav.conv(1e-6);
		dav.maxiter(100);	
		
		int nroots = m_opt.get<int>("nroots", ADC_NROOTS);
		
		dav.compute(dav_guess, nroots);
		
		LOG.os<>("Excitation energy of state nr. ", m_nroots, ": ", dav.eigval(), '\n');
		
		auto rvecs = dav.ritz_vectors();
		auto vec_k = rvecs[m_nroots-1];
		
		auto c_bo = m_reg.get_matrix<double>("c_bo");
		auto c_bv = m_reg.get_matrix<double>("c_bv");
		auto s_bb = m_reg.get_matrix<double>("s_bb");
		
		LOG.os<>("AOs\n");
		
		auto v_ht = dbcsr::create<double>()
			.name("ht")
			.set_world(m_world)
			.row_blk_sizes(o)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
			
		auto v_bb = dbcsr::create<double>()
			.name("v_bb")
			.set_world(m_world)
			.row_blk_sizes(b)
			.col_blk_sizes(b)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
		
		
		dbcsr::multiply('N', 'T', *vec_k, *c_bv, *v_ht).perform();
		dbcsr::multiply('N', 'N', *c_bo, *v_ht, *v_bb).perform();
		
		auto p = m_hfwfn->po_bb_A();
		
		v_bb->filter(1e-6);
		//p->filter(1e-6);
		
		dbcsr::print(*v_bb);
		
		LOG.os<>("Occupation: ", v_bb->occupation() * 100, "%\n");
		LOG.os<>("Compared to ", p->occupation()*100, "% for density matrix.\n");
		
		// list of exitations
		
		int nbas = m_hfwfn->mol()->c_basis()->nbf();
		
		auto v_ia_eigen = dbcsr::matrix_to_eigen(v_bb);
		
		auto get_lists = [&](std::vector<bool>& occ_list,
			std::vector<bool>& vir_list, double t)
		{
		
			for (int i = 0; i != nbas; ++i) {
				for (int a = 0; a != nbas; ++a) {
					if (fabs(v_ia_eigen(i,a)) > t) {
						occ_list[i] = true;
						vir_list[a] = true;
						//LOG.os<>(i, " -> ", a, " ", v_ia_eigen(i,a), '\n');
					}
				}
			}
			
		};
		
		auto count = [](std::vector<bool>& v) {
			int ntot = 0;
			for (auto b : v) {
				ntot += (b) ? 1 : 0;
			}
			return ntot;
		};
		
		std::vector<std::vector<bool>> occs(8, std::vector<bool>(nbas));
		std::vector<std::vector<bool>> virs(8, std::vector<bool>(nbas));
			
		for (int n = 0; n != 8; ++n) {
			
			double t = pow(10.0, -n);
			
			get_lists(occs[n], virs[n], t);
			int no = count(occs[n]);
			int nv = count(virs[n]);
			
			LOG.os<>("T: ", t, " with ", no, "/", nbas, " and ", nv, "/", nbas, '\n');
			
		}
		
		// Which blocks are absent?
		std::vector<bool> blk_list(b.size());
		
		dbcsr::iterator<double> iter(*v_bb);
		iter.start();
		
		while (iter.blocks_left()) {
			
			iter.next_block();
			
			blk_list[iter.row()] = true;
			blk_list[iter.col()] = true;
			
		}
		
		iter.stop();
			
		int nblk = 0;
		
		for (auto b : blk_list) {
			if (b) nblk++;
		}
		
		LOG.os<>("Basis blocks: ", nblk, " out of ", b.size(), '\n'); 
		
		exit(0);
		
}

void adcmod::analyze_sparsity(dbcsr::shared_matrix<double> u_ia, 
		dbcsr::shared_matrix<double> c_loc_o, dbcsr::shared_matrix<double> u_loc_o,
		dbcsr::shared_matrix<double> c_loc_v, dbcsr::shared_matrix<double> u_loc_v)
{
	/* Function to analyse sparsity f a trial vector u in a certain basis
	 * if u_loc's are not given -> canonical MOs are assumed
	 * if c_loc's are not given -> AOs are assumed
	 * 
	 * Outputs:
	 * 	- list of excitations o -> v for certain thresholds
	 * 	(1e-3,1e-4,1e-5 ...)
	 * 	- list & number of occ MOs and vir MOs involved
	 * 	- list & number of AOs involved
	 * 	- sparsity of u_ia_loc (different blk thresholds)
	 */
	
	auto b = c_loc_o->row_blk_sizes();
	auto o = u_ia->row_blk_sizes();
	auto oloc = c_loc_o->col_blk_sizes();
	auto vloc = c_loc_v->col_blk_sizes();
	auto w = c_loc_o->get_world();
	
	int nocc = c_loc_o->nfullcols_total();
	int nvir = c_loc_v->nfullcols_total();
	int nbas = c_loc_o->nfullrows_total();
	
	LOG.os<>("Number of occ./vir. MOs: ", nocc, " / ", nvir, '\n');
	
	decltype(u_ia) v_ia;
	
	// transform u_ia
	if (u_loc_o && u_loc_v) {
		
		auto v_ht = dbcsr::create<double>()
			.name("ht")
			.set_world(w)
			.row_blk_sizes(o)
			.col_blk_sizes(vloc)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
			
		v_ia = dbcsr::create<double>()
			.name("ht")
			.set_world(w)
			.row_blk_sizes(oloc)
			.col_blk_sizes(vloc)
			.matrix_type(dbcsr::type::no_symmetry)
			.get();
		
		std::cout << "1" << std::endl;
		dbcsr::multiply('N', 'T', *u_ia, *u_loc_v, *v_ht).perform();
		std::cout << "2" << std::endl;
		dbcsr::multiply('N', 'N', *u_loc_o, *v_ht, *v_ia).perform();
	} else {
		v_ia = u_ia;
	} 
	
	dbcsr::print(*v_ia);
		
	// list of exitations
	
	auto v_ia_eigen = dbcsr::matrix_to_eigen(v_ia);
	
	auto get_lists = [&](std::vector<bool>& occ_list,
		std::vector<bool>& vir_list, double t)
	{
	
		for (int i = 0; i != nocc; ++i) {
			for (int a = 0; a != nvir; ++a) {
				if (fabs(v_ia_eigen(i,a)) > t) {
					occ_list[i] = true;
					vir_list[a] = true;
					//LOG.os<>(i, " -> ", a, " ", v_ia_eigen(i,a), '\n');
				}
			}
		}
		
	};
	
	auto count = [](std::vector<bool>& v) {
		int ntot = 0;
		for (auto b : v) {
			ntot += (b) ? 1 : 0;
		}
		return ntot;
	};
	
	std::vector<std::vector<bool>> occs(8, std::vector<bool>(nocc));
	std::vector<std::vector<bool>> virs(8, std::vector<bool>(nvir));
		
	for (int n = 0; n != 8; ++n) {
		
		double t = pow(10.0, -n);
		
		get_lists(occs[n], virs[n], t);
		int no = count(occs[n]);
		int nv = count(virs[n]);
		
		LOG.os<>("T: ", t, " with ", no, "/", nocc, " and ", nv, "/", nvir, '\n');
		
	}
	
}

}
