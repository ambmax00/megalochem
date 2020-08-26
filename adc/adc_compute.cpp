#include "adc/adcmod.h"
#include "adc/adc_mvp.h"
#include "math/solvers/davidson.h"

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
		
		std::cout << eigen_ia << std::endl;
		
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
		
		for (int i = 0; i != m_nroots; ++i) {
			
			LOG.os<>("Guess ", i, '\n');
			
			Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(nocc,nvir);
			mat.data()[index[i]] = 1.0;
			
			std::string name = "guess_" + std::to_string(i);
			
			auto guessmat = dbcsr::eigen_to_matrix(mat, m_world, name,
				o, v, dbcsr::type::no_symmetry);
			
			dav_guess[i] = guessmat;
			
			dbcsr::print(*guessmat);
			
		}
		
		MVP* mvfacptr = new MVP_ao_ri_adc2(m_world, m_hfwfn->mol(), 
			m_opt, m_reg, epso, epsv);
		std::shared_ptr<MVP> mvfac(mvfacptr);
		
		mvfac->init();
		auto out = mvfac->compute(dav_guess[0], 0.0); 
		
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
		
		auto vec_k_e = dbcsr::matrix_to_eigen(vec_k);
		double thresh = 1e-6;
		
		std::vector<int> occs, virs;
		
		LOG.os<>("State involves the orbital(s):\n");
		for (int iocc = 0; iocc != vec_k_e.rows(); ++iocc) {
			for (int ivir = 0; ivir != vec_k_e.cols(); ++ivir) {
				if (fabs(vec_k_e(iocc,ivir)) >= thresh) {
					LOG.os<>(iocc, " -> ", ivir, " : ", vec_k_e(iocc,ivir), '\n');
					occs.push_back(iocc);
					virs.push_back(ivir);
				}
			}
		}
		
		auto get_unique = [](std::vector<int>& vec) {
			std::sort(vec.begin(), vec.end());
			vec.erase(unique( vec.begin(), vec.end() ), vec.end());
		};
		
		// which AOs are involved?
		get_unique(occs);
		get_unique(virs);
		
		auto c_bo = m_hfwfn->c_bo_A();
		auto c_bv = m_hfwfn->c_bv_A();
		
		auto c_bo_e = dbcsr::matrix_to_eigen(c_bo);
		auto c_bv_e = dbcsr::matrix_to_eigen(c_bv);
		
		// OCC
		std::vector<int> aolist;
		
		auto get_aos = [&](std::vector<int>& list, std::vector<int> mlist,
			Eigen::MatrixXd& c_bm) {
		
			for (auto im : mlist) {
				std::cout << "CHECKING " << im << std::endl;
				for (int ibas = 0; ibas != c_bm.rows(); ++ibas) {
					if (fabs(c_bm(ibas,im)) >= thresh) {
						list.push_back(ibas);
						std::cout << ibas << " ";
					}
				}
				std::cout << std::endl;
			}
		};
		
		std::cout << "DO OCCS" << std::endl;
		get_aos(aolist, occs, c_bo_e);
		std::cout << "DO VIRS" << std::endl;
		get_aos(aolist, virs, c_bv_e);
		
		get_unique(aolist);
		
		LOG.os<>("AOs involved: \n");
		for (auto l : aolist) {
			LOG.os<>(l, " ");
		} LOG.os<>('\n');
		
		auto get_shell_to_block = [&](std::vector<int>& list, std::vector<int>& offs) {
			
			vec<int> ao_to_block;
			
			for (auto l : list) {
			
				int a = -1;
			
				for (int ioff = 0; ioff != offs.size()-1; ++ioff) {
					if (l >= offs[ioff] && l < offs[ioff+1]) {
						a = ioff;
						ao_to_block.push_back(ioff);
						break;
					}
				}
				
				if (a == -1) ao_to_block.push_back(offs.size()-1);
				
			}
			
			return ao_to_block;
			
		};
		
		auto boff = c_bo->row_blk_offsets();
		
		auto aoblk = get_shell_to_block(aolist, boff);
		get_unique(aoblk);
		
		auto atoms = m_hfwfn->mol()->atoms();
		auto blkatom = m_hfwfn->mol()->c_basis().block_to_atom(atoms);
		
		LOG.os<>("ATOM CENTRES:\n");
		for (auto a : aoblk) {
			LOG.os<>(blkatom[a], " ");
		} LOG.os<>('\n');
		
}

}
