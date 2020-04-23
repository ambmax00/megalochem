#include "adc/adcmod.h"
#include "adc/adc_ri_u1.h"
#include "tensor/dbcsr_conversions.h"
#include "math/solvers/davidson.h"

#include <algorithm>

#include <fstream>
#include <sstream>

namespace adc {

void adcmod::compute() {

		// test MO values
		
		/*
		dbcsr::pgrid<4> grid4a({.comm = m_comm});
		vec<vec<int>> m_sizes = {osizes[1],osizes[1],vsizes[1],vsizes[1]};
		dbcsr::stensor<4> mmmm = dbcsr::make_stensor<4>({.name = "mmmm", .pgridN = grid4a, .map1 = {0,1}, .map2 = {2,3}, .blk_sizes = m_sizes});
		
		dbcsr::einsum<3,3,4>({.x = "Xij, Xkl -> ijkl", .t1 = *d_xoo, .t2 = *d_xvv, .t3 = *mmmm});
		
		dbcsr::print(*mmmm);*/
		
		
		// compute amplitudes, if applicable
		
		dbcsr::pgrid<2> grid2(m_comm);
		
		int nocc = m_hfwfn->mol()->nocc_alpha();
		int nvir = m_hfwfn->mol()->nvir_alpha();
		
		std::cout << "BEGIN" << std::endl;
		
		mo_amplitudes();
		
		//exit(0);
		
		mo_compute_diag();
		
		// now order it : there is probably a better way to do it
		auto eigen_ia = dbcsr::tensor_to_eigen(*m_mo.d_ov);
		
		std::vector<int> index(eigen_ia.size(), 0);
		for (int i = 0; i!= index.size(); ++i) {
			index[i] = i;
		}
		
		std::sort(index.begin(), index.end(), 
			[&](const int& a, const int& b) {
				return (eigen_ia.data()[a] < eigen_ia.data()[b]);
		});
		
		for (auto i : index) {
			std::cout << i << " ";
		} std::cout << std::endl;
		
		std::vector<dbcsr::stensor<2>> dav_guess(m_nroots);
			
		// generate the guesses
		
		vec<int> map1 = {0};
		vec<int> map2 = {1};
		arrvec<int,2> blksizes = {m_dims.o, m_dims.v};
		
		for (int i = 0; i != m_nroots; ++i) {
			
			std::cout << "on guess: " << i << std::endl;
			
			Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(nocc,nvir);
			mat.data()[index[i]] = 1.0;
			
			std::string name = "guess_" + std::to_string(i);
			std::cout << name << std::endl;
			
			auto ten = dbcsr::eigen_to_tensor(mat, name, grid2, map1, map2, blksizes);
			auto sten = ten.get_stensor();
			
			dav_guess[i] = sten;
			
			dbcsr::print(*sten);
			
		}
		
		ri_adc1_u1 ri_adc1(m_mo.eps_o, m_mo.eps_v, m_mo.b_xoo, m_mo.b_xov, m_mo.b_xvv); 
		
		math::davidson<ri_adc1_u1> dav = math::davidson<ri_adc1_u1>::create()
			.factory(ri_adc1).diag(m_mo.d_ov);
		
		dav.compute(dav_guess, m_nroots);
		
		auto rvs = dav.ritz_vectors();
		
		auto rn = rvs[m_nroots - 1];
		double omega = dav.eigval();

		// compute ADC2
		if (m_method == 0) {
			
			ri_adc2_diis_u1 ri_adc2(m_mo.eps_o, m_mo.eps_v, m_mo.b_xoo, m_mo.b_xov, m_mo.b_xvv, m_mo.t_ovov);
		
			math::modified_davidson<ri_adc2_diis_u1> dav
				= math::modified_davidson<ri_adc2_diis_u1>::create()
				.factory(ri_adc2).diag(m_mo.d_ov);
				
			dav.compute(rvs, m_nroots, omega);
			
		} else {
			
			std::cout << "SOS" << std::endl;
			//sos_ri_adc2_diis_u1 ri_adc2(m_mo.eps_o, m_mo.eps_v, m_mo.b_xoo, m_mo.b_xov, m_mo.b_xvv, m_mo.t_ovov);
		
			//math::modified_davidson<sos_ri_adc2_diis_u1> dav({.factory = ri_adc2, .diag = m_mo.d_ov});
			//dav.compute(rvs, m_nroots, omega);
			
		}
		
}

}
