#include <vector>
#include <stdexcept>
#include "ints/aofactory.h"
#include "ints/integrals.h"
#include "ints/registry.h"
#include "utils/pool.h"
#include "math/linalg/inverse.h"
#include <libint2.hpp>

#include <iostream>
#include <limits>

namespace ints {

template <int N>
dbcsr::stensor<N,double> aofactory::compute() {
		
		std::string intname;
		
		libint2::initialize();
		
		// ======= process operator =========== 
		libint2::Operator libOp = libint2::Operator::invalid;
		libint2::BraKet libBraKet = libint2::BraKet::invalid;
		
		if (*c_op == "coulomb") {
			libOp = libint2::Operator::coulomb;
			intname = "i_";
		} else if (*c_op == "overlap") {
			libOp = libint2::Operator::overlap;
			intname = "s_";
		} else if (*c_op == "kinetic") {
			libOp = libint2::Operator::kinetic;
			intname = "t_";
		} else if (*c_op == "nuclear") {
			libOp = libint2::Operator::nuclear;
			intname = "v_";
		} else if (*c_op == "erfc_coulomb") {
			libOp = libint2::Operator::erfc_coulomb;
			intname = "erfc_";
		}
		
		if (libOp == libint2::Operator::invalid) 
			throw std::runtime_error("Invalid operator: "+*c_op);
			
		// ======== process basis info =============
		vec<desc::cluster_basis> basvec;
		
		//std::cout << "A1" << std::endl;
		
		desc::cluster_basis c_bas = m_mol.c_basis();
		optional<desc::cluster_basis,val> x_bas = m_mol.c_dfbasis();
		
		//if (x_bas) std::cout << "ITS HERE IN INTS" << std::endl;
		
		//std::cout << "A3" << std::endl;
		
		if (*c_dim == "bb") { 
			basvec = {c_bas, c_bas};
		} else if (*c_dim == "xx") {
			basvec = {*x_bas, *x_bas};
			libBraKet = libint2::BraKet::xs_xs;
		} else if (*c_dim == "xbb") {
			basvec = {*x_bas, c_bas, c_bas};
			libBraKet = libint2::BraKet::xs_xx;
		} else if (*c_dim == "bbbb") {
			std::cout << "4" << std::endl;
			basvec = {c_bas, c_bas, c_bas, c_bas};
			libBraKet = libint2::BraKet::xx_xx;
		} else {
			throw std::runtime_error("Unsupported basis set specifications: "+ *c_dim);
		}
		
		intname += *c_dim;
		
		registry INT_REGISTRY;
		dbcsr::stensor<N> out = INT_REGISTRY.get<N>(m_mol.name(), intname);
		
		if (out) {
			std::cout << "PRESENT: " << m_mol.name() << " " << intname << std::endl;
			return out;
		}
		
		size_t max_nprim = 0;
		int max_l = 0;
		
		for (int i = 0; i != basvec.size(); ++i) {
			max_nprim = std::max(basvec[i].max_nprim(), max_nprim);
			max_l = std::max(basvec[i].max_l(), max_l);
		}
		
		//std::cout << "MAX " << max_nprim << " " << max_l << std::endl;
		
		libint2::Engine eng(libOp, max_nprim, max_l, 0, std::numeric_limits<double>::epsilon());
			
		if (libBraKet != libint2::BraKet::invalid) {
			eng.set(libBraKet);
		}
		
		// OPTIONS
		if (*c_op == "nuclear") {
			eng.set_params(make_point_charges(m_mol.atoms())); 
		} else if (*c_op == "erfc_coulomb") {
			double omega = 0.1; //*ctx.get<double>("INT/omega");
			eng.set_params(omega);
		}
			
		util::ShrPool<libint2::Engine> eng_pool = util::make_pool<libint2::Engine>(eng);
		
		dbcsr::pgrid<N> gridN(m_comm);
		arrvec<int,N> blksizes;
		
		for (int i = 0; i != N; ++i) {
			blksizes[i] = basvec[i].cluster_sizes();
			for (auto x : blksizes[i]) std::cout << x << " ";
			std::cout << std::endl;
		}
		
		dbcsr::tensor<N> t_ints = typename dbcsr::tensor<N>::create().name(intname)
			.ngrid(gridN).map1(*c_map1).map2(*c_map2).blk_sizes(blksizes);
		
		calc_ints(t_ints,eng_pool,basvec);
		out = t_ints.get_stensor();
		
		INT_REGISTRY.insert<N>(m_mol.name(), intname, out);
		
		libint2::finalize();
		
		return out;
		
}

dbcsr::stensor<2> aofactory::invert(dbcsr::stensor<2>& in, int order) {
	
	//if (method != 0) throw std::runtime_error("No other method for inverting has been implemented.");
	if (!(order == 1 || order == 2)) throw std::runtime_error("Wrong order for inverting.");
	
	dbcsr::stensor<2> out;
	
	std::string intname = in->name();
	
	if (order == 1) intname += "^-1";
	if (order == 2) intname += "^-1/2";
	
	registry INT_REGISTRY;
	out = INT_REGISTRY.get<2>(m_mol.name(), intname);
	
	if (out) return out;
	
	if (order == 1) {
		out = math::eigen_inverse(in,intname);
	} else {
		out = math::eigen_sqrt_inverse(in,intname);
	}
	
	INT_REGISTRY.insert<2>(m_mol.name(), intname, out);
	
	return out;
	
}

template dbcsr::stensor<2,double> aofactory::compute();
template dbcsr::stensor<3,double> aofactory::compute();
template dbcsr::stensor<4,double> aofactory::compute();

} // end namespace ints
