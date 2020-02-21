#include <vector>
#include <stdexcept>
#include "ints/aofactory.h"
#include "ints/integrals.h"
#include "ints/screening.h"
#include "ints/registry.h"
#include "utils/pool.h"
#include "utils/params.hpp"
#include "math/linalg/inverse.h"
#include <libint2.hpp>

#include <iostream>
#include <limits>

namespace ints {

template <int N>
dbcsr::stensor<N,double> aofactory::compute(aofac_params&& p) {
		
		std::string Op = *p.op;
		std::string bis = *p.bas;
		std::string intname;
		
		libint2::initialize();
		
		// ======= process operator =========== 
		libint2::Operator libOp = libint2::Operator::invalid;
		libint2::BraKet libBraKet = libint2::BraKet::invalid;
		
		if (Op == "coulomb") {
			libOp = libint2::Operator::coulomb;
			intname = "i_";
		} else if (Op == "overlap") {
			libOp = libint2::Operator::overlap;
			intname = "s_";
		} else if (Op == "kinetic") {
			libOp = libint2::Operator::kinetic;
			intname = "t_";
		} else if (Op == "nuclear") {
			libOp = libint2::Operator::nuclear;
			intname = "v_";
		} else if (Op == "erfc_coulomb") {
			libOp = libint2::Operator::erfc_coulomb;
			intname = "erfc_";
		}
		
		if (libOp == libint2::Operator::invalid) 
			throw std::runtime_error("Invalid operator: "+Op);
			
		// ======== process basis info =============
		vec<desc::cluster_basis> basvec;
		
		//std::cout << "A1" << std::endl;
		
		desc::cluster_basis c_bas = m_mol.c_basis();
		optional<desc::cluster_basis,val> x_bas = m_mol.c_dfbasis();
		
		//if (x_bas) std::cout << "ITS HERE IN INTS" << std::endl;
		
		//std::cout << "A3" << std::endl;
		
		if (bis == "bb") { 
			basvec = {c_bas, c_bas};
		} else if (bis == "xx") {
			basvec = {*x_bas, *x_bas};
			libBraKet = libint2::BraKet::xs_xs;
		} else if (bis == "xbb") {
			basvec = {*x_bas, c_bas, c_bas};
			libBraKet = libint2::BraKet::xs_xx;
		} else if (bis == "bbbb") {
			std::cout << "4" << std::endl;
			basvec = {c_bas, c_bas, c_bas, c_bas};
			libBraKet = libint2::BraKet::xx_xx;
		} else {
			throw std::runtime_error("Unsupported basis set specifications: "+bis);
		}
		
		intname += bis;
		
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
		if (Op == "nuclear") {
			eng.set_params(make_point_charges(m_mol.atoms())); 
		} else if (Op == "erfc_coulomb") {
			double omega = 0.1; //*ctx.get<double>("INT/omega");
			eng.set_params(omega);
		}
			
		util::ShrPool<libint2::Engine> eng_pool = util::make_pool<libint2::Engine>(eng);
		
		//if (Op == "coulomb") {
				
			//libint2::Engine screen(libint2::Operator::coulomb, max_nprim, max_l, 0, std::numeric_limits<double>::epsilon());
			//			screen.set(libint2::BraKet::xx_xx);
						
			//util::ShrPool<libint2::Engine> screen_pool = util::make_pool<libint2::Engine>(screen);
				
			//m_2e_ints_zmat = Zmat(m_comm, m_mol, screen_pool, "schwarz");
				
			//m_2e_ints_zmat->compute();
				
		//}
		
		// ======= porcess extended input ==========
		
		/*
		if (p.ext && N == 4) throw std::runtime_error("Aofactory: no extended input possible for 2e ints.");
		
		if (p.ext) {
			if ((N == 2) && (*p.ext != "^-1" || *p.ext != "^-1/2"))
				throw std::runtime_error("Aofactory: unknown option for extended input (N = 2)");
			if ((N == 3) && (*p.ext != "(P|Q)^-1" || *p.ext != "(P|Q)^-1/2"))
				throw std::runtime_error("Aofactory: unknown option for extended input (N = 2)");
		}
		*/
		//std::cout << "OUT" << std::endl;
		out = integrals<N>({.comm = m_comm, .engine = eng_pool, 
			.basvec = basvec, .bra = m_2e_ints_zmat, .ket = m_2e_ints_zmat, 
			.name = intname, .map1 = *p.map1, .map2 = *p.map2});
		
		/*
		if (p.ext && N == 2) {
			
			if (*p.ext == "^-1") {
				out = math::eigen_inverse(*out, *p.name);
			} else {
				out = math::eigen_sqrt_inverse(*out,*p.name);
			}
			
		} else if (p.ext && N == 3) {
			
			if (*p.ext == "(P|Q)^-1") {
				auto metric = this->compute({.op = p.op, .bas = "xx", .name = "XX", .map1 = {0}, .map2 = {1}y
				out = math::eigen_inverse(*out, *p.name);
			} else {
				out = math::eigen_sqrt_inverse(*out,*p.name);
			}
			
			
		*/	
		
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

//forward declarations
template dbcsr::stensor<2,double> aofactory::compute(aofac_params&& p);
template dbcsr::stensor<3,double> aofactory::compute(aofac_params&& p);
template dbcsr::stensor<4,double> aofactory::compute(aofac_params&& p);


} // end namespace ints
