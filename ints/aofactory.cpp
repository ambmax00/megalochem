#include <vector>
#include <stdexcept>
#include "ints/aofactory.h"
#include "ints/integrals.h"
#include "utils/pool.h"
#include <libint2.hpp>

#include <iostream>
#include <limits>

namespace ints {
	
class aofactory::impl {
private:

	desc::molecule& m_mol;
	dbcsr::world m_world;
	
	libint2::Operator m_Op = libint2::Operator::invalid;
	libint2::BraKet m_BraKet = libint2::BraKet::invalid;

public:
	
	vec<desc::cluster_basis> m_basvec;
	util::ShrPool<libint2::Engine> m_eng_pool;
	
	std::string m_intname = "";

public:
	
	impl(desc::molecule& mol, dbcsr::world w) :
		m_mol(mol), m_world(w) { init(); }
	
	void init() {
		libint2::initialize();
	}
	
	void set_operator(std::string op) {
		
		if (op == "coulomb") {
			m_Op = libint2::Operator::coulomb;
			m_intname = "i_";
		} else if (op == "overlap") {
			m_Op = libint2::Operator::overlap;
			m_intname = "s_";
		} else if (op == "kinetic") {
			m_Op = libint2::Operator::kinetic;
			m_intname = "t_";
		} else if (op == "nuclear") {
			m_Op = libint2::Operator::nuclear;
			m_intname = "v_";
		} else if (op == "erfc_coulomb") {
			m_Op = libint2::Operator::erfc_coulomb;
			m_intname = "erfc_";
		}
		
		if (m_Op == libint2::Operator::invalid) 
			throw std::runtime_error("Invalid operator: "+ op);
	}
	
	void set_braket(std::string dim) {
		
		auto cbas = m_mol.c_basis();
		auto xbas = m_mol.c_dfbasis();
		
		if (dim == "bb") { 
			m_basvec = {cbas,cbas};
		} else if (dim == "xx") {
			m_basvec = {*xbas, *xbas};
			m_BraKet = libint2::BraKet::xs_xs;
		} else if (dim == "xbb") {
			m_basvec = {*xbas, cbas, cbas};
			m_BraKet = libint2::BraKet::xs_xx;
		} else if (dim == "bbbb") {
			m_basvec = {cbas, cbas, cbas, cbas};
			m_BraKet = libint2::BraKet::xx_xx;
		} else {
			throw std::runtime_error("Unsupported basis set specifications: "+ dim);
		}
		
		m_intname += dim;
		
	}
	
	void setup_calc() {
		
		size_t max_nprim = 0;
		int max_l = 0;
		
		for (int i = 0; i != m_basvec.size(); ++i) {
			max_nprim = std::max(m_basvec[i].max_nprim(), max_nprim);
			max_l = std::max(m_basvec[i].max_l(), max_l);
		}
		
		libint2::Engine eng(m_Op, max_nprim, max_l, 0, std::numeric_limits<double>::epsilon());
			
		if (m_Op == libint2::Operator::nuclear) {
			eng.set_params(make_point_charges(m_mol.atoms())); 
		} else if (m_Op == libint2::Operator::erfc_coulomb) {
			double omega = 0.1; //*ctx.get<double>("INT/omega");
			eng.set_params(omega);
		}
		
		eng.set(m_BraKet);
			
		m_eng_pool = util::make_pool<libint2::Engine>(eng);
		
	}
	
	void finalize() {
		libint2::finalize();
	}
	
	dbcsr::smatrix<double> compute() {
		
		auto rowsizes = m_basvec[0].cluster_sizes();
		auto colsizes = m_basvec[1].cluster_sizes();
		
		dbcsr::mat_d m_ints = dbcsr::mat_d::create()
			.name(m_intname)
			.set_world(m_world)
			.row_blk_sizes(rowsizes).col_blk_sizes(colsizes)
			.type(dbcsr_type_symmetric);
		
		calc_ints(m_ints, m_eng_pool, m_basvec);
		return m_ints.get_smatrix();
		
	}
		
	
#define prototype(n) \
	dbcsr::stensor<n,double> compute_##n(vec<int>& map1, vec<int>& map2) { \
		dbcsr::pgrid<n> grid(m_world.comm()); \
		arrvec<int,n> blksizes; \
		for (int i = 0; i != n; ++i) { \
			blksizes[i] = m_basvec[i].cluster_sizes(); \
		} \
		dbcsr::tensor<n> t_ints = dbcsr::tensor<n>::create().name(m_intname) \
			.ngrid(grid).map1(map1).map2(map2).blk_sizes(blksizes); \
		calc_ints(t_ints, m_eng_pool, m_basvec); \
		return t_ints.get_stensor(); \
	}
	
	prototype(2)
	prototype(3)
	prototype(4)
	
};

aofactory& aofactory::op(std::string op) {
		pimpl->set_operator(op);
		return *this;
	}
aofactory& aofactory::dim(std::string dim) { 
		pimpl->set_braket(dim);
		return *this; 
}

aofactory::aofactory(desc::molecule& mol, dbcsr::world& w) : pimpl(new impl(mol, w))  {}
aofactory::~aofactory() { delete pimpl; };

dbcsr::smatrix<double> aofactory::compute() {
	pimpl->setup_calc();
	return pimpl->compute();
}

#define prototype2(n) \
dbcsr::stensor<n,double> aofactory::compute_##n(std::vector<int> map1, std::vector<int> map2) {\
	pimpl->setup_calc();\
	return pimpl->compute_##n(map1,map2);\
}
		
prototype2(2)
prototype2(3)
prototype2(4)
/*
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
*/

} // end namespace ints
