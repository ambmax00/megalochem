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
	
	inline static std::map<std::string,dbcsr::smat_d> m_matrix_registry;
	inline static std::map<std::string,dbcsr::stensor2_d> m_tensor2d_registry;
	inline static std::map<std::string,dbcsr::stensor3_d> m_tensor3d_registry;
	inline static std::map<std::string,dbcsr::stensor4_d> m_tensor4d_registry;

public:
	
	vec<desc::cluster_basis> m_basvec;
	util::ShrPool<libint2::Engine> m_eng_pool;
	
	std::string m_opname;
	std::string m_dimname;
	std::string m_screenname;
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
			m_opname = "i_";
		} else if (op == "overlap") {
			m_Op = libint2::Operator::overlap;
			m_opname = "s_";
		} else if (op == "kinetic") {
			m_Op = libint2::Operator::kinetic;
			m_opname = "t_";
		} else if (op == "nuclear") {
			m_Op = libint2::Operator::nuclear;
			m_opname = "v_";
		} else if (op == "erfc_coulomb") {
			m_Op = libint2::Operator::erfc_coulomb;
			m_opname = "erfc_";
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
		
		m_dimname = dim;
		
	}
	
	void set_screen(std::string screen) {
		
		if (screen == "schwarz") {
			m_screenname = screen;
		} else {
			throw std::runtime_error("Unknown screening procedure.");
		}
		
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
		
		m_intname = m_opname + "_" + m_dimname + "_" + m_screenname;
		
	}
	
	void finalize() {
		libint2::finalize();
	}
	
	dbcsr::smatrix<double> compute() {
		
		std::string rname = m_mol.name() + "_" + m_intname;
		
		if (m_matrix_registry.find(rname) != m_matrix_registry.end())
			return m_matrix_registry[rname];
		
		auto rowsizes = m_basvec[0].cluster_sizes();
		auto colsizes = m_basvec[1].cluster_sizes();
		
		dbcsr::mat_d m_ints = dbcsr::mat_d::create()
			.name(m_intname)
			.set_world(m_world)
			.row_blk_sizes(rowsizes).col_blk_sizes(colsizes)
			.type(dbcsr_type_symmetric);
			
		// reserve symmtric blocks
		int nblks = m_ints.nblkrows_total();
		
		vec<int> resrows, rescols;
		
		for (int i = 0; i != nblks; ++i) {
			for (int j = 0; j != nblks; ++j) {
				if (m_ints.proc(i,j) == m_world.rank() && i <= j) {
					resrows.push_back(i);
					rescols.push_back(j);
				}
			}
		}
		
		m_ints.reserve_blocks(resrows,rescols);
	
		if (m_screenname == "schwarz" && m_dimname == "bb") {
			calc_ints_schwarz_mn(m_ints,m_eng_pool,m_basvec);
		} else if (m_screenname == "schwarz" && m_dimname == "xx") {
			calc_ints_schwarz_xy(m_ints,m_eng_pool,m_basvec);
		} else {
			calc_ints(m_ints, m_eng_pool, m_basvec);
		}
		
		auto m_ints_out = m_ints.get_smatrix();
		
		m_matrix_registry[rname] = m_ints_out;
		
		return m_ints_out;
		
	}
		
	dbcsr::stensor<2,double> compute_2(vec<int>& map1, vec<int>& map2) { 
		dbcsr::pgrid<2> grid(m_world.comm()); 
		arrvec<int,2> blksizes;
		for (int i = 0; i != 2; ++i) { 
			blksizes[i] = m_basvec[i].cluster_sizes(); 
		} 
		dbcsr::tensor<2> t_ints = dbcsr::tensor<2>::create().name(m_intname) 
			.ngrid(grid).map1(map1).map2(map2).blk_sizes(blksizes); 
			
		t_ints.reserve_all();
			
		calc_ints(t_ints, m_eng_pool, m_basvec); 
		return t_ints.get_stensor();
	}
	
	dbcsr::stensor<3,double> compute_3(vec<int>& map1, vec<int>& map2, eigen_smat_f bra, eigen_smat_f ket) { 
		dbcsr::pgrid<3> grid(m_world.comm()); 
		arrvec<int,3> blksizes; 
		for (int i = 0; i != 3; ++i) { 
			blksizes[i] = m_basvec[i].cluster_sizes(); 
		} 
		
		dbcsr::tensor<3> t_ints = dbcsr::tensor<3>::create().name(m_intname) 
			.ngrid(grid).map1(map1).map2(map2).blk_sizes(blksizes); 
			
		if (bra && ket) {
			
			t_ints.reserve_all();
			
		} else {
		
			t_ints.reserve_all();
			
		}	
		
		calc_ints(t_ints, m_eng_pool, m_basvec); 
		
		// set up sparsity 
		
		return t_ints.get_stensor();
	}
	
	dbcsr::stensor<4,double> compute_4(vec<int>& map1, vec<int>& map2) { 
		dbcsr::pgrid<4> grid(m_world.comm()); 
		arrvec<int,4> blksizes; 
		for (int i = 0; i != 4; ++i) { 
			blksizes[i] = m_basvec[i].cluster_sizes(); 
		} 
		
		dbcsr::tensor<4> t_ints = dbcsr::tensor<4>::create().name(m_intname) 
			.ngrid(grid).map1(map1).map2(map2).blk_sizes(blksizes); 
			
		t_ints.reserve_all();
			
		calc_ints(t_ints, m_eng_pool, m_basvec); 
		return t_ints.get_stensor();
	}
	
};

aofactory& aofactory::op(std::string i_op) {
		pimpl->set_operator(i_op);
		return *this;
	}
aofactory& aofactory::dim(std::string i_dim) { 
		pimpl->set_braket(i_dim);
		return *this; 
}
aofactory& aofactory::screen(std::string i_screen) { 
		pimpl->set_screen(i_screen);
		return *this; 
}

aofactory::aofactory(desc::molecule& mol, dbcsr::world& w) : pimpl(new impl(mol, w))  {}
aofactory::~aofactory() { delete pimpl; };

dbcsr::smatrix<double> aofactory::compute() {
	pimpl->setup_calc();
	return pimpl->compute();
}

dbcsr::stensor<2,double> aofactory::compute_2(std::vector<int> map1, std::vector<int> map2) {
	pimpl->setup_calc();
	return pimpl->compute_2(map1,map2);
}

dbcsr::stensor<3,double> aofactory::compute_3(std::vector<int> map1, std::vector<int> map2, eigen_smat_f bra, eigen_smat_f ket) {
	pimpl->setup_calc();
	return pimpl->compute_3(map1,map2,bra,ket);
}

dbcsr::stensor<4,double> aofactory::compute_4(std::vector<int> map1, std::vector<int> map2) {
	pimpl->setup_calc();
	return pimpl->compute_4(map1,map2);
}
		
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
