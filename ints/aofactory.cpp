#include <vector>
#include <stdexcept>
#include "ints/aofactory.h"
#include "ints/integrals.h"
#include "ints/registry.h"
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
	
	vec<desc::cluster_basis> m_basvec;
	util::ShrPool<libint2::Engine> m_eng_pool;
	
	std::string m_intname = "";

public:

	registry m_reg;
	
	impl(desc::molecule& mol, dbcsr::world w) :
		m_mol(mol), m_world(w) { init(); }
	
	void init() {
		libint2::initialize();
	}
	
	void set_operator(std::string op) {
		
		if (op == "coulomb") {
			m_Op = libint2::Operator::coulomb;
		} else if (op == "overlap") {
			m_Op = libint2::Operator::overlap;
		} else if (op == "kinetic") {
			m_Op = libint2::Operator::kinetic;
		} else if (op == "nuclear") {
			m_Op = libint2::Operator::nuclear;
		} else if (op == "erfc_coulomb") {
			m_Op = libint2::Operator::erfc_coulomb;
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
		
	}
	
	void set_name(std::string istr) {
		m_intname = istr;
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

		calc_ints(m_ints, m_eng_pool, m_basvec);
		
		auto out = m_ints.get_smatrix();
		
		m_reg.insert_matrix<double>(m_intname,out);
		
		return out;
		
	}
	
	dbcsr::smatrix<double> compute_screen(std::string method, std::string dim) {
		
		auto rowsizes = m_basvec[0].cluster_sizes();
		auto colsizes = (dim == "bbbb") ? m_basvec[1].cluster_sizes() : vec<int>{1};
		
		char sym = (dim == "bbbb") ? dbcsr_type_symmetric : dbcsr_type_no_symmetry;
		
		dbcsr::mat_d m_ints = dbcsr::mat_d::create()
			.name(m_intname)
			.set_world(m_world)
			.row_blk_sizes(rowsizes).col_blk_sizes(colsizes)
			.type(sym);
			
		// reserve symmtric blocks
		int nblks = m_ints.nblkrows_total();
		
		if (sym == dbcsr_type_symmetric) {
		
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
			
		} else {
			
			m_ints.reserve_all();
			
		}
		
		if (dim == "bbbb" && method == "schwarz") {
			calc_ints_schwarz_mn(m_ints, m_eng_pool, m_basvec);
		} else if (dim == "xx" && method == "schwarz") {
			calc_ints_schwarz_x(m_ints, m_eng_pool, m_basvec);
		} else {
			throw std::runtime_error("Unknown screening method.");
		}
		
		auto out = m_ints.get_smatrix();
		
		m_reg.insert_matrix<double>(m_intname,out);
		
		return out;
		
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
	
	dbcsr::stensor<3,double> compute_3(vec<int>& map1, vec<int>& map2, dbcsr::smatrix<double> scr) { 
		dbcsr::pgrid<3> grid(m_world.comm()); 
		arrvec<int,3> blksizes; 
		for (int i = 0; i != 3; ++i) { 
			blksizes[i] = m_basvec[i].cluster_sizes(); 
		} 
		
		dbcsr::tensor<3> t_ints = dbcsr::tensor<3>::create().name(m_intname) 
			.ngrid(grid).map1(map1).map2(map2).blk_sizes(blksizes); 
			
		if (scr) {
			
			auto Z_blocks = dbcsr::block_norms(*scr);
			
			size_t x_nblks = blksizes[2].size();
			size_t b_nblks = blksizes[1].size();
			
			size_t ntot = x_nblks*b_nblks*b_nblks;
			
			arrvec<int,3> res;
			res[0].reserve(ntot);
			res[1].reserve(ntot);
			res[2].reserve(ntot);
			
			size_t totblk = 0;
			
			auto blk_idx_loc = t_ints.blks_local();
			
			int blk_mu, blk_nu, blk_x;
			
			for (int i = 0; i != blk_idx_loc[1].size(); ++i) {
				blk_mu = blk_idx_loc[1][i];
				for (int j = 0; j != blk_idx_loc[2].size(); ++j) {
					blk_nu = blk_idx_loc[2][j];
					
					if (Z_blocks(blk_mu,blk_nu) >= dbcsr::filter_eps) {
						for (int x = 0; x != blk_idx_loc[0].size(); ++x) {
							blk_x = blk_idx_loc[0][x];
							res[0].push_back(blk_x);
							res[1].push_back(blk_mu);
							res[2].push_back(blk_nu);
						}
					} else {
						++totblk;
					}
				}
			}
			
			std::cout << "SCREENED: " << totblk << std::endl;
			
			t_ints.reserve(res);
			
		} else {
			
			t_ints.reserve_all();
			
		}	
		
		calc_ints(t_ints, m_eng_pool, m_basvec); 
		auto out = t_ints.get_stensor();
		
		m_reg.insert_tensor<3,double>(m_intname,out);
		
		return out;
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
		auto out = t_ints.get_stensor();
		
		m_reg.insert_tensor<4,double>(m_intname,out);
		
		dbcsr::print(*out);
		
		return out;
	}
		
};

aofactory::aofactory(desc::molecule& mol, dbcsr::world& w) : m_mol(mol), pimpl(new impl(mol, w))  {}
aofactory::~aofactory() { delete pimpl; };

dbcsr::smatrix<double> aofactory::ao_overlap() {
	
	std::string intname = m_mol.name() + "_s_bb";
	
	auto out = pimpl->m_reg.get_matrix<double>(intname);
	if (out) return out;
	
	pimpl->set_name("s_bb");
	pimpl->set_braket("bb");
	pimpl->set_operator("overlap");
	pimpl->setup_calc();
	return pimpl->compute();
}
	
dbcsr::smatrix<double> aofactory::ao_kinetic() {
	
	std::string intname = m_mol.name() + "_k_bb";
	
	auto out = pimpl->m_reg.get_matrix<double>(intname);
	if (out) return out;
	
	pimpl->set_name("t_bb");
	pimpl->set_braket("bb");
	pimpl->set_operator("kinetic");
	pimpl->setup_calc();
	return pimpl->compute();
}

dbcsr::smatrix<double> aofactory::ao_nuclear() {
	
	std::string intname = m_mol.name() + "_v_bb";
	
	auto out = pimpl->m_reg.get_matrix<double>(intname);
	if (out) return out;
	
	pimpl->set_name("v_bb");
	pimpl->set_braket("bb");
	pimpl->set_operator("nuclear");
	pimpl->setup_calc();
	return pimpl->compute();
}

dbcsr::smatrix<double> aofactory::ao_3coverlap() {
	
	std::string intname = m_mol.name() + "_s_xx";
	
	auto out = pimpl->m_reg.get_matrix<double>(intname);
	if (out) return out;
	
	pimpl->set_name("s_xx");
	pimpl->set_braket("xx");
	pimpl->set_operator("coulomb");
	pimpl->setup_calc();
	return pimpl->compute();
}

dbcsr::stensor<3,double> aofactory::ao_3c2e(vec<int> map1, vec<int> map2, dbcsr::smatrix<double> scr) {
	
	auto name = m_mol.name() + "_i_xbb_" + pimpl->m_reg.map_to_string(map1,map2);
	
	auto out = pimpl->m_reg.get_tensor<3,double>(name);
	if (out) return out;
	
	pimpl->set_name(name);
	pimpl->set_braket("xbb");
	pimpl->set_operator("coulomb");
	pimpl->setup_calc();
	return pimpl->compute_3(map1,map2,scr);
}

dbcsr::stensor<4,double> aofactory::ao_eri(vec<int> map1, vec<int> map2, bool reorder, bool move) {
	
	auto name = m_mol.name() + "_i_bbbb_" + pimpl->m_reg.map_to_string(map1,map2);
	
	auto out = pimpl->m_reg.get_tensor<4,double>(name,reorder,move);
	if (out) return out;
	
	std::cout << "Computing ints." << std::endl;
	
	pimpl->set_name(name);
	pimpl->set_braket("bbbb");
	pimpl->set_operator("coulomb");
	pimpl->setup_calc();
	return pimpl->compute_4(map1,map2);
}

dbcsr::smatrix<double> aofactory::ao_schwarz() {
	pimpl->set_name("Z_mn");
	pimpl->set_braket("bbbb");
	pimpl->set_operator("coulomb");
	pimpl->setup_calc();
	return pimpl->compute_screen("schwarz", "bbbb");
}
	
dbcsr::smatrix<double> aofactory::ao_3cschwarz() {
	pimpl->set_name("Z_x");
	pimpl->set_braket("xx");
	pimpl->set_operator("coulomb");
	pimpl->setup_calc();
	return pimpl->compute_screen("schwarz", "xx");
}
		

} // end namespace ints
