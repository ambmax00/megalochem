#include <vector>
#include <stdexcept>
#include "ints/aofactory.h"
#include "ints/integrals.h"
#include "ints/registry.h"
#include "ints/screening.h"
#include "utils/pool.h"
#include <libint2.hpp>

#include <iostream>
#include <limits>

namespace ints {
	
class aofactory::impl {
protected:

	desc::smolecule m_mol;
	dbcsr::world m_world;
	
	const desc::cluster_basis m_cbas;
	std::optional<const desc::cluster_basis> m_xbas;
	
	libint2::Operator m_Op = libint2::Operator::invalid;
	libint2::BraKet m_BraKet = libint2::BraKet::invalid;
	
	vec<const desc::cluster_basis*> m_basvec;
	util::ShrPool<libint2::Engine> m_eng_pool;
	
	std::string m_intname = "";

public:

	registry m_reg;
	
	impl(desc::smolecule mol, dbcsr::world w) :
		m_mol(mol), 
		m_world(w),
		m_cbas(m_mol->c_basis()),
		m_xbas((m_mol->c_dfbasis()) ? 
			std::make_optional<const desc::cluster_basis>(*m_mol->c_dfbasis()) :
			std::nullopt)
		{ init(); }
	
	void init() {
		libint2::initialize();
	}
	
	desc::smolecule mol() { return m_mol; }
	
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
			std::cout << "ERROR" << std::endl;
			m_Op = libint2::Operator::erfc_coulomb;
		}
		
		if (m_Op == libint2::Operator::invalid) 
			throw std::runtime_error("Invalid operator: "+ op);
	}
	
	void set_braket(std::string dim) {
		
		if (dim == "bb") { 
			m_basvec = {&m_cbas,&m_cbas};
		} else if (dim == "xx") {
			m_basvec = {&*m_xbas, &*m_xbas};
			m_BraKet = libint2::BraKet::xs_xs;
		} else if (dim == "xbb") {
			m_basvec = {&*m_xbas, &m_cbas, &m_cbas};
			m_BraKet = libint2::BraKet::xs_xx;
		} else if (dim == "bbbb") {
			m_basvec = {&m_cbas, &m_cbas, &m_cbas, &m_cbas};
			m_BraKet = libint2::BraKet::xx_xx;
		} else {
			throw std::runtime_error("Unsupported basis set specifications: "+ dim);
		}
		
	}
	
	void set_name(std::string istr) {
		m_intname = istr;
	}
	
	void setup_calc(bool screen = false) {
		
		size_t max_nprim = 0;
		int max_l = 0;
		
		for (int i = 0; i != m_basvec.size(); ++i) {
			max_nprim = std::max(m_basvec[i]->max_nprim(), max_nprim);
			max_l = std::max(m_basvec[i]->max_l(), max_l);
		}
		
		libint2::Engine eng(m_Op, max_nprim, max_l, 0, std::numeric_limits<double>::epsilon());
			
		if (m_Op == libint2::Operator::nuclear) {
			eng.set_params(make_point_charges(m_mol->atoms())); 
		} else if (m_Op == libint2::Operator::erfc_coulomb) {
			eng.set_params(global::omega);
		}
		
		eng.set(m_BraKet);
		
		if (screen) {
			eng.set_precision(pow(global::precision,2));
		} else {
			eng.set_precision(global::precision);
		}
			
		m_eng_pool = util::make_pool<libint2::Engine>(eng);
				
	}
	
	void finalize() {
		libint2::finalize();
	}
	
	template <int N, typename T = double>
	dbcsr::stensor<N,T> setup_tensor(vec<int> map1, vec<int> map2) {
		
		dbcsr::pgrid<N> grid(m_world.comm()); 
		arrvec<int,N> blksizes; 
		for (int i = 0; i != N; ++i) { 
			blksizes[i] = m_basvec[i]->cluster_sizes(); 
		} 
			
		dbcsr::stensor<N,T> t_out = dbcsr::make_stensor<N,T>(
			typename dbcsr::tensor<N,T>::create().name(m_intname) 
			.ngrid(grid).map1(map1).map2(map2).blk_sizes(blksizes));
			
		return t_out;
		
	}
	
	dbcsr::smatrix<double> compute() {
		
		auto rowsizes = m_basvec[0]->cluster_sizes();
		auto colsizes = m_basvec[1]->cluster_sizes();
		
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
		
		return out;
		
	}
	
	dbcsr::smatrix<double> compute_screen(std::string method, std::string dim) {
		
		auto rowsizes = (dim == "bbbb") ? m_mol->dims().s() : m_mol->dims().xs();
		auto colsizes = (dim == "bbbb") ? m_mol->dims().s() : vec<int>{1};
		
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
	
		return out;
		
	}
		
	dbcsr::stensor<2,double> compute_2(vec<int>& map1, vec<int>& map2) { 
		dbcsr::pgrid<2> grid(m_world.comm()); 
		arrvec<int,2> blksizes;
		for (int i = 0; i != 2; ++i) { 
			blksizes[i] = m_basvec[i]->cluster_sizes(); 
		} 
		dbcsr::tensor<2> t_ints = dbcsr::tensor<2>::create().name(m_intname) 
			.ngrid(grid).map1(map1).map2(map2).blk_sizes(blksizes); 
			
		t_ints.reserve_all();
			
		calc_ints(t_ints, m_eng_pool, m_basvec); 
		return t_ints.get_stensor();
	}
	
	dbcsr::stensor<3,double> compute_3_full(vec<int>& map1, vec<int>& map2, screener* scr) { 
		dbcsr::pgrid<3> grid(m_world.comm()); 
		arrvec<int,3> blksizes; 
		for (int i = 0; i != 3; ++i) { 
			blksizes[i] = m_basvec[i]->cluster_sizes(); 
		} 
		
		dbcsr::tensor<3> t_ints = dbcsr::tensor<3>::create().name(m_intname) 
			.ngrid(grid).map1(map1).map2(map2).blk_sizes(blksizes); 
	
		if (scr) {
			
			size_t x_nblks = blksizes[0].size();
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
					for (int x = 0; x != blk_idx_loc[0].size(); ++x) {
						blk_x = blk_idx_loc[0][x];
						
						if (scr->skip_block(blk_x,blk_mu,blk_nu)) {
							++totblk;
							continue;
						}
						
						res[0].push_back(blk_x);
						res[1].push_back(blk_mu);
						res[2].push_back(blk_nu);
						
					}
				}
			}
			
			std::cout << "SCREENED: " << totblk << std::endl;
			
			t_ints.reserve(res);
			
		} else {
		
			t_ints.reserve_all();
			
		}	
		
		calc_ints(t_ints, m_eng_pool, m_basvec, scr); 
		auto out = t_ints.get_stensor();
		
		return out;
	}
	
	void compute_3_partial(dbcsr::stensor<3>& t_in, vec<vec<int>>& blkbounds,
		screener* scr) {
			
		auto blksizes = t_in->blk_sizes(); 
			
		size_t totblk = 0;
		
		auto blk_idx_loc = t_in->blks_local();
		
		auto idx_speed = t_in->idx_speed();
		
		const int dim0 = idx_speed[2];
		const int dim1 = idx_speed[1];
		const int dim2 = idx_speed[0];
		
		const size_t nblk0 = blkbounds[0][1] - blkbounds[0][0] + 1;
		const size_t nblk1 = blkbounds[1][1] - blkbounds[1][0] + 1;
		const size_t nblk2 = blkbounds[2][1] - blkbounds[2][0] + 1;
		
		const size_t maxblks = nblk0 * nblk1 * nblk2;
		
		int iblk[3];
		
		arrvec<int,3> res;
		
		for (auto& r : res) r.reserve(maxblks);

		for (int i0 = 0; i0 != blk_idx_loc[dim0].size(); ++i0) {
			
			iblk[dim0] = blk_idx_loc[dim0][i0];
			if (iblk[dim0] < blkbounds[dim0][0] || iblk[dim0] > blkbounds[dim0][1]) continue;
			 
			for (int i1 = 0; i1 != blk_idx_loc[dim1].size(); ++i1) {
				
				iblk[dim1] = blk_idx_loc[dim1][i1];
				if (iblk[dim1] < blkbounds[dim1][0] || iblk[dim1] > blkbounds[dim1][1]) continue;
				
				for (int i2 = 0; i2 != blk_idx_loc[dim2].size(); ++i2) {
					iblk[dim2] = blk_idx_loc[dim2][i2];
					
					if (iblk[dim2] < blkbounds[dim2][0] || iblk[dim2] > blkbounds[dim2][1]) continue;
					
					if (scr->skip_block(iblk[0],iblk[1],iblk[2])) {
						++totblk;
						continue;
					}
					
					res[0].push_back(iblk[0]);
					res[1].push_back(iblk[1]);
					res[2].push_back(iblk[2]);
					
				}
			}
			
		}
		
		t_in->reserve(res);
		
		calc_ints(*t_in, m_eng_pool, m_basvec, scr); 
		
	}
	
	void compute_3_partial_sym(dbcsr::stensor<3>& t_in, vec<vec<int>>& blkbounds,
		screener* scr) {
			
		auto blksizes = t_in->blk_sizes(); 
			
		size_t totblk = 0;
		
		auto blk_idx_loc = t_in->blks_local();
		
		const size_t nblk0 = blkbounds[0][1] - blkbounds[0][0] + 1;
		const size_t nblk1 = blkbounds[1][1] - blkbounds[1][0] + 1;
		const size_t nblk2 = blkbounds[2][1] - blkbounds[2][0] + 1;
		
		const size_t maxblks = nblk0 * nblk1 * nblk2;
		
		arrvec<int,3> res;
		
		for (auto& r : res) r.reserve(maxblks);

		for (auto const& iblkx : blk_idx_loc[0]) {
			
			if (iblkx < blkbounds[0][0] || iblkx > blkbounds[0][1]) continue;
			 
			for (auto const& iblknu : blk_idx_loc[2]) {
				
				if (iblknu < blkbounds[2][0] || iblknu > blkbounds[2][1]) continue;
				
				for (auto const& iblkmu : blk_idx_loc[1]) {
					
					if (iblkmu < blkbounds[1][0] || iblkmu > blkbounds[1][1]) continue;
					
					if (scr->skip_block(iblkx,iblkmu,iblknu)) {
						++totblk;
						continue;
					}
					
					res[0].push_back(iblkx);
					res[1].push_back(iblkmu);
					res[2].push_back(iblknu);
					
				}
			}
			
		}
		
		t_in->reserve(res);
		
		calc_ints(*t_in, m_eng_pool, m_basvec, scr); 
		
	}
	
	void compute_3_simple(dbcsr::stensor<3>& t_in) {
		
		calc_ints(*t_in, m_eng_pool, m_basvec, nullptr); 
		
	}
	
	dbcsr::stensor<4,double> compute_4(vec<int>& map1, vec<int>& map2) {
		
		dbcsr::pgrid<4> grid(m_world.comm()); 
		arrvec<int,4> blksizes; 
		for (int i = 0; i != 4; ++i) { 
			blksizes[i] = m_basvec[i]->cluster_sizes(); 
		} 
		
		dbcsr::tensor<4> t_ints = dbcsr::tensor<4>::create().name(m_intname) 
			.ngrid(grid).map1(map1).map2(map2).blk_sizes(blksizes); 
			
		t_ints.reserve_all();
			
		calc_ints(t_ints, m_eng_pool, m_basvec); 
		auto out = t_ints.get_stensor();
		
		//dbcsr::print(*out);
		
		return out;
	}
	
	std::function<void(dbcsr::stensor<3>&,vec<vec<int>>&)>
	get_generator(shared_screener s_scr) {
		
		using namespace std::placeholders;
		
		auto gen = std::bind(&aofactory::impl::compute_3_partial_sym, this, _1, _2, s_scr.get());
			
		return gen;
		
	}
	
};

aofactory::aofactory(desc::smolecule mol, dbcsr::world& w) : m_mol(mol), pimpl(new impl(mol, w))  {}
aofactory::~aofactory() { delete pimpl; };

desc::smolecule aofactory::mol() { return pimpl->mol(); }

dbcsr::smatrix<double> aofactory::ao_overlap() {
	
	std::string intname = m_mol->name() + "_s_bb";
	
	pimpl->set_name("s_bb");
	pimpl->set_braket("bb");
	pimpl->set_operator("overlap");
	pimpl->setup_calc();
	return pimpl->compute();
}
	
dbcsr::smatrix<double> aofactory::ao_kinetic() {
	
	std::string intname = m_mol->name() + "_k_bb";
	
	pimpl->set_name("t_bb");
	pimpl->set_braket("bb");
	pimpl->set_operator("kinetic");
	pimpl->setup_calc();
	return pimpl->compute();
}

dbcsr::smatrix<double> aofactory::ao_nuclear() {
	
	std::string intname = m_mol->name() + "_v_bb";
	
	pimpl->set_name("v_bb");
	pimpl->set_braket("bb");
	pimpl->set_operator("nuclear");
	pimpl->setup_calc();
	return pimpl->compute();
}

dbcsr::smatrix<double> aofactory::ao_3coverlap(std::string metric) {
	
	pimpl->set_name("s_xx_"+metric);
	pimpl->set_braket("xx");
	pimpl->set_operator(metric);
	pimpl->setup_calc();
	return pimpl->compute();
}

/*
dbcsr::smatrix<double> ao_3coverlap_inv() {
	
	std::string name = m_mol->name() + "_s_xx_inv";
	
	auto s_xx = this->ao_3coverlap();
	math::hermitian_eigen_solver solver(s_xx, 'V');
	solver.compute();
	
	out = solver.inverse();
	out.set_name(name);
	
	return out;
	
}

dbcsr::smatrix<double> ao_3coverlap_invsqrt() {
	
	std::string name = m_mol->name() + "_s_xx_invsqrt";
	
	auto s_xx = this->ao_3coverlap();
	math::hermitian_eigen_solver solver(s_xx, 'V');
	solver.compute();
	
	out = solver.inverse_sqrt();
	out.set_name(name);
	
	return out;
	
}
*/
dbcsr::stensor<3,double> aofactory::ao_3c2e(vec<int> map1, vec<int> map2, std::string metric, screener* scr) {
	
	auto name = m_mol->name() + "_i_xbb_" + pimpl->m_reg.map_to_string(map1,map2);
	
	pimpl->set_name(name);
	pimpl->set_braket("xbb");
	pimpl->set_operator(metric);
	pimpl->setup_calc();
	return pimpl->compute_3_full(map1,map2,scr);
}

void aofactory::ao_3c2e_setup(std::string metric) {
	
	pimpl->set_braket("xbb");
	pimpl->set_operator(metric);
	pimpl->setup_calc();
	
}

dbcsr::stensor<3,double> aofactory::ao_3c2e_setup_tensor(vec<int> map1, vec<int> map2) {
	auto name = m_mol->name() + "_i_xbb_" + pimpl->m_reg.map_to_string(map1,map2);
	pimpl->set_name(name);
	return pimpl->setup_tensor<3>(map1,map2);
	
}

void aofactory::ao_3c2e_fill(dbcsr::stensor<3,double>& t_in, vec<vec<int>>& blkbounds, shared_screener scr, bool sym) {
	
	if (!sym) {
		pimpl->compute_3_partial(t_in,blkbounds,scr.get());
	} else {
		pimpl->compute_3_partial_sym(t_in,blkbounds,scr.get());
	}
}

void aofactory::ao_3c2e_fill(dbcsr::stensor<3,double>& t_in) {
	
	pimpl->compute_3_simple(t_in);
	
}
	

dbcsr::stensor<4,double> aofactory::ao_eri(vec<int> map1, vec<int> map2) {
	
	auto name = m_mol->name() + "_i_bbbb_" + pimpl->m_reg.map_to_string(map1,map2);
	
	pimpl->set_name(name);
	pimpl->set_braket("bbbb");
	pimpl->set_operator("coulomb");
	pimpl->setup_calc();
	return pimpl->compute_4(map1,map2);
}

dbcsr::smatrix<double> aofactory::ao_schwarz(std::string metric) {
	pimpl->set_name("Z_mn");
	pimpl->set_braket("bbbb");
	pimpl->set_operator(metric);
	pimpl->setup_calc(true);
	return pimpl->compute_screen("schwarz", "bbbb");
}
	
dbcsr::smatrix<double> aofactory::ao_3cschwarz(std::string metric) {
	pimpl->set_name("Z_x");
	pimpl->set_braket("xx");
	pimpl->set_operator(metric);
	pimpl->setup_calc(true);
	return pimpl->compute_screen("schwarz", "xx");
}

std::function<void(dbcsr::stensor<3>&,vec<vec<int>>&)>
	aofactory::get_generator(shared_screener s_scr) {
		
		return pimpl->get_generator(s_scr);
		
}


} // end namespace ints
