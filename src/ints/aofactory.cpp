#include <vector>
#include <stdexcept>
#include "ints/aofactory.h"
#include "ints/integrals.h"
#include "ints/screening.h"
#include "utils/pool.h"
#include <libint2.hpp>

#include <iostream>
#include <limits>

namespace ints {

using vshell_l = std::vector<libint2::Shell>;

using vvshell_l = std::vector<std::vector<libint2::Shell>>;

using shared_vvshell_l = std::shared_ptr<
	std::vector<std::vector<libint2::Shell>>>;
	
class aofactory::impl {
protected:

	dbcsr::world m_world;
	
	desc::smolecule m_mol;
	shared_vvshell_l m_cbas;
	shared_vvshell_l m_xbas;
	
	libint2::Operator m_Op = libint2::Operator::invalid;
	libint2::BraKet m_BraKet = libint2::BraKet::invalid;
	
	vec<const vvshell_l*> m_basvec;
	util::ShrPool<libint2::Engine> m_eng_pool;
	
	std::string m_intname = "";

public:
	
	impl(desc::smolecule mol, dbcsr::world w) :
		m_world(w),
		m_mol(mol)
		{ init(); }
	
	void init() {
		
		libint2::initialize();
		
		// convert to libint format
		auto cbas = m_mol->c_basis();
		auto xbas = m_mol->c_dfbasis();
		
		auto convert_vec = [](std::vector<double> vec) {
			libint2::svector<double> out;
			for (auto e : vec) {
				out.push_back(e);
			} 
			return out;
		};
		
		auto convert_bas = [&convert_vec](auto& mega_bas) {
			shared_vvshell_l libint_bas = 
				std::make_shared<vvshell_l>(0);
			
			for (auto& cluster : mega_bas) {
				vshell_l vs_l;
				for (auto& s : cluster) {
					
					libint2::Shell s_l;
					
					s_l.O = s.O;
					s_l.alpha = convert_vec(s.alpha);
					
					libint2::svector<libint2::Shell::Contraction> c_l(1);
					
					c_l[0].l = s.l;
					c_l[0].pure = s.pure;
					
					auto coeff = convert_vec(s.coeff);
					libint2::svector<double> max_ln_c(s.alpha.size(), 
						-std::numeric_limits<double>::max());
					
					for (int i = 0; i != coeff.size(); ++i) {
						max_ln_c[i] = std::max(max_ln_c[i], std::log(std::abs(coeff[i])));
					}
									
					c_l[0].coeff = coeff;
					
					s_l.contr = c_l;
					s_l.max_ln_coeff = max_ln_c;
					vs_l.push_back(s_l);
			
				}
				libint_bas->push_back(vs_l);
			}
			
			return libint_bas;
			
		};
		
		if (cbas) m_cbas = convert_bas(*cbas);
		if (xbas) m_xbas = convert_bas(*xbas);
		
		std::cout << "LIBINT2 BASIS:" << std::endl;
		for (auto c : *m_cbas) {
			std::cout << "CLUSTER: " << std::endl;
			for (auto s : c) {
				std::cout << s << std::endl;
			}
		}
		
		if (xbas) {
			std::cout << "LIBINT2 X BASIS:" << std::endl;
			for (auto c : *m_xbas) {
				std::cout << "CLUSTER: " << std::endl;
				for (auto s : c) {
					std::cout << s << std::endl;
				}
			}
		}
		
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
			//std::cout << "ERROR" << std::endl;
			m_Op = libint2::Operator::erfc_coulomb;
		}
		
		if (m_Op == libint2::Operator::invalid) 
			throw std::runtime_error("Invalid operator: "+ op);
	}
	
	void set_braket(std::string dim) {
		
		if (dim == "bb") { 
			m_basvec = {m_cbas.get(),m_cbas.get()};
		} else if (dim == "xx") {
			m_basvec = {m_xbas.get(),m_xbas.get()};
			m_BraKet = libint2::BraKet::xs_xs;
		} else if (dim == "xbb") {
			m_basvec = {m_xbas.get(), m_cbas.get(),m_cbas.get()};
			m_BraKet = libint2::BraKet::xs_xx;
			//std::cout << "NNNN " << m_xbas->nbf() << std::endl;
		} else if (dim == "bbbb") {
			m_basvec = {m_cbas.get(),m_cbas.get(),m_cbas.get(),m_cbas.get()};
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
		
		for (auto& c : m_basvec) {
			for (auto& vs : *c) {
				max_nprim = std::max(max_nprim, libint2::max_nprim(vs));
				max_l = std::max(max_l, libint2::max_l(vs));
			}
		}
		
		std::cout << "MAX L:" << max_l << std::endl;
		std::cout << "MAX_NPRIM:" << max_nprim << std::endl;
		
		libint2::Engine eng(m_Op, max_nprim, max_l, 0, std::numeric_limits<double>::epsilon());
			
		if (m_Op == libint2::Operator::nuclear) {
			std::vector<libint2::Atom> latoms;
			for (auto a : m_mol->atoms()) {
				libint2::Atom a_out;
				a_out.x = a.x;
				a_out.y = a.y;
				a_out.z = a.z;
				a_out.atomic_number = a.atomic_number;
				latoms.push_back(a_out);
			}
			eng.set_params(make_point_charges(latoms)); 
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
	
	std::vector<int> cluster_sizes(const vvshell_l& vvs) {
		std::vector<int> out;
		for (auto& c : vvs) {
			int size = 0;
			for (auto& s : c) {
				size += (int)s.size();
			}
			out.push_back(size);
		}
		
		std::cout << "SIZES: " << std::endl;
		for (auto i : out) {
			std::cout << i << " ";
		} std::cout << std::endl;
		
		return out;
	}
	
	template <int N, typename T = double>
	dbcsr::stensor<N,T> setup_tensor(dbcsr::shared_pgrid<N> spgrid, vec<int> map1, vec<int> map2) {
		
		arrvec<int,N> blksizes; 
		for (int i = 0; i != N; ++i) { 
			blksizes[i] = cluster_sizes(*m_basvec[i]);
		} 
			
		auto t_out = dbcsr::tensor_create<N,T>().name(m_intname)
			.pgrid(spgrid).map1(map1).map2(map2).blk_sizes(blksizes).get();
			
		return t_out;
		
	}
	
	dbcsr::shared_matrix<double> compute() {
		
		auto rowsizes = cluster_sizes(*m_basvec[0]);
		auto colsizes = cluster_sizes(*m_basvec[1]);
		
		auto m_ints = dbcsr::create<double>()
			.name(m_intname)
			.set_world(m_world)
			.row_blk_sizes(rowsizes).col_blk_sizes(colsizes)
			.matrix_type(dbcsr::type::symmetric)
			.get();
			
		// reserve symmtric blocks
		int nblks = m_ints->nblkrows_total();
		
		vec<int> resrows, rescols;
		
		for (int i = 0; i != nblks; ++i) {
			for (int j = 0; j != nblks; ++j) {
				if (m_ints->proc(i,j) == m_world.rank() && i <= j) {
					resrows.push_back(i);
					rescols.push_back(j);
				}
			}
		}
		
		m_ints->reserve_blocks(resrows,rescols);

		calc_ints(*m_ints, m_eng_pool, m_basvec);
		
		return m_ints;
		
	}
	
	dbcsr::shared_matrix<double> compute_screen(std::string method, std::string dim) {
		
		auto rowsizes = (dim == "bbbb") ? m_mol->dims().s() : m_mol->dims().xs();
		auto colsizes = (dim == "bbbb") ? m_mol->dims().s() : vec<int>{1};
		
		auto sym = (dim == "bbbb") ? dbcsr::type::symmetric : dbcsr::type::no_symmetry;
		
		auto m_ints = dbcsr::create<double>()
			.name(m_intname)
			.set_world(m_world)
			.row_blk_sizes(rowsizes).col_blk_sizes(colsizes)
			.matrix_type(sym).get();
			
		// reserve symmtric blocks
		int nblks = m_ints->nblkrows_total();
		
		if (sym == dbcsr::type::symmetric) {
		
			vec<int> resrows, rescols;
			
			for (int i = 0; i != nblks; ++i) {
				for (int j = 0; j != nblks; ++j) {
					if (m_ints->proc(i,j) == m_world.rank() && i <= j) {
						resrows.push_back(i);
						rescols.push_back(j);
					}
				}
			}
			
			m_ints->reserve_blocks(resrows,rescols);
			
		} else {
			
			m_ints->reserve_all();
			
		}
		
		if (dim == "bbbb" && method == "schwarz") {
			calc_ints_schwarz_mn(*m_ints, m_eng_pool, m_basvec);
		} else if (dim == "xx" && method == "schwarz") {
			calc_ints_schwarz_x(*m_ints, m_eng_pool, m_basvec);
		} else {
			throw std::runtime_error("Unknown screening method.");
		}
		
		return m_ints;
		
	}
		
	dbcsr::stensor<2,double> compute_2(dbcsr::shared_pgrid<2> spgrid, vec<int>& map1, vec<int>& map2) { 
		
		arrvec<int,2> blksizes;
		for (int i = 0; i != 2; ++i) { 
			blksizes[i] = cluster_sizes(*m_basvec[i]); 
		} 
		
		auto t_ints = dbcsr::tensor_create<2>()
			.name(m_intname) 
			.pgrid(spgrid)
			.map1(map1).map2(map2)
			.blk_sizes(blksizes)
			.get(); 
			
		t_ints->reserve_all();
			
		calc_ints(*t_ints, m_eng_pool, m_basvec); 
		return t_ints;
	}
	
	void compute_3_partial(dbcsr::shared_tensor<3>& t_in, vec<vec<int>>& blkbounds,
		shared_screener s_scr) {
			
		auto scr = s_scr.get();
			
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
	
	void compute_4_partial(dbcsr::shared_tensor<4>& t_in, vec<vec<int>>& blkbounds,
		shared_screener s_scr) {
		
		auto scr = s_scr.get();
		
		auto blksizes = t_in->blk_sizes(); 
			
		size_t totblk = 0;
		
		auto blk_idx_loc = t_in->blks_local();
		
		auto idx_speed = t_in->idx_speed();
		
		const int dim0 = idx_speed[3];
		const int dim1 = idx_speed[2];
		const int dim2 = idx_speed[1];
		const int dim3 = idx_speed[0];
		
		const size_t nblk0 = blkbounds[0][1] - blkbounds[0][0] + 1;
		const size_t nblk1 = blkbounds[1][1] - blkbounds[1][0] + 1;
		const size_t nblk2 = blkbounds[2][1] - blkbounds[2][0] + 1;
		const size_t nblk3 = blkbounds[3][1] - blkbounds[3][0] + 1;
		
		const size_t maxblks = nblk0 * nblk1 * nblk2 * nblk3;
		
		int iblk[4];
		
		arrvec<int,4> res;
		
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
					
					for (int i3 = 0; i3 != blk_idx_loc[dim3].size(); ++i3) {
						iblk[dim3] = blk_idx_loc[dim3][i3];
						
						if (iblk[dim3] < blkbounds[dim3][0] || iblk[dim3] > blkbounds[dim3][1]) continue;
					
						res[0].push_back(iblk[0]);
						res[1].push_back(iblk[1]);
						res[2].push_back(iblk[2]);
						res[3].push_back(iblk[3]);
						
					}
					
				}
			}
			
		}
		
		t_in->reserve(res);
		
		calc_ints(*t_in, m_eng_pool, m_basvec); 
		
	}
	
	void compute_3_partial_sym(dbcsr::shared_tensor<3>& t_in, vec<vec<int>>& blkbounds,
		shared_screener s_scr) {
			
		auto scr = s_scr.get();
		
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
	
	void compute_3_partial(dbcsr::shared_tensor<3>& t_in, arrvec<int,3>& idx,
		shared_screener s_scr) {
			
		auto scr = s_scr.get();
		
		arrvec<int,3> newblks;
		
		for (auto x : idx[0]) {
			for (auto m : idx[1]) {
				for (auto n : idx[2]) {
					
					if (scr->skip_block(x,m,n)) continue;
					
					newblks[0].push_back(x);
					newblks[1].push_back(m);
					newblks[2].push_back(n);
			
				}
			}
		}
		
		t_in->reserve(newblks);
		
		calc_ints(*t_in, m_eng_pool, m_basvec, scr); 
		
	}
	
	std::function<void(dbcsr::shared_tensor<3>&,vec<vec<int>>&)>
	get_generator(shared_screener s_scr) {
		
		using namespace std::placeholders;
		
		auto gen = std::bind(&aofactory::impl::compute_3_partial_sym, this, _1, _2, s_scr);
			
		return gen;
		
	}
	
	desc::smolecule mol() { return m_mol; }
	
};

aofactory::aofactory(desc::smolecule mol, dbcsr::world& w) : 
	pimpl(new impl(mol, w))  {}
	
aofactory::~aofactory() { delete pimpl; }
	
dbcsr::shared_matrix<double> aofactory::ao_overlap() {
	
	std::string intname = "s_bb";
	
	pimpl->set_name("s_bb");
	pimpl->set_braket("bb");
	pimpl->set_operator("overlap");
	pimpl->setup_calc();
	return pimpl->compute();
}
	
dbcsr::shared_matrix<double> aofactory::ao_kinetic() {
	
	std::string intname = "k_bb";
	
	pimpl->set_name("t_bb");
	pimpl->set_braket("bb");
	pimpl->set_operator("kinetic");
	pimpl->setup_calc();
	return pimpl->compute();
}

dbcsr::shared_matrix<double> aofactory::ao_nuclear() {
	
	std::string intname = "v_bb";
	
	pimpl->set_name("v_bb");
	pimpl->set_braket("bb");
	pimpl->set_operator("nuclear");
	pimpl->setup_calc();
	return pimpl->compute();
}

dbcsr::shared_matrix<double> aofactory::ao_3coverlap(std::string metric) {
	
	pimpl->set_name("s_xx_"+metric);
	pimpl->set_braket("xx");
	pimpl->set_operator(metric);
	pimpl->setup_calc();
	return pimpl->compute();
}

void aofactory::ao_3c2e_setup(std::string metric) {
	
	pimpl->set_braket("xbb");
	pimpl->set_operator(metric);
	pimpl->setup_calc();
	
}

void aofactory::ao_eri_setup(std::string metric) {
	
	pimpl->set_braket("bbbb");
	pimpl->set_operator(metric);
	pimpl->setup_calc();
	
}

dbcsr::shared_tensor<3,double> aofactory::ao_3c2e_setup_tensor(dbcsr::shared_pgrid<3> spgrid, 
	vec<int> map1, vec<int> map2) {
	
	std::string name = "i_xbb";
	pimpl->set_name(name);
	return pimpl->setup_tensor<3>(spgrid,map1,map2);
	
}

dbcsr::shared_tensor<4,double> aofactory::ao_eri_setup_tensor(dbcsr::shared_pgrid<4> spgrid, 
	vec<int> map1, vec<int> map2) {
	
	std::string name = "i_bbbb";
	pimpl->set_name(name);
	return pimpl->setup_tensor<4>(spgrid,map1,map2);
	
}

void aofactory::ao_3c2e_fill(dbcsr::shared_tensor<3,double>& t_in, 
	vec<vec<int>>& blkbounds, shared_screener scr, bool sym) {
	
	if (!sym) {
		pimpl->compute_3_partial(t_in,blkbounds,scr);
	} else {
		pimpl->compute_3_partial_sym(t_in,blkbounds,scr);
	}
}

void aofactory::ao_3c2e_fill(dbcsr::shared_tensor<3>& t_in, 
	arrvec<int,3>& idx, shared_screener s_scr) {
	
	pimpl->compute_3_partial(t_in, idx, s_scr);
	
}
				
void aofactory::ao_eri_fill(dbcsr::shared_tensor<4,double>& t_in, 
	vec<vec<int>>& blkbounds, shared_screener scr, bool sym) {
	
	pimpl->compute_4_partial(t_in,blkbounds,scr);
	
}

dbcsr::shared_matrix<double> aofactory::ao_schwarz(std::string metric) {
	pimpl->set_name("Z_mn");
	pimpl->set_braket("bbbb");
	pimpl->set_operator(metric);
	pimpl->setup_calc(true);
	return pimpl->compute_screen("schwarz", "bbbb");
}
	
dbcsr::shared_matrix<double> aofactory::ao_3cschwarz(std::string metric) {
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

desc::smolecule aofactory::mol() { return pimpl->mol(); }


} // end namespace ints
