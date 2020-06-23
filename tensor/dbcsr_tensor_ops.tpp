#ifndef DBCSR_TENSOR_OPS_HPP
#define DBCSR_TENSOR_OPS_HPP

#:include "dbcsr.fypp"
#include <dbcsr_tensor.hpp>

namespace dbcsr {
	
template <typename D>
D* unfold_bounds(vec<vec<D>>& v) {
	int b_size = v.size();
	D* f_bounds = new D[2 * b_size];
	for (int j = 0; j != b_size; ++j) {
		for (int i = 0; i != 2; ++i) {
			f_bounds[i + j * 2] = v[j][i];
		}
	}
	return f_bounds;
}

template <int N, typename T>
class copy_base {

    #:set list = [ &
        ['order','vec<int>','optional','val'],&
        ['sum','bool','optional','val'],&
        ['bounds','vec<vec<int>>', 'optional', 'ref'],&
        ['move_data','bool','optional','val']]
    ${make_param('copy_base',list)}$
        
private:

    tensor<N,T>& c_t_in;
    tensor<N,T>& c_t_out;
    
public:

    copy_base(tensor<N,T>& t1, tensor<N,T>& t2) : c_t_in(t1), c_t_out(t2) {}
    
    void perform() {
        
        int* fbounds = (c_bounds) ? unfold_bounds<int>(*c_bounds) : nullptr;
        
        c_dbcsr_t_copy(c_t_in.m_tensor_ptr, N, c_t_out.m_tensor_ptr,
            (c_order) ? c_order->data() : nullptr, 
            (c_sum) ? &*c_sum : nullptr, fbounds,
            (c_move_data) ? &*c_move_data : nullptr, nullptr);
            
        if (fbounds) delete [] fbounds;
            
    }
            
};

template <int N, typename T>
copy_base(tensor<N,T>& t1, tensor<N,T>& t2) -> copy_base<tensor<N,T>::dim, typename tensor<N,T>::value_type>;

template <typename tensortype>
inline copy_base<tensortype::dim, typename tensortype::value_type>
copy(tensortype& t1, tensortype& t2) {
	return copy_base(t1,t2);
}

template <int N1, int N2, int N3, typename T>
class contract_base {

    #:set list = [ &
        ['alpha','T','optional','val'],&
        ['beta', 'T', 'optional', 'val'],&
        ['con1', 'vec<int>', 'required', 'val'],&
        ['ncon1', 'vec<int>', 'required', 'val'],&
        ['con2', 'vec<int>', 'required', 'val'],&
        ['ncon2', 'vec<int>', 'required', 'val'],&
        ['map1', 'vec<int>', 'required', 'val'],&
        ['map2', 'vec<int>', 'required', 'val'],&
        ['bounds1','vec<vec<int>>','optional','ref'],&
        ['bounds2','vec<vec<int>>','optional','ref'],&
        ['bounds3','vec<vec<int>>','optional','ref'],&
        ['filter','double','optional','val'],&
        ['flop','long long int','optional','ref'],&
        ['move','bool','optional','val'],&
        ['retain_sparsity','bool','optional','val'],&
        ['print','bool','optional','val'],&
        ['log','bool','optional','val']]
    ${make_param('contract_base',list)}$

private:

    dbcsr::tensor<N1,T>& c_t1;
    dbcsr::tensor<N2,T>& c_t2;
    dbcsr::tensor<N3,T>& c_t3;
    
public:
    
    contract_base(dbcsr::tensor<N1,T>& t1, dbcsr::tensor<N2,T>& t2, dbcsr::tensor<N3,T>& t3) 
        : c_t1(t1), c_t2(t2), c_t3(t3) {}
    
    void perform() {
        
        int* f_b1 = (c_bounds1) ? unfold_bounds<int>(*c_bounds1) : nullptr;
        int* f_b2 = (c_bounds2) ? unfold_bounds<int>(*c_bounds2) : nullptr;
        int* f_b3 = (c_bounds3) ? unfold_bounds<int>(*c_bounds3) : nullptr;
        
        int out = 6;
        int* unit_nr = (c_print) ? ((*c_print) ? &out : nullptr) : nullptr;  
        
        c_dbcsr_t_contract_r_dp(
            (c_alpha) ? *c_alpha : 1,
            c_t1.m_tensor_ptr, c_t2.m_tensor_ptr,
            (c_beta) ? *c_beta : 0,
            c_t3.m_tensor_ptr, 
            c_con1->data(), c_con1->size(), 
            c_ncon1->data(), c_ncon1->size(),
            c_con2->data(), c_con2->size(),
            c_ncon2->data(), c_ncon2->size(),
            c_map1->data(), c_map1->size(),
            c_map2->data(), c_map2->size(), 
            f_b1, f_b2, f_b3,
            nullptr, nullptr, nullptr, nullptr, 
            (c_filter) ? &*c_filter : &filter_eps,
            (c_flop) ? &*c_flop : nullptr,
            (c_move) ? &*c_move : nullptr,
            (c_retain_sparsity) ? &*c_retain_sparsity : nullptr,
            unit_nr, (c_log) ? &*c_log : nullptr);
            
        delete [] f_b1, f_b2, f_b3;
             
    }
    
    void perform (std::string formula) {
        
        eval(formula);
        
        perform();
        
    }
    
    void eval(std::string str) {
        
        std::vector<std::string> idxs(4);
        std::vector<int> con1, con2, ncon1, ncon2, map1, map2;
	
        int i = 0;
	
        // Parsing input
        for (int ic = 0; ic != str.size(); ++ic) {
		
            char c = str[ic];
            
            if (c == ' ') continue;
                
            if (c == ',') {
                if (++i > 2) throw std::runtime_error("Invalid synatax: " + str); 
            } else if ((c == '-')) {
                
                
                if (str[++ic] == '>') {
                    ++i;
                } else {
                    throw std::runtime_error("Invalid syntax: " + str);
                }
                
                if (i != 2) throw std::runtime_error("Invalid syntax."+str);
            
            } else {
                idxs[i].push_back(c);
            }
            
        }
        
        if (idxs[2].size() == 0) throw std::runtime_error("Implicit mode not implemented.");
        
        //for (auto v : idxs) {
        //	std::cout << v << std::endl;
        //}
        
        // evaluating input
        auto t1 = idxs[0];
        auto t2 = idxs[1];
        auto t3 = idxs[2];
        
        //std::cout << "t3 map" << std::endl;
        
        if ((std::unique(t1.begin(), t1.end()) != t1.end()) || 
            (std::unique(t2.begin(), t2.end()) != t2.end()) ||
            (std::unique(t3.begin(), t3.end()) != t3.end()))
                throw std::runtime_error("Duplicate tensor indices: "+str);
                
        
        std::string scon, sncon1, sncon2;
        
        for (int i1 = 0; i1 != t1.size(); ++i1)  {
            auto c1 = t1[i1];
            for (int i2 = 0; i2 != t2.size(); ++i2) {
                auto c2 = t2[i2];
                if (c1 == c2) { 
                    scon.push_back(c1);
                    con1.push_back(i1);
                    con2.push_back(i2);
                }
            }
        }
        
        /*
        std::cout << "To be contrcated: " << scon << std::endl;	
        std::cout << "Maps:" << std::endl;
        for (auto v : con1) std::cout << v << " ";
        std::cout << std::endl;
        for (auto v : con2) std::cout << v << " ";
        std::cout << std::endl;
        */
        
        for (int i = 0; i != t1.size(); ++i) {
            auto found = std::find(scon.begin(), scon.end(), t1[i]);
            if (found == scon.end()) {
                sncon1.push_back(t1[i]);
                ncon1.push_back(i);
            }
        }
        
        for (int i = 0; i != t2.size(); ++i) {
            auto found = std::find(scon.begin(), scon.end(), t2[i]);
            if (found == scon.end()) {
                sncon2.push_back(t2[i]);
                ncon2.push_back(i);
            }
        }
        
        /*
        std::cout << "not con1: " << sncon1 << std::endl;
        std::cout << "not con2: " << sncon2 << std::endl;
        std::cout << "Maps:" << std::endl;
        for (auto v : ncon1) std::cout << v << " ";
        std::cout << std::endl;
        for (auto v : ncon2) std::cout << v << " ";
        std::cout << std::endl;
        */
        
        if (ncon1.size() + ncon2.size() != t3.size()) throw std::runtime_error("Wrong tensor dimensions: "+str);
        
        for (int i = 0; i != t3.size(); ++i) {
            auto found1 = std::find(sncon1.begin(),sncon1.end(),t3[i]);
            if (found1 != sncon1.end()) {
                map1.push_back(i);
            }
            auto found2 = std::find(sncon2.begin(),sncon2.end(),t3[i]);
            if (found2 != sncon2.end()) {
                map2.push_back(i);
            }
        }

        /*
        std::cout << "Maps tensor 3" << std::endl;
        for (auto v : map1) std::cout << v << " ";
        std::cout << std::endl;
        for (auto v : map2) std::cout << v << " ";
        std::cout << std::endl;
        */
        
        if (map1.size() + map2.size() != t3.size()) 
            throw std::runtime_error("Incompatible tensor dimensions: "+str);
            
        this->map1(map1);
        this->map2(map2);
        this->con1(con1);
        this->con2(con2);
        this->ncon1(ncon1);
        this->ncon2(ncon2);
        
    }
        
    
};

template <int N1, int N2, int N3, typename T>
contract_base(tensor<N1,T>& t1, tensor<N2,T>& t2, tensor<N3,T>& t3) 
-> contract_base<tensor<N1,T>::dim,tensor<N2,T>::dim,tensor<N3,T>::dim, T>;

template <typename tensor1, typename tensor2, typename tensor3>
inline contract_base<tensor1::dim,tensor2::dim,tensor3::dim,typename tensor1::value_type> 
contract(tensor1& t1, tensor2& t2, tensor3& t3) 
{
	return contract_base(t1,t2,t3);
}

template <typename T = double>
void copy_2Dtensor_to_3Dtensor(tensor<2,T>& t2, tensor<3,T>& t3, bool sum = false) {
	
	iterator_t<2> it2(t2);
    index<3> idx3d;
    index<3> size3;
    
    idx3d[2] = 0;
    size3[2] = 1;
    
    arrvec<int,3> resblkidx;
	for (int i = 0; i != 3; ++i) {
		resblkidx[i] = vec<int>(1);
	}
	
	resblkidx[2][0] = 0;
    
    it2.start();
	
	while (it2.blocks_left()) {
		
		it2.next();
		
		auto& idx2d = it2.idx();
        auto& size2 = it2.size();
				
		idx3d[0] = idx2d[0];
		idx3d[1] = idx2d[1];
		
		size3[0] = size2[0];
		size3[1] = size2[1];
        
		bool found = false;
		auto blk2 = t2.get_block(idx2d, size2, found);
		
		block<3,T> blk3(size3,blk3.data());
		
		resblkidx[0][0] = idx3d[0];
		resblkidx[1][0] = idx3d[1];
		
		t3.reserve(resblkidx);
		
		t3.put_block(idx3d, blk3);
		
	}
    
    it2.stop();
		
	//print(new_t);
    t2.finalize();
    t3.finalize();
	
}

template <typename T = double>
void copy_3Dtensor_to_2Dtensor(tensor<3,T>& t3, tensor<2,T>& t2, bool sum = false) {
	
	iterator_t<3> it3(t3);
    index<2> idx2d;
    index<2> size2;
    
    arrvec<int,2> resblkidx;
	for (int i = 0; i != 2; ++i) {
		resblkidx[i] = vec<int>(1);
	}
    
    it3.start();
	
	while (it3.blocks_left()) {
		
		it3.next();
		
		auto& idx3d = it3.idx();
        auto& size3 = it3.size();
				
		idx2d[0] = idx3d[0];
		idx2d[1] = idx3d[1];
		
		size2[0] = size3[0];
		size2[1] = size3[1];
        
		bool found = false;
		auto blk3 = t3.get_block(idx3d, size3, found);
		
		block<2,T> blk2(size2,blk3.data());
		
		resblkidx[0][0] = idx2d[0];
		resblkidx[1][0] = idx2d[1];
		
		t2.reserve(resblkidx);
		
		t2.put_block(idx2d, blk2);
		
	}
    
    it3.stop();
		
	//print(new_t);
    t2.finalize();
    t3.finalize();
	
}

template <typename T>
void copy_matrix_to_3Dtensor(matrix<T>& m, tensor<3,T>& t, bool sum = false, bool sym = false) {
	
	if (!sum) t.clear();
	
	iterator iter(m);
    
    iter.start();	
    
    dbcsr::idx3 idx = {0,0,0};
    dbcsr::idx3 sizes = {0,0,1};
    arrvec<int,3> res = {vec<int>{0},vec<int>{0},vec<int>{0}};
    
    char symtype = m.matrix_type();
    
    while (iter.blocks_left()) {
		iter.next_block();
		
		int r = iter.row();
		int c = iter.col();
		
		res[0].push_back(r);
		res[1].push_back(c);
		res[2].push_back(0);
		
		if (sym && symtype == dbcsr_type_symmetric && r != c) {
			res[0].push_back(c);
			res[1].push_back(r);
			res[2].push_back(0);
		} 
		
	}
	
	iter.stop();
	
	t.reserve(res);
	
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next_block();
				
		idx[0] = iter.row();
		idx[1] = iter.col();
        		
		sizes[0] = iter.row_size();
		sizes[1] = iter.col_size();
		block<3,T> blk(sizes,iter.data());
		
		t.put_block(idx, blk, sum);
		
		if (sym && symtype == dbcsr_type_symmetric && idx[0] != idx[1]) {
			
			sizes[0] = iter.col_size();
			sizes[1] = iter.row_size();
			
			idx[0] = iter.col();
			idx[1] = iter.row();
			
			block<3,T> blk(sizes);
			
			for (int i = 0; i != sizes[0]; ++i) {
				for (int j = 0; j != sizes[1]; ++j) {
					blk(i,j,0) = iter(j,i);
				}
			}
			
			t.put_block(idx, blk, sum);
			
		}
		
	}
    
    iter.stop();
		
    t.finalize();
    m.finalize();
	
}

template <typename T>
void copy_3Dtensor_to_matrix(tensor<3,T>& t, matrix<T>& m, bool sum = false) {
	
	if (sum) m.clear();
	
	iterator_t<3,double> iter3(t);
    
    iter3.start();
    
	vec<int> resrow, rescol;
	std::array<int,2> size2 = {0,0};
	resrow = rescol = {0};
	
	while (iter3.blocks_left()) {
		
		iter3.next();
				
		auto& idx = iter3.idx();
		auto& size3 = iter3.size();
		
		bool found = true;
	
		block<3,T> blk3 = t.get_block(idx,size3,found);
		
		size2[0] = size3[0];
		size2[1] = size3[1];
		
		block<2,T> blk2(size2,blk3.data());
		
		resrow[0] = idx[0];
		rescol[0] = idx[1];
		
		t.reserve(resrow,rescol);
		
		t.put_block(idx,blk3);
		
	}
    
    iter3.stop();
		
    t.finalize();
    m.finalize();
	
}

template <int N>
double dot(tensor<N,double>& t1, tensor<N,double>& t2) {

    // have to have same pgrid!
	double sum = 0.0;
	
    //#pragma omp parallel 
    //{
        
        iterator_t<2> iter(t1);
        iter.start();
        
		while (iter.blocks_left()) {
			
			iter.next();
            
            auto& idx = iter.idx();
            auto& size = iter.size();
			
			bool found = false;
            
			auto b1 = t1.get_block(idx, size, found);
			auto b2 = t2.get_block(idx, size, found);
			
			if (!found) continue;
			
			sum += std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), 0.0);
			
		}
        
        t1.finalize();
        t2.finalize();
        
        iter.stop();
        
    //}
		
    
	
	double MPIsum = 0.0;
	
	MPI_Allreduce(&sum,&MPIsum,1,MPI_DOUBLE,MPI_SUM,t1.comm());
	
	return MPIsum;
		
}


template <int N, typename T = double>
void ewmult(tensor<N,T>& t1, tensor<N,T>& t2, tensor<N,T>& tout) {
	
	// elementwise multiplication
	// make sure tensors have same grid and dimensions, caus I sure don't do it here
	
	//#pragma omp parallel 
   // {
        dbcsr::iterator_t<N> it(t1);
        
        it.start();
        
        while (it.blocks_left()) {
                
            it.next();
            auto& idx = it.idx();
            auto& blksize = it.size();
                
            bool found = false;
            bool found3 = false;
            
            auto b1 = t1.get_block(idx, blksize, found);
            auto b2 = t2.get_block(idx, blksize, found);
            auto b3 = tout.get_block(idx, blksize, found3);
                
            if (!found) continue;
            if (!found3) {
                arrvec<int,N> res;
                for (int i = 0; i != N; ++i) {
                    res[i].push_back(idx[i]);
                }
               
                tout.reserve(res);
                
            }
                
            std::transform(b1.data(), b1.data() + b1.ntot(), b2.data(), b3.data(), std::multiplies<T>());
            
            tout.put_block(idx, b3);
            
        }
        
        it.stop();
        t1.finalize();
        t2.finalize();
        tout.finalize();
        
  //  }
		
	tout.filter();
	
}

/*
template <int N>
void ewmult(tensor<N,double>& t1, tensor<N,double>& t2, tensor<N,double>& t3) {
	
	// dot product only for N = 2 at the moment
	//assert(N == 2);

	double sum = 0.0;
	
	dbcsr::iterator_t<N> it(t1);
	
	while (it.blocks_left()) {
			
			it.next();
			
			bool found = false;
			auto b1 = t1.get_block({.idx = it.idx(), .blk_size = it.sizes(), .found = found});
			auto b2 = t2.get_block({.idx = it.idx(), .blk_size = it.sizes(), .found = found});
			
			if (!found) continue;
			
			//std::cout  << std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), T()) << std::endl;
			
			//std::cout << "ELE: " << std::endl;
			//for (int i = 0; i != b1.ntot(); ++i) { std::cout << b1(i) << " " << b2(i) << std::endl; }
			
			
			sum += std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), 0.0);
			
			//std::cout << "SUM: " << std::inner_product(b1.data(), b1.data() + b1.ntot(), b2.data(), T()) << std::endl;
			
	}
		
	
	
	double MPIsum = 0.0;
	
	MPI_Allreduce(&sum,&MPIsum,1,MPI_DOUBLE,MPI_SUM,t1.comm());
	
	return MPIsum;
		
}*/ /*

*/
template <int N, typename T>
T RMS(tensor<N,T>& t_in) {
	
	T prod = dot(t_in, t_in);
	
	// get total number of elements
	auto tot = t_in.nfull_total();
	
	size_t ntot = 1.0;
	for (auto i : tot) {
		ntot *= i;
	}
	
	return sqrt(prod/ntot);
	
}


/*
template <typename T>
vec<T> diag(tensor<2,T>& t) {
	
	int myrank = -1;
	int commsize = 0;
	
	MPI_Comm_rank(t.comm(), &myrank); 
	MPI_Comm_size(t.comm(), &commsize);
	
	auto nfull = t.nfull_tot();
	
	int n = nfull[0];
	int m = nfull[1];
	
	if (n != m) throw std::runtime_error("Cannot take diagonal of non-square matrix.");
	
	vec<T> dvec(n, T());
	
	auto blksize = t.blk_size();
	auto blkoff = t.blk_offset();
	
	// loop over diagonal blocks
	for (int D = 0; D != blksize[0].size(); ++D) {
		
		int proc = -1;
		idx2 idx = {D,D};
		
		//std::cout << "BLOCK: " << D << " " << D << std::endl;
		
		t.get_stored_coordinates({.idx = idx, .proc = proc});
		
		//std::cout << "RANK: " << proc << std::endl;
		
		block<2,T> blk;
		int size = blksize[0][D];
		
		int off = blkoff[0][D];
		
		if (proc == myrank) {
			
			bool found = false;
			blk = t.get_block({.idx = idx, .blk_size = {size,size}, .found = found});	
			
			if (found) {
				//std::cout << "FOUND" << std::endl;
			
				for (int d = 0; d != size; ++d) {
					dvec[off + d] = blk(d,d);
				}
				
			}
			
		}
			
		MPI_Bcast(&dvec[off],size,MPI_DOUBLE,proc,t.comm());
		
	}
	
	//print(t);
	
	//if (myrank == 0) {
	//	for (auto x : dvec) {
	//		std::cout << x << " ";
	//	} std::cout << std::endl;
	//}
	
	return dvec;
	
}
                      
*/

} // end namespace dbcsr

#endif
