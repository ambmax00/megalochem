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
class tensor_copy_base {

    #:set list = [ &
        ['order','vec<int>','optional','val'],&
        ['sum','bool','optional','val'],&
        ['bounds','vec<vec<int>>', 'optional', 'ref'],&
        ['move_data','bool','optional','val']]
    ${make_param('tensor_copy_base',list)}$
        
private:

    tensor<N,T>& c_t_in;
    tensor<N,T>& c_t_out;
    
public:

    tensor_copy_base(tensor<N,T>& t1, tensor<N,T>& t2) : c_t_in(t1), c_t_out(t2) {}
    
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
tensor_copy_base(tensor<N,T>& t1, tensor<N,T>& t2) 
	-> tensor_copy_base<tensor<N,T>::dim, typename tensor<N,T>::value_type>;

template <typename tensortype>
inline tensor_copy_base<tensortype::dim, typename tensortype::value_type>
copy(tensortype& t1, tensortype& t2) {
	return tensor_copy_base(t1,t2);
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
        
        if (
			(c_bounds1 && c_bounds1->size() != c_con1->size()) ||
			(c_bounds2 && c_bounds2->size() != c_ncon1->size()) ||
			(c_bounds3 && c_bounds3->size() != c_ncon2->size()) 
		){
			throw std::runtime_error("Wrong bound dimensions.");
		}
        
        int rank = -1;
        MPI_Comm_rank(c_t1.comm(),&rank);
        
        int out = (rank == 0) ? 6 : -1;
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
            (c_filter) ? &*c_filter : nullptr,
            (c_flop) ? &*c_flop : nullptr,
            (c_move) ? &*c_move : nullptr,
            (c_retain_sparsity) ? &*c_retain_sparsity : nullptr,
            unit_nr, (c_log) ? &*c_log : nullptr);
            
        if (f_b1) delete [] f_b1;
        if (f_b2) delete [] f_b2;
        if (f_b3) delete [] f_b3;
             
    }
    
    void perform (std::string formula) {
        
        eval(formula);
        
        perform();
        
    }
    
    arrvec<int,N3> get_index() {
		
		long long int nblkmax = c_dbcsr_t_max_nblks_local(c_t3.m_tensor_ptr);
		int* indices = new int[N3*nblkmax];
		int nblkloc = 0;
		
		int* f_b1 = (c_bounds1) ? unfold_bounds<int>(*c_bounds1) : nullptr;
        int* f_b2 = (c_bounds2) ? unfold_bounds<int>(*c_bounds2) : nullptr;
        int* f_b3 = (c_bounds3) ? unfold_bounds<int>(*c_bounds3) : nullptr;
        
        int rank = -1;
        MPI_Comm_rank(c_t1.comm(),&rank);
        
        int out = (rank == 0) ? 6 : -1;
        int* unit_nr = (c_print) ? ((*c_print) ? &out : nullptr) : nullptr;  
        
        c_dbcsr_t_contract_index_r_dp(
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
            (c_filter) ? &*c_filter : nullptr,
            &nblkloc, indices, nblkmax, N3);
            
        if (f_b1) delete [] f_b1;
        if (f_b2) delete [] f_b2;
        if (f_b3) delete [] f_b3;
        
		arrvec<int,N3> idx_out;
		
		for (int i = 0; i != N3; ++i) {
			idx_out[i] = vec<int>(nblkloc);
			std::copy(indices + i*nblkmax, indices + i*nblkmax + nblkloc, idx_out[i].data());
		}
		
		delete[] indices;
		
		return idx_out;
		
	}  
	
	arrvec<int,N3> get_index(std::string formula) {
        
        eval(formula);
        return get_index();
        
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
void copy_matrix_to_3Dtensor_new(matrix<T>& m, tensor<3,T>& t, bool sym = false) {
	
	auto w = m.get_world();
	
	int mpirank = w.rank();
	int mpisize = w.size();
	auto comm = w.comm();
	
	vec<int> send_nblk_p(mpisize,0);
	vec<int> send_nze_p(mpisize,0);
	vec<vec<int>> send_idx0_p(mpisize);
	vec<vec<int>> send_idx1_p(mpisize);
	
	matrix<T>* m_ptr;
	
	if (sym) { 
		m_ptr = new matrix<T>(std::move(*m.desymmetrize()));
	} else {
		m_ptr = &m;
	}
	
	iterator iter(*m_ptr);
	idx3 idxt = {0,0,0};
	
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next_block();
		
		idxt[0] = iter.row();
		idxt[1] = iter.col();
		
		int nze = iter.row_size() * iter.col_size();
	
		int dest_p = t.proc(idxt);
		
		send_nblk_p[dest_p] += 1;
		send_nze_p[dest_p] += nze;
		send_idx0_p[dest_p].push_back(idxt[0]);
		send_idx1_p[dest_p].push_back(idxt[1]);
		
	}
	
	iter.stop();
	
	vec<int> recv_nblk_p(mpisize);
	vec<int> recv_nze_p(mpisize);
	
	// send info around
	
	for (int ip = 0; ip != mpisize; ++ip) {
		
		MPI_Gather(&send_nblk_p[ip],1,MPI_INT,recv_nblk_p.data(),1,
			MPI_INT,ip,comm);
			
		MPI_Gather(&send_nze_p[ip],1,MPI_INT,recv_nze_p.data(),1,
			MPI_INT,ip,comm);
			
	}
	
	// allocate space on sender
	
	vec<vec<double>> send_blk_data(mpisize);
	vec<int> send_blk_offset(mpisize,0);
	
	for (int ip = 0; ip != mpisize; ++ip) {
		send_blk_data[ip].resize(send_nze_p[ip]);
	}
	
	// copy blocks
	
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next_block();
		
		idxt[0] = iter.row();
		idxt[1] = iter.col();
		
		int rsize = iter.row_size();
		int csize = iter.col_size();
		
		int nze = rsize*csize;
		int dest_p = t.proc(idxt);
		
		std::copy(iter.data(),iter.data()+nze,
			send_blk_data[dest_p].begin() + send_blk_offset[dest_p]);
			
		send_blk_offset[dest_p] += nze;
		
	}
	
	iter.stop();
		
	// allocate space on receiver 
	
	int recv_blktot = std::accumulate(recv_nblk_p.begin(),recv_nblk_p.end(),0);
	int recv_nzetot = std::accumulate(recv_nze_p.begin(),recv_nze_p.end(),0);
	
	vec<int> recv_blk_offset(mpisize);
	vec<int> recv_nze_offset(mpisize);
	
	int blkoffset = 0;
	int nzeoffset = 0;
	
	for (int ip = 0; ip != mpisize; ++ip) {
		recv_blk_offset[ip] = blkoffset;
		recv_nze_offset[ip] = nzeoffset;
		
		blkoffset += recv_nblk_p[ip];
		nzeoffset += recv_nze_p[ip];
	}
	
	arrvec<int,3> recv_blkidx;
	vec<double> recv_blk_data(recv_nzetot);
	
	for (auto& v : recv_blkidx) {
		v.resize(recv_blktot);
	}
	
	// send over block indices
	
	for (int ip = 0; ip != mpisize; ++ip) {
	
		MPI_Gatherv(send_idx0_p[ip].data(),send_nblk_p[ip],MPI_INT,
			recv_blkidx[0].data(),recv_nblk_p.data(),recv_blk_offset.data(),MPI_INT,ip,comm);
			
		MPI_Gatherv(send_idx1_p[ip].data(),send_nblk_p[ip],MPI_INT,
			recv_blkidx[1].data(),recv_nblk_p.data(),recv_blk_offset.data(),MPI_INT,ip,comm);
			
	}
	
	// send over block data
	
	for (int ip = 0; ip != mpisize; ++ip) {
		
		MPI_Gatherv(send_blk_data[ip].data(),send_nze_p[ip],MPI_DOUBLE,
			recv_blk_data.data(),recv_nze_p.data(),recv_nze_offset.data(),
			MPI_DOUBLE,ip,comm);
			
	}
	
	// allocate blocks
	
	t.reserve(recv_blkidx);
	
	nzeoffset = 0;
	
	auto rowblksizes = m_ptr->row_blk_sizes();
	auto colblksizes = m_ptr->col_blk_sizes();
	
	idx3 sizes = {1,1,1};
	
	for (int iblk = 0; iblk != recv_blkidx[0].size(); ++iblk) {
		
		idxt[0] = recv_blkidx[0][iblk];
		idxt[1] = recv_blkidx[1][iblk];
		
		//std::cout << idxt[0] << " " << idxt[1] << std::endl;
		
		sizes[0] = rowblksizes[idxt[0]];
		sizes[1] = colblksizes[idxt[1]];
		
		T* ptr = recv_blk_data.data() + nzeoffset;
		
		t.put_block(idxt, ptr, sizes);
		
		nzeoffset += sizes[0]*sizes[1];
		
	}
	
	if (sym) { 
		m_ptr->release();
		delete m_ptr;
	}
	
	//dbcsr::print(*m_ptr);
	//dbcsr::print(t);
	
	//exit(0);
		
}

template <typename T>
void copy_3Dtensor_to_matrix_new(tensor<3,T>& t, matrix<T>& m) {
	
	auto w = m.get_world();
	
	int mpirank = w.rank();
	int mpisize = w.size();
	auto comm = w.comm();
	
	vec<int> send_nblk_p(mpisize,0);
	vec<int> send_nze_p(mpisize,0);
	vec<vec<int>> send_idx0_p(mpisize);
	vec<vec<int>> send_idx1_p(mpisize);
	
	iterator_t<3,T> itert(t);
	
	itert.start();
	
	while (itert.blocks_left()) {
		
		itert.next();
		auto& idxt = itert.idx();
		auto& size = itert.size();
		
		int nze = size[0] * size[1];
	
		int dest_p = m.proc(idxt[0],idxt[1]);
		
		send_nblk_p[dest_p] += 1;
		send_nze_p[dest_p] += nze;
		send_idx0_p[dest_p].push_back(idxt[0]);
		send_idx1_p[dest_p].push_back(idxt[1]);
		
	}
	
	itert.stop();
	
	vec<int> recv_nblk_p(mpisize);
	vec<int> recv_nze_p(mpisize);
	
	// send info around
	
	/*MPI_Barrier(comm);
	
	for (int i = 0; i != mpisize; ++i) {
		
		if (i == mpirank) {
			std::cout << "RANK " << i << std::endl;
			std::cout << "IDX" << std::endl;
			for (auto m : send_idx0_p) {
				for (auto s : m) {
					std::cout << s << " ";
				} std::cout << std::endl;
			}
			for (auto m : send_idx1_p) {
				for (auto s : m) {
					std::cout << s << " ";
				} std::cout << std::endl;
			}
		}
		
		MPI_Barrier(comm);	
		
	}
	
	MPI_Barrier(comm);
	
	for (int i = 0; i != mpisize; ++i) {
		
		if (i == mpirank) {
			for (auto m : send_nblk_p) {
				std::cout << m << " "; 
			} std::cout << '\n';
			for (auto m : send_nze_p) {
				std::cout << m << " "; 
			} std::cout << std::endl;
		}
		
		MPI_Barrier(comm);	
		
	}*/
	
	for (int ip = 0; ip != mpisize; ++ip) {
		
		MPI_Gather(&send_nblk_p[ip],1,MPI_INT,recv_nblk_p.data(),1,
			MPI_INT,ip,comm);
			
		MPI_Gather(&send_nze_p[ip],1,MPI_INT,recv_nze_p.data(),1,
			MPI_INT,ip,comm);
			
	}
	
	/*for (int i = 0; i != mpisize; ++i) {
		
		if (i == mpirank) {
			for (auto m : recv_nblk_p) {
				std::cout << m << " "; 
			} std::cout << '\n';
			for (auto m : recv_nze_p) {
				std::cout << m << " "; 
			} std::cout << std::endl;
		}
		
		MPI_Barrier(comm);	
		
	}*/
	
	// allocate space on sender
	
	vec<vec<double>> send_blk_data(mpisize);
	vec<int> send_blk_offset(mpisize,0);
	
	for (int ip = 0; ip != mpisize; ++ip) {
		send_blk_data[ip].resize(send_nze_p[ip]);
	}
	
	// copy blocks
	
	itert.start();
	
	while (itert.blocks_left()) {
		
		itert.next();
		
		auto& idxt = itert.idx();
		auto& size = itert.size();
		
		int nze = size[0] * size[1];
		int dest_p = m.proc(idxt[0],idxt[1]);
		
		bool found;
		auto blk = t.get_block(idxt,size,found);
		
		std::copy(blk.data(),blk.data()+nze,
			send_blk_data[dest_p].begin() + send_blk_offset[dest_p]);
			
		send_blk_offset[dest_p] += nze;
		
	}
	
	
	/*for (int i = 0; i != mpisize; ++i) {
		
		if (i == mpirank) {
			std::cout << "RANK " << i << std::endl;
			for (auto m : send_blk_data) {
				for (auto s : m) {
					std::cout << s << " ";
				} std::cout << std::endl;
			}
		}
		
		MPI_Barrier(comm);	
		
	}*/
	
	itert.stop();
		
	// allocate space on receiver 
	
	int recv_blktot = std::accumulate(recv_nblk_p.begin(),recv_nblk_p.end(),0);
	int recv_nzetot = std::accumulate(recv_nze_p.begin(),recv_nze_p.end(),0);
	
	vec<int> recv_blk_offset(mpisize);
	vec<int> recv_nze_offset(mpisize);
	
	int blkoffset = 0;
	int nzeoffset = 0;
	
	for (int ip = 0; ip != mpisize; ++ip) {
		recv_blk_offset[ip] = blkoffset;
		recv_nze_offset[ip] = nzeoffset;
		
		blkoffset += recv_nblk_p[ip];
		nzeoffset += recv_nze_p[ip];
	}
	
	/*MPI_Barrier(comm);
	
	std::cout << "OFFSETS" << std::endl;
	for (int i = 0; i != mpisize; ++i) {
		
		if (i == mpirank) {
			std::cout << "RANK " << i << std::endl;
			for (auto m : recv_blk_offset) {
					std::cout << m << " ";
			} std::cout << std::endl;
		}
		
		MPI_Barrier(comm);	
		
	}*/
	
	arrvec<int,2> recv_blkidx;
	vec<double> recv_blk_data(recv_nzetot);
	
	for (auto& v : recv_blkidx) {
		v.resize(recv_blktot);
	}
	
	// send over block indices
	
	for (int ip = 0; ip != mpisize; ++ip) {
	
		MPI_Gatherv(send_idx0_p[ip].data(),send_nblk_p[ip],MPI_INT,
			recv_blkidx[0].data(),recv_nblk_p.data(),recv_blk_offset.data(),MPI_INT,ip,comm);
			
		MPI_Gatherv(send_idx1_p[ip].data(),send_nblk_p[ip],MPI_INT,
			recv_blkidx[1].data(),recv_nblk_p.data(),recv_blk_offset.data(),MPI_INT,ip,comm);
			
	}
	
	/*MPI_Barrier(comm);
	
	for (int i = 0; i != mpisize; ++i) {
		
		if (i == mpirank) {
			std::cout << "RANK " << i << std::endl;
			for (auto m : recv_blkidx) {
				for (auto s : m) {
					std::cout << s << " ";
				} std::cout << std::endl;
			}
		}
		
		MPI_Barrier(comm);	
		
	}
	
	MPI_Barrier(comm);*/
	
	// send over block data
	
	for (int ip = 0; ip != mpisize; ++ip) {
		
		MPI_Gatherv(send_blk_data[ip].data(),send_nze_p[ip],MPI_DOUBLE,
			recv_blk_data.data(),recv_nze_p.data(),recv_nze_offset.data(),
			MPI_DOUBLE,ip,comm);
			
	}
	
	/*MPI_Barrier(comm);
	
	for (int i = 0; i != mpisize; ++i) {
		
		if (i == mpirank) {
			std::cout << "RANK " << i << std::endl;
			for (auto m : recv_blk_data) {
					std::cout << m << " ";
			} std::cout << std::endl;
		}
		
		MPI_Barrier(comm);	
		
	}
	
	MPI_Barrier(comm);*/
	
	//std::cout << "0 " << recv_blk_data[0] << std::endl;
	
	// allocate blocks
	// check if symmetric
	
	vec<int> rowres, colres;
	
	bool sym = (m.matrix_type() == dbcsr::type::symmetric) ? true : false;
	
	if (sym) {
		
		rowres.reserve(recv_blkidx[0].size());
		colres.reserve(recv_blkidx[1].size());
		
		for (int i = 0; i != recv_blkidx[0].size(); ++i) {
			int ix = recv_blkidx[0][i];
			int jx = recv_blkidx[1][i];
			
			if (ix <= jx) {
				rowres.push_back(ix);
				colres.push_back(jx);
			}
			
		}
		
	} else {
		
		rowres.resize(recv_blkidx[0].size());
		colres.resize(recv_blkidx[1].size());
		
		std::copy(recv_blkidx[0].begin(),recv_blkidx[0].end(),
			rowres.begin());
		std::copy(recv_blkidx[1].begin(),recv_blkidx[1].end(),
			colres.begin());
		
	}
	
	m.reserve_blocks(rowres,colres);
	nzeoffset = 0;
	
	auto rowblksizes = m.row_blk_sizes();
	auto colblksizes = m.col_blk_sizes();
	
	for (int iblk = 0; iblk != recv_blkidx[0].size(); ++iblk) {
		
		int ix = recv_blkidx[0][iblk];
		int jx = recv_blkidx[1][iblk];
		
		int rowsize = rowblksizes[ix];
		int colsize = colblksizes[jx];
		
		T* ptr = recv_blk_data.data() + nzeoffset;
		nzeoffset += rowsize * colsize;
		
		if (sym && ix > jx) continue;
		
		m.put_block_p(ix, jx, ptr, rowsize, colsize);
		
	}
	
	//dbcsr::print(t);
	//dbcsr::print(m);
		
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

template <int N, typename T>
void copy_local_to_global(tensor<N,T>& t_loc, tensor<N,T>& t_glob) {
	
	MPI_Comm comm_glob = t_glob.comm();
	
	int mpirank = -1;
	int mpisize = -1;
	
	MPI_Comm_rank(comm_glob, &mpirank);
	MPI_Comm_size(comm_glob, &mpisize);
	
	auto blksizes_glob = t_glob.blk_sizes();
	auto blksizes_loc = t_loc.blk_sizes();
	
	for (int in = 0; in != N; ++in) {
		if (blksizes_glob[in] != blksizes_loc[in]) {
			throw std::runtime_error("Tensor copy: icompatible block sizes.");
		}
	}
	
	vec<int> send_nblk_p(mpisize,0);
	vec<int> send_nze_p(mpisize,0);
	
	vec<arrvec<int,N>> send_idx_p(mpisize);
	
	dbcsr::iterator_t<N> iter(t_loc);
	
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next();
		auto& idx = iter.idx();
		auto& size = iter.size();
		
		int nze = std::accumulate(&size[0], &size[0] + N, 1,
			std::multiplies<int>());
	
		int dest_p = t_glob.proc(idx);
		
		send_nblk_p[dest_p] += 1;
		send_nze_p[dest_p] += nze;
		
		for (int i = 0; i != N; ++i) { 
			send_idx_p[dest_p][i].push_back(idx[i]);
		}
		
	}
	
	iter.stop();
	
	vec<int> recv_nblk_p(mpisize);
	vec<int> recv_nze_p(mpisize);
	
	// send info around
	
	for (int ip = 0; ip != mpisize; ++ip) {
		
		MPI_Gather(&send_nblk_p[ip],1,MPI_INT,recv_nblk_p.data(),1,
			MPI_INT,ip,comm_glob);
			
		MPI_Gather(&send_nze_p[ip],1,MPI_INT,recv_nze_p.data(),1,
			MPI_INT,ip,comm_glob);
			
	}
	
	// allocate space on sender
	
	vec<vec<double>> send_blk_data(mpisize);
	vec<int> send_blk_offset(mpisize,0);
	
	for (int ip = 0; ip != mpisize; ++ip) {
		send_blk_data[ip].resize(send_nze_p[ip]);
	}
	
	// copy blocks
	
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next();
		
		auto& idx = iter.idx();
		auto& size = iter.size();
		
		int dest_p = t_glob.proc(idx);
		
		bool found = true;
		auto blk = t_loc.get_block(idx, size, found);
		
		std::copy(blk.data(), blk.data()+blk.ntot(),
			send_blk_data[dest_p].begin() + send_blk_offset[dest_p]);
			
		send_blk_offset[dest_p] += blk.ntot();
		
	}
	
	iter.stop();
		
	// allocate space on receiver 
	
	int recv_blktot = std::accumulate(recv_nblk_p.begin(),recv_nblk_p.end(),0);
	int recv_nzetot = std::accumulate(recv_nze_p.begin(),recv_nze_p.end(),0);
	
	vec<int> recv_blk_offset(mpisize);
	vec<int> recv_nze_offset(mpisize);
	
	int blkoffset = 0;
	int nzeoffset = 0;
	
	for (int ip = 0; ip != mpisize; ++ip) {
		recv_blk_offset[ip] = blkoffset;
		recv_nze_offset[ip] = nzeoffset;
		
		blkoffset += recv_nblk_p[ip];
		nzeoffset += recv_nze_p[ip];
	}
	
	arrvec<int,N> recv_blkidx;
	vec<double> recv_blk_data(recv_nzetot);
	
	for (auto& v : recv_blkidx) {
		v.resize(recv_blktot);
	}
	
	// send over block indices
	
	for (int ip = 0; ip != mpisize; ++ip) {
		for (int in = 0; in != N; ++in) {
			
			MPI_Gatherv(send_idx_p[ip][in].data(),send_nblk_p[ip],MPI_INT,
				recv_blkidx[in].data(),recv_nblk_p.data(),recv_blk_offset.data(),
				MPI_INT,ip,comm_glob);
		}
	}	
	
	// send over block data
	
	for (int ip = 0; ip != mpisize; ++ip) {
		
		MPI_Gatherv(send_blk_data[ip].data(),send_nze_p[ip],MPI_DOUBLE,
			recv_blk_data.data(),recv_nze_p.data(),recv_nze_offset.data(),
			MPI_DOUBLE,ip,comm_glob);
			
	}
	
	// allocate blocks
	
	t_glob.reserve(recv_blkidx);
	
	nzeoffset = 0;
	
	std::array<int,N> idxt, sizet;
	
	for (int iblk = 0; iblk != recv_blkidx[0].size(); ++iblk) {
		
		for (int in = 0; in != N; ++in) {
			idxt[in] = recv_blkidx[in][iblk];
			sizet[in] = blksizes_glob[in][idxt[in]];
		}
		
		T* ptr = recv_blk_data.data() + nzeoffset;
		
		dbcsr::block<N,T> blk(sizet, ptr);
		
		t_glob.put_block(idxt, blk);
		
		nzeoffset += blk.ntot();
		
	}
		
}

} // end namespace dbcsr

#endif
