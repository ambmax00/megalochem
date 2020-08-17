#ifndef MATH_BLOCK_MAP_H
#define MATH_BLOCK_MAP_H

#include <map>
#include <array>
#include <list>
#include <algorithm>
#include <functional>
#include <numeric>
#include <libxsmm.h>

namespace math {
	
struct mm_stack {

	std::map<std::array<int,3>,std::vector<std::array<double*,3>>> map;
	
	void insert(int m, int n, int k, double* a, double* b, double* c) {
		
		std::array<double*,3> ptrs = {a,b,c};
		std::array<int,3> idx = {m,n,k};
		
		auto iter = map.find(idx);
		
		if (iter == map.end()) {
			std::vector<std::array<double*,3>> v = {ptrs};
			map[idx] = v;
		} else {
			iter->second.push_back(ptrs);
		}
	}
	
	void process(double alpha, double beta) {
		
		auto print = [](double* p, int m, int n) {
			for (int i = 0; i != m; ++i) {
				for  (int j = 0; j != n; ++j) {
					std::cout << p[i + j*m] << " ";
				} std::cout << std::endl;
			}
		};
		
		for (auto& p : map) {
				
			int m = p.first[0];
			int n = p.first[1];
			int k = p.first[2];
				
			libxsmm_mmfunction<double> xmm(0, m, n, k, alpha, beta);

			std::cout << p.first[0] << " " << p.first[1] << " " << p.first[2] << " : " 
				<< p.second.size() << std::endl;
				
			for (auto& mms : p.second) {
				
				std::cout << "A" << std::endl;
				print(mms[0],m,k);
				std::cout << "B" << std::endl;
				print(mms[1],k,n);
				std::cout << "C" << std::endl;
				print(mms[2],m,n);
		
				xmm(mms[0], mms[1], mms[2]);
				
				std::cout << "Cres" << std::endl;
				print(mms[2],m,n);
				
			}
			 	
		}
		
	}
	
};

// economy single-thread sparse block tensor class

template <int N>
using bmap = std::map<const std::array<int,N>,size_t>;

template <int N>
using arrvec = std::array<std::vector<int>,N>;

template <int N, typename T = double>
struct blockmap {
private:

	arrvec<N> m_blksizes;
	std::vector<double> m_storage;
	bmap<N> map;

public:

	blockmap() {}
	
	blockmap(arrvec<N>& blksizes) : 
		m_blksizes(blksizes),
		m_storage(0) {}
	
	~blockmap() { clear(); }
	
	void clear() { 
		m_storage.clear();
		m_storage.shrink_to_fit();
		map.clear();
	}
	
	void reserve(arrvec<N>& indices) {
		
		size_t nblks = indices[0].size();
		size_t offset = m_storage.size();
		
		for (size_t iblk = 0; iblk != nblks; ++iblk) {
			
			std::array<int,N> blkidx;
			size_t blksize = 1;
			
			for (int n = 0; n != N; ++n) {
				blkidx[n] = indices[n][iblk];
				std::cout << n << " " << blkidx[n] << " " << m_blksizes[n][blkidx[n]] << std::endl;
 				blksize *= m_blksizes[n][blkidx[n]];
			}
			
			std::cout << blksize << std::endl;
			
			map[blkidx] = offset;
			offset += blksize;
			
		}
		
		m_storage.resize(offset);
		
	}
	
	void delete_block(const std::array<int,N>& idx) {
		map[idx] = -1;
	}
		
	void filter(T eps) {
		
		for (auto& blk : map) {
			
			auto& idx = blk.first;
			auto& offset = blk.second;
			
			auto bsize = get_blksize(idx);
			
			double blknorm = 0.0;
			
			for (int i = 0; i != bsize; ++i) {
				blknorm += pow(m_storage[offset + i],2.0);
			}
			 
			if (sqrt(blknorm) <= eps) blk.second = -1;
		}
		
		finalize();
		
	}
	
	inline size_t get_blksize(const std::array<int,N>& idx) {
		
		size_t blksize = 1;
			
		for (int n = 0; n != N; ++n) {
			blksize *= m_blksizes[n][idx[n]];
		}
		
		return blksize;
		
	}
	
	void finalize(bool reshuffle = true) {
		
		size_t newsize = 0;
		
		for (auto it = map.begin(); it != map.end();) {
			if (it->second == -1) {
				std::cout << "ERASED " << it->first[0] << " " 
				<< it->first[1] << " " << it->first[2] << std::endl;
				it = map.erase(it);
			} else {
				newsize += get_blksize(it->first);
				++it;
			}
		}
		
		if (reshuffle) {
		
			std::vector<double> newvector(newsize);
			size_t offset = 0;
			
			for (auto& blk : map) {
				size_t blksize = get_blksize(blk.first);
				size_t blkoffset = blk.second;
				
				std::copy(m_storage.begin() + blkoffset,
					m_storage.begin() + blkoffset + blksize,
					newvector.begin() + offset);
				
				blk.second = offset;
				
				offset += blksize;
				
			}
			
			m_storage = newvector;
			
		}
		
	}
	
	inline T* get_block(const std::array<int,N>& idx) {
		auto it = map.find(idx);
		return (it != map.end()) ? &m_storage[it->second] : nullptr;
	}
	
	void print() {
		for (auto& p : map) {
			std::cout << "[";
			for (int i = 0; i != N; ++i) {
				std::cout << p.first[i] << " ";
			} std::cout << "]";
			
			std::cout << "(";
			for (int i = 0; i != N; ++i) {
				std::cout << m_blksizes[i][p.first[i]] << " ";
			} std::cout << ")";
			
			auto blksize = get_blksize(p.first);
			T* data = &m_storage[0] + p.second;
			
			std::cout << "{";
			for (int i = 0; i != blksize; ++i) {
				std::cout << data[i] << " ";
			} std::cout << "}" << std::endl;
			
		}
	}
	
	inline typename bmap<N>::iterator begin() { return map.begin(); }	
	inline typename bmap<N>::iterator end() { return map.end(); } 
	
	arrvec<N> blk_sizes() {
		return m_blksizes;
	}
	
	void insert(blockmap& b_in) {
		
		map.insert(b_in.map.begin(), b_in.map.end());
		m_storage.insert(m_storage.end(), b_in.m_storage.begin(), b_in.m_storage.end());
		
		b_in.clear();
		
	}
	
	int num_blocks() {
		return map.size();
	}
	
	T* get_data(size_t off) {
		return &m_storage[off];
	}
	
};

} // end namespace

#endif
