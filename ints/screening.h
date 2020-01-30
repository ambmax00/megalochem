#ifndef INTS_SCREENING_H
#define INTS_SCREENING_H

#include <mpi.h>
#include <string>

#include "desc/molecule.h"
#include "utils/pool.h"
#include "math/tensor/dbcsr.hpp"

#include <libint2.hpp>
#include <Eigen/Core>
#include <map>
#include <utility>

// screening classes, inspired by MPQC

namespace std
{
    template<typename T, size_t N>
    struct hash<array<T, N> >
    {
        typedef array<T, N> argument_type;
        typedef size_t result_type;

        result_type operator()(const argument_type& a) const
        {
            hash<T> hasher;
            result_type h = 0;
            for (result_type i = 0; i < N; ++i)
            {
                h = h * 31 + hasher(a[i]);
            }
            return h;
        }
    };
}

typedef std::unordered_map<dbcsr::idx2, std::pair<float,dbcsr::block<2>>> blockmap;

namespace ints {

// Matrix which holds info about screening of blocks and shells
class Zmat {
private:

	//???
	MPI_Comm m_comm;
	desc::molecule& m_mol;
	util::ShrPool<libint2::Engine>& m_eng;
	std::string m_method;
	
	vec<vec<int>> m_blk_sizes;
	
	// methods
	void compute_schwarz();
	// void compute_QQR
	
public:

	Zmat(MPI_Comm comm, desc::molecule& mol, util::ShrPool<libint2::Engine>& engine, std::string method)
		: m_comm(comm), m_mol(mol), m_eng(engine), m_method(method) {}
	
	void compute();
	
	~Zmat() {
		
		//for (auto m : m_blkmap) {
		//	delete m.second.second;;
		//}
	}
	
	// map storing the block index, and its associated norm and block
	blockmap m_blkmap;
	
};

} // end namespace

#endif
