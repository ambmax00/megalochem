#ifndef HF_GUESS_H
#define HF_GUESS_H

#include "desc/molecule.h"
#include "desc/options.h"
#include "math/tensor/dbcsr.hpp"

namespace hf {
	
class guess {
private:

	desc::molecule& m_mol;
	desc::options& m_opt;
	MPI_Comm m_comm;
	
	dbcsr::tensor<2,double> c_ob;
	dbcsr::tensor<2,double> p_bb;
	
public:

	guess(desc::molecule& mol, MPI_Comm& comm) : m_mol(mol), m_opt(opt), m_comm(comm) {}
	~guess() {}
	
	void compute(dbcsr::tensor<2,double>& Hcore);
	//void compute(desc::molecule& mol);
	
};

}
