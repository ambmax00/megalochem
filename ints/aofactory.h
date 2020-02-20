#ifndef INTS_AOFACTORY_H
#define INTS_AOFACTORY_H

#include "math/tensor/dbcsr.hpp"
#include "desc/molecule.h"
#include <string>
#include <map>
#include <mpi.h>

/* Loads the AO integrals
 * Op: Operator (Coulom, kinetic, ...)
 * bis: tensor space, e.g. bb is a matrix in the AO basis, xbb are the 3c2e integrals etc...
*/

namespace ints {

class Zmat;

/*
class registry {
private:

	std::map<std::string, dbcsr::tensor<2>*> t2;
	std::map<std::string, dbcsr::tensor<3>*> t3;
	std::map<std::string, dbcsr::tensor<4>*> t4;

public:

	registry() {}

	~registry() {
		for (auto& m2 : t2) { m2.second->destroy(); delete m2.second; }
		for (auto& m3 : t3) { m3.second->destroy(); delete m3.second; }
		for (auto& m4 : t4) { m4.second->destroy(); delete m4.second; }
	}	
	
	dbcsr::tensor<2>* get(std::string key) {
		if (t2.find(key) != it.end()) return t2[key];
		return nullptr;
	}
	dbcsr::tensor<3>* get(std::string key) {
		if (t3.find(key) != it.end()) return t3[key];
		return nullptr;
	}
	dbcsr::tensor<4>* get(std::string key) {
		if (t4.find(key) != it.end()) return t4[key];
		return nullptr;
	}
	
	void put(dbcsr::tensor<2>* t_in, std::string key) {
		if (t2.find(key) != it.end()) throw std::runtime_error("Already in integrals registry: "+key);
		t2[key] = t_in;
	}
	void put(dbcsr::tensor<3>* t_in, std::string key) {
		if (t3.find(key) != it.end()) throw std::runtime_error("Already in integrals registry: "+key);
		t3[key] = t_in;
	}
	void put(dbcsr::tensor<4>* t_in, std::string key) {
		if (t4.find(key) != it.end()) throw std::runtime_error("Already in integrals registry: "+key);
		t4[key] = t_in;
	}
		
}
*/

class aofactory {
	
private:
	
	desc::molecule& m_mol;
	MPI_Comm m_comm;
	
public:

	aofactory(desc::molecule& mol, MPI_Comm c) : m_mol(mol), m_comm(c)  {};
	
	~aofactory() {};
	
	//static registry reg;
	
	struct aofac_params {
		required<std::string,val> 	op;
		required<std::string,val> 	bas;
		required<std::string,val>	name;
		required<vec<int>,val>		map1;
		required<vec<int>,val>		map2;
	};
	
	template <int N>
	dbcsr::stensor<N,double> compute(aofac_params&& p);
	
	optional<Zmat,val> m_2e_ints_zmat;
	optional<Zmat,val> m_3c2e_ints_zmat;
		
}; // end class aofactory

} // end namespace ints
			
		
#endif
