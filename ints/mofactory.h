#ifndef MO_FACTORY_H
#define MO_FACTORY_H 

#include "math/tensor/dbcsr.hpp"
#include "hf/

namespace ints {
	
class mofactory {
	
private:
	
	aofactory m_aofac;
	
public:

	mofactory(desc::molecule& mol, MPI_Comm c) : m_aofac(mol,c) {};
	
	~mofactory() {};
	
	/*
	struct mofac_params {
		required<std::string,val> 	op;
		required<std::string,val>	name;
		required<vec<int>,val>		map1;
		required<vec<int>,val>		map2;
		optional<dbcsr::stensor<2>,val> c1;
		optional<dbcsr::stensor<2>,val> c2;
		//optional<dbcsr::stensor<2>,val> c3;
		//optional<dbcsr::stensor<2>,val> c4;
	};
	
	template <int N>
	dbcsr::stensor<N,double> compute(mofac_params&& p);
	
	
	// BETTER: INTTRAN3 (name, t_in, c1, c2, b1, b2
	
	
	dbcsr::stensor<3> transform(std::string name, dbcsr::stensor<3>& t_ints, dbcsr::stensor<2>& c_1, dbcsr::stensor<2>& c_2);
	
}; // end class aofactory

} // end namespace ints
