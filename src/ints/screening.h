#ifndef INTS_SCREENING_H
#define INTS_SCREENING_H

#include "ints/aofactory.h"

#include <Eigen/Core>
#include <utility>
#include <string>

// screening classes, inspired by MPQC

namespace ints {

class screener {
protected:

	dbcsr::world m_world;
	desc::smolecule m_mol;
	
	aofactory m_fac;
	
	double m_blk_threshold = dbcsr::global::filter_eps;
	double m_int_threshold = global::precision;
	
public:

	screener(dbcsr::world w, desc::smolecule mol, std::string method) : 
		m_world(w), m_mol(mol), m_fac(mol, w) {}
		
	virtual void compute() = 0;
	
	virtual bool skip_block_xbb(int i, int j, int k) = 0;
	virtual bool skip_xbb(int i, int j, int k) = 0;
	
	virtual bool skip_block_bbbb(int i, int j, int k, int l) = 0;
	virtual bool skip_bbbb(int i, int j, int k, int l) = 0;
		
	~screener() {}
	
};

class schwarz_screener : public screener {
protected:

	Eigen::MatrixXd m_blk_norms_mn;
	Eigen::MatrixXd m_blk_norms_x;
	Eigen::MatrixXd m_z_mn;
	Eigen::MatrixXd m_z_x;
		
public:

	schwarz_screener(dbcsr::world w, desc::smolecule mol) : 
		screener(w, mol, "schwarz") {}
		
	void compute() override;
		
	bool skip_block_xbb(int i, int j, int k) override;
	bool skip_xbb(int i, int j, int k) override;
	
	bool skip_block_bbbb(int i, int j, int k, int l) override;
	bool skip_bbbb(int i, int j, int k, int l) override;
	
	~schwarz_screener() {}
	
};

using shared_screener = std::shared_ptr<screener>;

} // end namespace

#endif
