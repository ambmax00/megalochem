#ifndef INTS_SCREENING_H
#define INTS_SCREENING_H

#include "ints/aofactory.hpp"

#include <Eigen/Core>
#include <utility>
#include <string>

// screening classes, inspired by MPQC

namespace megalochem {

namespace ints {

class screener {
protected:

	world m_world;
	dbcsr::cart m_cart;
	desc::shared_molecule m_mol;
	std::string m_method;
	
	aofactory m_fac;
	
	double m_blk_threshold = dbcsr::global::filter_eps;
	double m_int_threshold = global::precision;
	
public:

	screener(world w, desc::shared_molecule mol, std::string method) : 
		m_world(w), 
		m_cart(w.dbcsr_grid()), 
		m_mol(mol), 
		m_method(method),
		m_fac(mol, w) {}
		
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

	schwarz_screener(world w, desc::shared_molecule mol) : 
		screener(w, mol, "schwarz") {}
		
	void compute() override;
		
	bool skip_block_xbb(int i, int j, int k) override;
	bool skip_xbb(int i, int j, int k) override;
	
	bool skip_block_bbbb(int i, int j, int k, int l) override;
	bool skip_bbbb(int i, int j, int k, int l) override;
	
	~schwarz_screener() {}
	
};

class atomic_screener : public screener {
protected:

	std::vector<int> m_atom_list;
	
	std::vector<bool> m_blklist_b;
	std::vector<bool> m_blklist_x; 
	
	schwarz_screener m_schwarz;
	
public:

	atomic_screener(world w, desc::shared_molecule mol, std::vector<int> alist) :
		screener(w, mol, "atomic"), 
		m_atom_list(alist), 
		m_schwarz(w,mol) 
	{}
		
	void compute() override;
		
	bool skip_block_xbb(int i, int j, int k) override;
	bool skip_xbb(int i, int j, int k) override;
	
	bool skip_block_bbbb(int i, int j, int k, int l) override;
	bool skip_bbbb(int i, int j, int k, int l) override;
	
	std::vector<bool> blklist_b() { return m_blklist_b; }
	std::vector<bool> blklist_x() { return m_blklist_x; }
	
	~atomic_screener() {}
	
	
};

using shared_screener = std::shared_ptr<screener>;

} // end namespace

} // end namespace mega

#endif
