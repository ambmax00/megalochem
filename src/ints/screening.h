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

	std::shared_ptr<aofactory> p_fac;
	std::string m_method;
	
	double m_blk_threshold = dbcsr::global::filter_eps;
	double m_int_threshold = global::precision;
	
public:

	screener(std::shared_ptr<aofactory> ifac, std::string method) : 
		p_fac(ifac), m_method(method) {}
	
	virtual void compute() = 0;
	
	virtual bool skip_block(int i, int j, int k);
	virtual bool skip(int i, int j, int k);
	
	virtual double val_x(int i) { return 0; }
	virtual double val_bb(int i, int j) { return 0; }
	
	virtual double blknorm_x(int i) { return 0; }
	virtual double blknorm_bb(int i, int j) { return 0; }
	
	~screener() {}
	
};

class schwarz_screener : public screener {
protected:

	Eigen::MatrixXd m_blk_norms_mn;
	Eigen::MatrixXd m_blk_norms_x;
	Eigen::MatrixXd m_z_mn;
	Eigen::MatrixXd m_z_x;
	
	std::string m_metric;
	
public:

	schwarz_screener(std::shared_ptr<aofactory> ifac, std::string metric) : 
		screener(ifac, "schwarz") {}
		
	void compute() override;
	
	bool skip_block(int i, int j, int k) override;
	bool skip(int i, int j, int k) override;
	
	double val_x(int i) override { return m_z_x(i,0); }
	double val_bb(int i, int j) override  { return m_z_mn(i,j); }
	
	double blknorm_x(int i) override { return m_blk_norms_x(i,0); }
	double blknorm_bb(int i, int j) override  { return m_blk_norms_mn(i,j); }
	
	~schwarz_screener() {}
	
};

class ovlp_screener : public screener {
protected:

	Eigen::MatrixXd m_blk_norms_mn;
	
public:

	ovlp_screener(std::shared_ptr<aofactory> ifac) : 
		screener(ifac, "ovlp") {}
		
	void compute() override;
	
	bool skip_block(int i, int j, int k) override;
	bool skip(int i, int j, int k) override;
	
	double val_x(int i) override { return 0; }
	double val_bb(int i, int j) override  { return 0; }
	
	double blknorm_x(int i) override { return 1; }
	double blknorm_bb(int i, int j) override  { return m_blk_norms_mn(i,j); }
	
	~ovlp_screener() {}
	
};

using shared_screener = std::shared_ptr<screener>;

} // end namespace

#endif
