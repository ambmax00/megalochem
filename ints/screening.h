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
	
	~screener() {}
	
};

class schwarz_screener : public screener {
protected:

	Eigen::MatrixXd m_blk_norms_mn;
	Eigen::MatrixXd m_blk_norms_x;
	Eigen::MatrixXd m_z_mn;
	Eigen::MatrixXd m_z_x;
	
public:

	schwarz_screener(std::shared_ptr<aofactory> ifac) : 
		screener(ifac, "schwarz") {}
		
	void compute() override;
	
	bool skip_block(int i, int j, int k) override;
	bool skip(int i, int j, int k) override;
	
	~schwarz_screener() {}
	
};

} // end namespace

#endif
