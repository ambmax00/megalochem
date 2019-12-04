#include "math/laplace/laplace.h"
#include <cmath>
#include <algorithm>

namespace math {

/*
myfloat Lap(myfloat* &omega, myfloat* &alpha, int k, myfloat x) {
	
	myfloat sum = mystrtof("0", NULL);
	
	for (int nu = 0; nu != k; ++nu) {
		sum += omega[nu]*myexp(-alpha[nu]*x);
	}
	
	return sum - mypow(x, -1);
}
*/

float128 laplace::deltak() {
	
		float128 maxval = 0;
	
		for (int n = 0; n != 2*m_k+1; ++n) {
			maxval = std::max(fabs(Eta(m_xi(n))), maxval);
		}
		//std::cout << "MAX is " << max << std::endl;
		return maxval;
}

float128 laplace::Eta(float128 x) {
	
	// CALCULATE THE ERROR DISTRIBUTION FUNCTION
	
	float128 sum = 0;
	
	for (int nu = 0; nu != m_k; ++nu) {
		sum += m_omega(nu)*exp(-m_alpha(nu)*x);
	}
	
	return sum - pow(x, -1);	
	
} 

float128 laplace::dEta(float128 x) {
	
	// CALCULATE THE FIRST DERIVATIVE OF THE ERROR DISTRIBUTION FUNCTION
	float128 tot = 0;
	
	for (int nu = 0; nu != m_k; ++nu) {
		tot += m_alpha(nu)*m_omega(nu)*exp(-m_alpha(nu)*x); 
	}
	
	return -tot + pow(x, -2);	

} 

float128 laplace::ddEta(float128 x) {
	
	// CALCULATE THE SECOND DERIVATIVE OF THE ERROR DISTRIBUTION FUNCTION
	float128 tot = 0;
	
	for (int nu = 0; nu != m_k; ++nu) {
		tot += pow(m_alpha(nu),2)*m_omega(nu)*exp(-m_alpha(nu)*x);
	}
	
	
	return tot - 2*pow(x, -3);
	
}

}
