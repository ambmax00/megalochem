#ifndef MATH_DAVIDSON_H
#define MATH_DAVIDSON_H

#include <vector>
#include <stdexcept>
#inlcude "math/tensor/dbcsr_conversions.hpp"

namespace math {

using std::vector<dbcsr::stensor<2>> = vtensor;

class Davidson {
private:

		int m_k; // number of eigvals to compute
		vtensor m_vecs; // b(0) and b(k) vectors
		int m_maxiter = 10;
		
public:
	
		template <class MVFactory>
		void compute(MVFactory& Mv, vtensor& guess) {
			
			for (int ITER = 0; ITER != m_max_iter; ++ITER) {
				
				// current subspace size
				int N = m_vecs.size();
				
				// compute sigma vectors
				vtensor sigma_vectors;
				
				for (int i = 0; i != N; ++i) {
					auto Av_i = MVFactory(m_vecs[0]);
					sigma_vectors.push_back(Av_i);
				}
				
				// form subspace matrix
				Eigen::MatrixXd A(N,N);
				
				for (int i = 0; i != N; ++i) {
					for (int j = i; j != N; ++j) {
						double val = dbcsr::dot<2>(*m_vecs[i],*sigma_vectors[j]);
						A(i,j) = val;
						A(j,i) = val;
					}
				}
				
			}
			
		}

}; // end class 

} // end namespace
