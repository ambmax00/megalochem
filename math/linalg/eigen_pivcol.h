#ifndef MATH_EIGEN_PIVCOL_H
#define MATH_EIGEN_PIVCOL_H

#include <Eigen/Core>
#include <limits>

namespace math {
	
class eigen_pivcol {
private:

	Eigen::MatrixXd m_imat;
	Eigen::MatrixXd m_lmat;
	Eigen::VectorXi m_pmat;
	int N;

public:

	eigen_pivcol(Eigen::MatrixXd& in) {
			
		m_imat = in;
		N = m_imat.cols();
		m_lmat = Eigen::MatrixXd::Zero(N,N);
		m_pmat = Eigen::VectorXi::Zero(N);
	
	}
		
	void compute () {
		
		auto& M = m_imat;
		auto& L = m_lmat;
		auto& P = m_pmat;
		
		auto& Mback = M;
		
		int rank = 0;
		
		double threshold = std::numeric_limits<double>::epsilon();
		
		for (size_t i = 0; i != N; ++i) {
			 
			std::cout << "STEP: " << i << std::endl;
			
			std::ptrdiff_t j, jtilde;
			
			std::cout << "Current M" << std::endl;
			std::cout << M << std::endl;
			
			// take submatrix
			auto U = M.block(i,i,N-i,N-i);
		
			// choose maximum diagonal element
			double maxAbsUjj = U.diagonal().cwiseAbs().maxCoeff(&j);
			double maxUjj = U(j,j);
			
			int mpermi = P(i);
			P(i) = P(j + i);
			P(j + i) = mpermi;
			
			std::cout << "Permutations: " << std::endl;
			std::cout << P << std::endl;
				
			std::cout << "Swapping " << i << " and " << j+i << std::endl;
			// swap columns
			Eigen::VectorXd col0 = U.col(0);
			Eigen::VectorXd colj = U.col(j);
			U.col(0) = colj;
			U.col(j) = col0;
			
			// swap rows
			//Eigen::VectorXd row0 = U.row(0);
			//Eigen::VectorXd rowj = U.row(j);
			//U.row(0) = rowj;
			//U.row(j) = row0;
				
			std::cout << "After swapping: " << std::endl;
			std::cout << U << std::endl;
			std::cout << "M after swapping." << std::endl;
			std::cout << M << std::endl;
			
			if (maxAbsUjj < threshold) {
				
				std::cout << "Exiting at level " << ++rank << std::endl;
				break;
				
			} else {
				
				++rank;
				
				if (U.size() == 1) {
					L(i,i) = sqrt(fabs(U(0,0)));
					std::cout << "breaking." << std::endl;
					break;
				}
			
				// form Utilde - 1/U11 * u * uT
				Eigen::VectorXd u = U.block(1,0,N-i-1,1);
				
				std::cout << "u" << std::endl;
				std::cout << u << std::endl;
				
				// scale the vector
				//for (int k = 0; k != u.size(); ++k) {
				//	double uk = u(k);
				//	int sgn = (uk < 0) ? -1 : ((uk > 0) ? 1 : 0);
				//	u(k) = sgn * std::min(fabs(uk), sqrt(U(0,0) * U(k,k)));
				//}
				
				//std::cout << "u scaled" << std::endl;
				//std::cout << u << std::endl;
				
				U.block(1,1,N-i-1,N-i-1) -= 1/maxUjj * u * u.transpose();
				
				double Uii = U(0,0);
				
				std::cout << "MINUS" << std::endl;
				auto min = 1/Uii * u * u.transpose();
				
				std::cout << min << std::endl;
				
				std::cout << "M" << std::endl;
				std::cout << M << std::endl;
				
				auto Utilde = U.block(1,1,N-i-1,N-i-1);
				
				/*
				// find max element of Utilde
				double maxAbsUjjtilde = U.diagonal().cwiseAbs().maxCoeff(&jtilde);
				double maxUjjtilde = U(jtilde,jtilde);
				
				std::cout << "max element at " << jtilde << std::endl;
				
				std::cout << "Utilde" << std::endl;
				std::cout << Utilde << std::endl;
				
				std::cout << "Current M:" << std::endl;
				std::cout << M << std::endl;
				
				std::cout << "Permute u" << std::endl;
				
				double u0 = u(0);
				u(0) = u(jtilde);
				u(jtilde) = u0;
				
				// update m_L
				
				m_L(i,i) = sqrt(fabs(U(0,0)));
				m_L.col(i).tail(N-i-1) = 1/maxUjjtilde * u;
				
				std:cout << "New L" << std::endl;
				std::cout << m_L << std::endl;
				*/
				
				L(i,i) = fabs(sqrt(Uii));
				
				std::cout << "L" << std::endl;
				std::cout << L << std::endl;
				
				double maxAbsUjjTilde = Utilde.diagonal().cwiseAbs().maxCoeff(&jtilde);
				double maxUjjtilde = Utilde(jtilde,jtilde);
				
				// swap rows of L
				//double u0 = u(0);
				//u(0) = u(jtilde);
				//u(jtilde) = u0;
				
				L.col(i).tail(N-i-1) = u;
				
				std::cout << "After swapping: " << std::endl;
				std::cout << L << std::endl;
			
			}
		
		
		}
		
		
		std::cout << "M_rank = " << rank << std::endl;
		//L = L.block(0,0,N,m_rank);
		
		std::cout << "Final L (not permd) " << std::endl;
		std::cout << L << std::endl;
		
		auto slice_p = P; // .tail(rank);
		std::cout << "Slice p" << std::endl;
		std::cout << slice_p << std::endl;
		
		Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic,int> PermMat(slice_p);
		
		auto L1 = PermMat.inverse() * L;
		auto L2 = PermMat.transpose() * L;
		auto L3 = PermMat * L;
		
		std::cout << "Final L (permd) " << std::endl;
		std::cout << L << std::endl;
		
		auto res0 = M - L * L.transpose();
		auto res1 = Mback - L1 * L1.transpose();
		auto res2 = Mback - L2 * L2.transpose();
		auto res3 = Mback - L3 * L3.transpose();
		
		std::cout << "RES0" << std::endl;
		std::cout << res0 << std::endl;
		
		std::cout << "RES1" << std::endl;
		std::cout << res1 << std::endl;
		
		std::cout << "RES2" << std::endl;
		std::cout << res2 << std::endl;
		
		std::cout << "RES3" << std::endl;
		std::cout << res3 << std::endl;
		
	}
	
};

} // end namespace

#endif
