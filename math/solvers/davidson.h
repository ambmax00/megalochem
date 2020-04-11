#ifndef MATH_DAVIDSON_H
#define MATH_DAVIDSON_H

#include <vector>
#include <stdexcept>
#include <optional>
#include <Eigen/Eigenvalues>
#include "math/tensor/dbcsr_conversions.hpp"

namespace math {

using vtensor = std::vector<dbcsr::stensor<2>>;

template <class MVFactory>
class davidson {
private:

	dbcsr::stensor<2> m_diag; // diagonal of matrix
	MVFactory& m_fac; // Matrix vector product engine

	int m_nroots; // targeted eigenvalue
	int m_subspace; // current size of subspace
	
	bool m_pseudo; //whether this is a pseudo-eval problem
	bool m_converged; //wether procedure has converged
	
	double m_eigval; // targeted eigenvalue
	
	vtensor m_vecs; // b vectors
	vtensor m_sigmas; // Ab vectors
	vtensor m_ritzvecs; // ritz vectors at end of computation
	
	double m_conv;
	int m_maxiter;
		
public:
	
	struct dav_param {
		required<MVFactory,ref> 		factory;
		required<dbcsr::stensor<2>,ref>	diag;
		optional<bool,val>				pseudo;
		optional<double,val>			conv;
		optional<int,val>				maxiter;
	};
	davidson(dav_param&& p) :
		m_fac(*p.factory),
		m_diag(*p.diag),
		m_eigval(0.0),
		m_pseudo((p.pseudo) ? *p.pseudo : false),
		m_maxiter((p.maxiter) ? *p.maxiter : 20),
		m_conv((p.conv) ? *p.conv : 1e-7),
		m_converged(false)
	{}	

	void compute(vtensor& guess, int nroots, std::optional<double> omega = std::nullopt) {
		
		if (!omega && m_pseudo) 
			throw std::runtime_error("Davidson solver initialized as pseudo-eigenvalue problem, but no omega given.");
		
		m_vecs = guess;
		
		std::cout << "GUESS SIZE: " << m_vecs.size() << std::endl;
		
		m_nroots = nroots;
		
		auto blkoffs = m_vecs[0]->blk_offset();
		auto blksizes = m_vecs[0]->blk_size();
		auto nfull = m_vecs[0]->nfull_tot();
		
		int no = nfull[0];
		int nv = nfull[1];
		
		double conv = 1e-6;
		
		Eigen::MatrixXd Asub;
		Eigen::VectorXd evals;
		Eigen::MatrixXd evecs;
		
		m_sigmas.clear();
		
		for (int ITER = 0; ITER != m_maxiter; ++ITER) {
			
			std::cout << "DAVIDSON ITERATION: " << ITER << std::endl;
			
			// current subspace size
			m_subspace = m_vecs.size();
			int prev_subspace = m_sigmas.size();
			
			// compute sigma vectors
			std::cout << m_vecs.size() << std::endl;
			
			for (int i = prev_subspace; i != m_subspace; ++i) {
				
				std::cout << "GUESS VECTOR: " << i << std::endl;
				dbcsr::print(*m_vecs[i]);
				
				auto Av_i = (m_pseudo) ? m_fac.compute(m_vecs[i],*omega) : m_fac.compute(m_vecs[i]);
				m_sigmas.push_back(Av_i);
				
				std::cout << "SIGMA VECTOR: " << i << std::endl;
				dbcsr::print(*Av_i); 
				
			}
			
			// form subspace matrix
			Asub.conservativeResize(m_subspace,m_subspace);
			
			for (int i = prev_subspace; i != m_subspace; ++i) {
				for (int j = 0; j != i+1; ++j) {
					double val = dbcsr::dot<2>(*m_vecs[i],*m_sigmas[j]);
					Asub(i,j) = val;
					Asub(j,i) = val;
				}
			}
			
			std::cout << "SUBSPACE MATRIX" << std::endl;
			std::cout << Asub << std::endl;
			
			// diagonalize it
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
			
			es.compute(Asub);
			
			evals = es.eigenvalues();
			evecs = es.eigenvectors();
			
			for (int i = 0; i != m_subspace; ++i) {
				// get min and max coeff
				double minc = evecs.col(i).minCoeff();
				double maxc = evecs.col(i).maxCoeff();
				double maxabs = (fabs(maxc) > fabs(minc)) ? maxc : minc;
				
				if (maxabs < 0) evecs.col(i) *= -1;
				
			}
			
			std::cout << "EIGENVALUES: " << std::endl;
			for (int i = 0; i != evals.size(); ++i) {
				std::cout << evals[i] << " ";
			} std::cout << std::endl;
			
			std::cout << "EIGENVECTORS: " << std::endl;
			std::cout << evecs << std::endl;
			
			// compute residual
			// r_k = sum_i U_ik sigma_i - sum_i U_ik lambda_k b_i 
			
			std::cout << "SETUP" << std::endl;
			
			// ======= r_k ========
			
			dbcsr::tensor<2> r_k({.tensor_in = *m_vecs[0], .name = "r_k"});
			dbcsr::tensor<2> temp({.tensor_in = *m_vecs[0], .name = "temp"});
			
			std::cout << "LOOP" << std::endl;
			
			for (int i = 0; i != m_subspace; ++i) {
				
				std::cout << "Copy" << std::endl;
				dbcsr::copy<2>({.t_in = *m_sigmas[i], .t_out = temp});
				
				temp.scale(evecs(i,m_nroots-1));
				
				dbcsr::copy<2>({.t_in = temp, .t_out = r_k, .sum = true, .move_data = true});
				
				dbcsr::copy<2>({.t_in = *m_vecs[i], .t_out = temp});
				
				temp.scale(- evals(m_nroots - 1) * evecs(i,m_nroots-1));
				
				dbcsr::copy<2>({.t_in = temp, .t_out = r_k, .sum = true, .move_data = true});
				
			}
			
			temp.destroy();
			
			std::cout << "RESIDUAL" << std::endl;
			dbcsr::print(r_k);
			
			double rms = dbcsr::RMS(r_k);
			std::cout << "RMS: " << rms << std::endl;
			
			// Convergence criteria
			// Normal davidson: RMS of residual is below certain threshold
			// pseudo davidson: RMS of residual is higher than |omega - omega'| 
			
			if (m_pseudo) std::cout << "EVALS: " << *omega << " " << evals(m_nroots-1) << std::endl;
			
			m_converged = (m_pseudo) ? (rms < fabs(*omega - evals(m_nroots - 1))) : (rms < conv);
			
			if (m_converged) break;
			
			// correction 
			// Only Diagonal-Preconditioned-Residue for now (DPR)
			// D_ia = (lamda_k - A_iaia)^-1
			
			dbcsr::tensor<2> D({.tensor_in = r_k, .name = "D"});
			
			D.reserve_all();
			
			dbcsr::iterator<2> d_iter(D);
			
			while (d_iter.blocks_left()) {
				
				d_iter.next();
				
				auto blksize = d_iter.sizes();
				auto idx = d_iter.idx();
				
				bool found = false;
				
				dbcsr::block<2> d_blk(blksize);
				dbcsr::block<2> diag_blk = 
					m_diag->get_block({.idx = idx, .blk_size = blksize, .found = found});
			
				if (found) {
				
					for (int i = 0; i != d_blk.ntot(); ++i) {
						d_blk.data()[i] = pow(evals(m_nroots - 1) - diag_blk.data()[i],-1);
					}
					
				} else {
					
					for (int i = 0; i != d_blk.ntot(); ++i) {
						d_blk.data()[i] = pow(evals(m_nroots - 1),-1);
					}
					
				}
					
				D.put_block({.idx = idx, .blk = d_blk});
				
			} //end while
			
			std::cout << "PRECONDITIONER" << std::endl;
			
			dbcsr::print(D);
			
			// form new vector
			dbcsr::tensor<2> d_k({.tensor_in = r_k, .name = "d_k"});
			
			// d(k)_ia = D(k)_ia * q(k)_ia
			
			dbcsr::print(r_k);
			
			dbcsr::ewmult<2>(r_k, D, d_k);
			
			r_k.destroy();
			D.destroy();
			
			dbcsr::print(d_k);
			
			// GRAM SCHMIDT
			// b_new = d_k - sum_j proj_bi(d_k)
			// where proj_b(v) = dot(b,v)/dot(b,b)
			
			dbcsr::tensor<2> bnew({.tensor_in = d_k, .name = "b_" + std::to_string(m_subspace)});
			dbcsr::tensor<2> temp2({.tensor_in = d_k, .name = "temp"});
			
			dbcsr::copy<2>({.t_in = d_k, .t_out = bnew});
			
			for (int i = 0; i != m_vecs.size(); ++i) {
				
				dbcsr::copy<2>({.t_in = *m_vecs[i], .t_out = temp2});
				double proj = dbcsr::dot(*m_vecs[i], d_k)/dbcsr::dot(*m_vecs[i],*m_vecs[i]);
				
				std::cout << proj << std::endl;
				
				temp2.scale(-proj);
				
				dbcsr::copy<2>({.t_in = temp2, .t_out = bnew, .sum = true});
				
			}
				
			temp2.destroy();
			
			// normalize
			double bnorm = sqrt(dbcsr::dot(bnew,bnew));
			
			bnew.scale(1/bnorm);
			
			dbcsr::print(bnew);
			
			auto bnewptr = bnew.get_stensor();
			
			m_vecs.push_back(bnewptr);
				
		}
		
		m_eigval = evals(m_nroots-1);
		
		dbcsr::tensor<2> temp3({.tensor_in = *m_vecs[0], .name = "temp3"});
		
		m_ritzvecs.clear();
		
		// Make sure that abs max element of eigenvectors is positive.
		
		std::cout << "EVECS" << std::endl;
		std::cout << evecs << std::endl;
		
		/*for (int i = 0; i != m_subspace; ++i) {
				// get min and max coeff
				double minc = evecs.col(i).minCoeff();
				double maxc = evecs.col(i).maxCoeff();
				double maxabs = (fabs(maxc) > fabs(minc)) ? maxc : minc;
				
				if (maxabs < 0) evecs.col(i) *= -1;
				
		}*/
		
		std::cout << "EVECS" << std::endl;
		std::cout << evecs << std::endl;
		
		// x_k = sum_i ^M U_ik * b_i
		for (int k = 0; k != m_nroots; ++k) {
			dbcsr::tensor<2> x_k({.tensor_in = *m_vecs[0], .name = "new_guess_"+std::to_string(k)});
			for (int i = 0; i != m_vecs.size(); ++i) {
				
				dbcsr::print(*m_vecs[i]);
				std::cout << evecs(i,k) << std::endl;
				
				dbcsr::copy<2>({.t_in = *m_vecs[i], .t_out = temp3});
				temp3.scale(evecs(i,k));
				
				dbcsr::print(temp3);
				
				dbcsr::copy<2>({.t_in = temp3, .t_out = x_k, .sum = true, .move_data = true});
				
			}
				
			dbcsr::print(x_k);
			m_ritzvecs.push_back(x_k.get_stensor());
			
		}
		
		std::cout << "FINISHED COMPUTATION" << std::endl;
		
	}
	
	vtensor ritz_vectors() {
		
		return m_ritzvecs;
		
	}
	
	double eigval() {
		return m_eigval;
	}

}; // end class 

template <class MVFactory>
class modified_davidson {
private:

	dbcsr::stensor<2> m_diag; // diagonal of matrix
	
	davidson<MVFactory> m_solver;

	bool m_converged; //wether procedure has converged
	
	double m_eigval; // targeted eigenvalue
	
	vtensor m_ritzvecs; // ritz vectors at end of computation
	
	int m_maxiter;
	double m_conv;
		
public:
	
	struct moddav_param {
		required<MVFactory,ref> 		factory;
		required<dbcsr::stensor<2>,ref>	diag;
		optional<double,val>			conv;
		optional<int,val>				maxiter;
		optional<int,val>				micro_maxiter;
	};
	modified_davidson(moddav_param&& p) :
		m_eigval(0.0),
		m_converged(false),
		m_maxiter((p.maxiter) ? *p.maxiter : 10),
		m_conv((p.conv) ? *p.conv : 1e-7),
		m_solver({.factory = p.factory, .diag = p.diag, .pseudo = true, .maxiter = p.micro_maxiter})
	{}	

	void compute(vtensor& guess, int nroots, double omega) {
		
		vtensor current_guess = guess;
		double current_omega = omega;
		
		for (int i = 0; i != 5; ++i) {
			
			std::cout << " == MACROITERATION: == " << i << std::endl;
			
			double old_omega = current_omega;
			
			m_solver.compute(current_guess,nroots,current_omega);
			
			current_guess = m_solver.ritz_vectors();
			current_omega = m_solver.eigval();
			
			std::cout << "MACRO ITERATION ERROR: " << fabs(current_omega - old_omega) << std::endl;
			std::cout << "EIGENVALUE: " << current_omega << std::endl;
			
		}
		
		m_eigval = current_omega;
		m_ritzvecs = current_guess;
		
	}
	
	vtensor ritz_vectors() {
		
		return m_ritzvecs;
		
	}
	
	double eigval() {
		return m_eigval;
	}

}; // end class

} // end namespace

#endif
