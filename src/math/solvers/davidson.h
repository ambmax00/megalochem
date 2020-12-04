#ifndef MATH_DAVIDSON_H
#define MATH_DAVIDSON_H

#include <vector>
#include <stdexcept>
#include <optional>
#include <Eigen/Eigenvalues>
#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_conversions.hpp>
#include "utils/mpi_log.h"

namespace math {

using smat = dbcsr::shared_matrix<double>;

template <class MVFactory>
class davidson {
private:
	
	util::mpi_log LOG;
	smat m_diag; // diagonal of matrix
	std::shared_ptr<MVFactory> m_fac; // Matrix vector product engine

	int m_nroots; // requested number of eigenvalues
	int m_subspace; // current size of subspace
	
	bool m_pseudo; //whether this is a pseudo-eval problem
	bool m_converged; //wether procedure has converged
	bool m_block; //wether we optimize all roots at once
	
	std::vector<double> m_eigvals; // eignvalues
	std::vector<double> m_errs; // errors of the roots
	
	std::vector<smat> m_vecs; // b vectors
	std::vector<smat> m_sigmas; // Ab vectors
	std::vector<smat> m_ritzvecs; // ritz vectors at end of computation
	
	double m_conv;
	int m_maxiter;
		
public:

	davidson& set_factory(std::shared_ptr<MVFactory>& fac) {
		m_fac = fac;
		return *this;
	}
	
	davidson& set_diag(smat& in) {
		m_diag = in;
		return *this;
	}
	
	davidson& pseudo(bool is_pseudo) {
		m_pseudo = is_pseudo;
		return *this;
	}
	
	davidson& conv(double c) {
		m_conv = c;
		return *this;
	}
	
	davidson& maxiter(int maxi) {
		m_maxiter = maxi;
		return *this;
	}
	
	davidson& block(bool is_block) {
		m_block = is_block;
		return *this;
	}
	
	davidson(MPI_Comm comm, int nprint) : 
		LOG(comm, nprint),
		m_converged(false),
		m_pseudo(false),
		m_conv(1e-5),
		m_maxiter(100),
		m_block(false) 
	{}

	void compute(std::vector<smat>& guess, int nroots, std::optional<double> omega = std::nullopt) {
		
		LOG.os<>("Launching davidson diagonalization.\n");
		LOG.os<>("Convergence: ", m_conv, '\n');
		
		if (!omega && m_pseudo) 
			throw std::runtime_error("Davidson solver initialized as pseudo-eigenvalue problem, but no omega given.");
		
		m_vecs = guess;
		
		LOG.os<1>("GUESS SIZE: ", m_vecs.size(), '\n');
		
		m_nroots = nroots;
		
		if (m_nroots > m_vecs.size()) {
			std::string msg = std::to_string(m_nroots) + " roots requested, but only "
				+ std::to_string(m_vecs.size()) + " guesses given.\n";
			throw std::runtime_error(msg);
		}
		
		m_errs.clear();
		m_errs.resize(m_nroots);
		
		Eigen::MatrixXd Asub;
		Eigen::VectorXd evals;
		Eigen::MatrixXd evecs;
		Eigen::VectorXd prev_evals(0);
		Eigen::MatrixXd prev_evecs(0,0);
		
		m_sigmas.clear();
		
		for (int ITER = 0; ITER != m_maxiter; ++ITER) {
			
			LOG.os<1>("DAVIDSON ITERATION: ", ITER, '\n');
			
			// current subspace size
			m_subspace = m_vecs.size();
			int prev_subspace = m_sigmas.size();
			
			LOG.os<1>("SUBSPACE: ", m_subspace, " PREV: ", prev_subspace, '\n');
			
			// compute sigma vectors
			std::cout << m_vecs.size() << std::endl;
			
			LOG.os<1>("Computing MV products.\n");
			for (int i = prev_subspace; i != m_subspace; ++i) {
				
				std::cout << "GUESS VECTOR: " << i << std::endl;
				dbcsr::print(*m_vecs[i]);
				
				auto Av_i = (m_pseudo) ? m_fac->compute(m_vecs[i],*omega) : m_fac->compute(m_vecs[i]);
				m_sigmas.push_back(Av_i);
				
				std::cout << "SIGMA VECTOR: " << i << std::endl;
				dbcsr::print(*Av_i); 
				
			}
			
			// form subspace matrix
			Asub.conservativeResize(m_subspace,m_subspace);
			
			for (int i = prev_subspace; i != m_subspace; ++i) {
				for (int j = 0; j != i+1; ++j) {
					
					double val = m_vecs[i]->dot(*m_sigmas[j]);
					Asub(i,j) = val;
					Asub(j,i) = val;
				}
			}
		
			LOG.os<2>("SUBSPACE MATRIX:\n");
			LOG.os<2>(Asub, '\n');
			
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
			
			/* if zero eigenvalues, collapse subspace
			bool has_zero = false;
			for (int i = 0; i != */
			
			if (LOG.global_plev() >= 1) {
				LOG.os<1>("EIGENVALUES: \n");
				for (int i = 0; i != evals.size(); ++i) {
					LOG.os<1>(evals[i], " ");
				} LOG.os<1>('\n');
			}
			
			LOG.os<2>("EIGENVECTORS: \n", evecs, '\n');
			
			prev_evals = evals;
			prev_evecs = evecs;	 
			
			// compute residual
			// r_k = sum_i U_ik sigma_i - sum_i U_ik lambda_k b_i 
			
			LOG.os<1>("Computing residual.\n");
			
			// ======= Compute all residuals r_k ========
			
			std::vector<smat> residuals(m_nroots);
			smat temp = dbcsr::create_template<double>(m_vecs[0]).name("temp").get();
			double max_err = 0;
			
			int start_root = 0; //(m_block) ? 0 : m_nroots - 1;
			
			for (int iroot = start_root; iroot != m_nroots; ++iroot) {
			//int iroot = m_nroots - 1;
			
				smat r_k = dbcsr::create_template<double>(m_vecs[0]).name("temp").get();
				
				for (int i = 0; i != m_subspace; ++i) {
					
					temp->copy_in(*m_sigmas[i]);
					temp->scale(evecs(i,iroot));
					
					r_k->add(1.0, 1.0, *temp);
					
					temp->clear();
					temp->copy_in(*m_vecs[i]);
					
					temp->scale(- evals(iroot) * evecs(i,iroot));
					
					r_k->add(1.0,1.0, *temp);
					
					temp->clear();
				}
				
				// because we are probably dealing with an approximation to
				// the real diagonal of the matrix, we are checking both the 
				// residual and the eigenvectors of the subspace matrix
				double alpha_ki = (ITER == 0) ? std::numeric_limits<double>::max() 
						: fabs(evecs(m_subspace-1,iroot));
				
				m_errs[iroot] = //std::min(
					r_k->norm(dbcsr_norm_frobenius);//,
					//alpha_ki
				//);
				
				max_err = std::max(max_err, m_errs[iroot]);
				
				residuals[iroot] = r_k;
				
			}
			
			temp->release();
			
			//if (LOG.global_plev() >= 2) {
			//	dbcsr::print(*r_k);
			//}
			
			LOG.os<>("Root errors: \n");
			for (int iroot = start_root; iroot != m_nroots; ++iroot) {
				LOG.os<>("Root ", iroot, ": ", m_errs[iroot], '\n');
			}
			LOG.os<>("Max Error: ", max_err, '\n');
			
			// Convergence criteria
			// Normal davidson: largest norm of residuals is below certain threshold
			// pseudo davidson: largest norm of residual is higher than |omega - omega'| 
			
			if (m_pseudo) LOG.os<1>("EVALS: ", *omega, " ", evals(m_nroots-1), '\n');
			
			m_converged = (m_pseudo) ? (max_err < fabs(*omega - evals(m_nroots - 1))) : (max_err < m_conv);
			
			if (m_converged) break;
			
			// correction 
			// Only Diagonal-Preconditioned-Residue for now (DPR)
			// D_ia = (lamda_k - A_iaia)^-1
			
			LOG.os<1>("Computing correction vectors.\n");
			int nvecs_prev = m_vecs.size();			
			
			for (int iroot = start_root; iroot != m_nroots; ++iroot) {
			//int iroot = m_nroots - 1;
				
				LOG.os<1>("Computing for root ", iroot, '\n');
				LOG.os<1>("Computing preconditioner.\n");
				
				smat D = dbcsr::create_template(residuals[iroot]).name("D").get();
				
				D->reserve_all();
				
				D->set(evals(iroot));
				
				D->add(1.0,-1.0,*m_diag);
				
				D->apply(dbcsr::func::inverse);
				
				if (LOG.global_plev() >= 2) {			
					dbcsr::print(*D);
				}
				
				// form new vector
				smat d_k = dbcsr::create_template(residuals[iroot])
					.name("d_k").get();
				
				// d(k)_ia = D(k)_ia * q(k)_ia
								
				d_k->hadamard_product(*residuals[iroot], *D);
				
				double dnorm = sqrt(d_k->dot(*d_k));
				
				d_k->scale(1.0/dnorm);
				
				residuals[iroot]->release();
				D->release();
								
				// GRAM SCHMIDT
				// b_new = d_k - sum_j proj_bi(d_k)
				// where proj_b(v) = dot(b,v)/dot(b,b)
				
				smat bnew = dbcsr::create_template(d_k).name("b_" + std::to_string(m_subspace)).get();
				smat temp2 = dbcsr::create_template(d_k).name("temp2").get();
				
				//dbcsr::copy(d_k, bnew).perform();
				bnew->copy_in(*d_k);
				
				LOG.os<1>("Computing new guess vector.\n");
				for (int i = 0; i != m_vecs.size(); ++i) {
									
					//dbcsr::copy(*m_vecs[i], temp2).perform();
					temp2->copy_in(*m_vecs[i]);
					
					double proj = (d_k->dot(*m_vecs[i])) / (m_vecs[i]->dot(*m_vecs[i]));
					
					temp2->scale(-proj);
					
					bnew->add(1.0,1.0,*temp2);
					
				}
					
				temp2->release();
				
				// normalize
				double bnorm = sqrt(bnew->dot(*bnew));
				
				std::cout << "D/B norm: " << dnorm << " " << bnorm << " "
					<< bnorm/dnorm << std::endl;
				
				bnew->scale(1.0/bnorm);
				
				if (LOG.global_plev() >= 2) {
					dbcsr::print(*bnew);
				}
							
				if (bnorm/dnorm > 1e-3) m_vecs.push_back(bnew);
				
			}
			
			if (m_vecs.size() == nvecs_prev) {
				
				LOG.os<1>("No new vectors were added, collapsing subspace...\n");
				
				std::vector<smat> new_vecs;
				smat tempx = dbcsr::create_template<double>(m_vecs[0])
					.name("tempx").get();
				
				for (int k = 0; k != m_nroots; ++k) {
				
					smat x_k = dbcsr::create_template(m_vecs[0])
						.name("new_guess_"+std::to_string(k))
						.get(); 
					
					for (int i = 0; i != m_vecs.size(); ++i) {
												
						tempx->copy_in(*m_vecs[i]);
						tempx->scale(evecs(i,k));
											
						x_k->add(1.0,1.0,*tempx);
						
					}
					
					new_vecs.push_back(x_k);
					
				}
				
				m_vecs = new_vecs;
				m_sigmas.clear();
			
			}
							
		}
		
		m_eigvals.resize(m_nroots);
		std::copy(evals.data(), evals.data() + m_nroots, m_eigvals.data());
				
		smat temp3 = dbcsr::create_template<double>(m_vecs[0]).name("temp3").get();
		
		m_ritzvecs.clear();
		
		LOG.os<1>("Forming Ritz vectors.\n");
		// x_k = sum_i ^M U_ik * b_i
		for (int k = 0; k != m_nroots; ++k) {
			
			smat x_k = dbcsr::create_template(m_vecs[0])
				.name("new_guess_"+std::to_string(k))
				.get(); 
			
			for (int i = 0; i != m_vecs.size(); ++i) {
				
				temp3->copy_in(*m_vecs[i]);
				temp3->scale(evecs(i,k));
								
				x_k->add(1.0,1.0,*temp3);
				
			}
				
			m_ritzvecs.push_back(x_k);
			
		}
		
		//std::cout << "FINISHED COMPUTATION" << std::endl;
		LOG.os<>("Finished computation.\n");
		
	}
	
	std::vector<smat> ritz_vectors() {
		
		return m_ritzvecs;
		
	}
	
	std::vector<double> eigvals() {
		return m_eigvals;
	}

}; // end class 

template <class MVFactory>
class modified_davidson {
private:

	util::mpi_log LOG;
	
	int m_macro_maxiter;	
	davidson<MVFactory> m_dav;
	
		
public:

	modified_davidson& set_factory(std::shared_ptr<MVFactory>& fac) {
		m_dav.set_factory(fac);
		return *this;
	}
	
	modified_davidson& set_diag(smat& in) {
		m_dav.set_diag(in);
		return *this;
	}
	
	modified_davidson& conv(double c) {
		m_dav.conv(c);
		return *this;
	}
	
	modified_davidson& macro_maxiter(int maxi) {
		m_macro_maxiter = maxi;
		return *this;
	}
	
	modified_davidson& micro_maxiter(int maxi) {
		m_dav.maxiter = maxi;
		return *this;
	}
	
	modified_davidson(MPI_Comm comm, int nprint) : 
		LOG(comm, nprint),
		m_dav(comm, nprint)
	{}
	
	void compute(std::vector<smat>& guess, int nroots, double omega) {

		std::vector<smat> current_guess = guess;
		double current_omega = omega;
		
		for (int i = 0; i != m_macro_maxiter; ++i) {
			
			LOG.os<>(" == MACROITERATION: ==\n");
			
			double old_omega = current_omega;
			
			m_dav.compute(current_guess,nroots,current_omega);
			
			current_guess = m_dav.ritz_vectors();
			current_omega = m_dav.eigvals()[nroots-1];
			
			LOG.os<>("MACRO ITERATION ERROR: ", fabs(current_omega - old_omega), '\n');
			LOG.os<>("EIGENVALUE: ", current_omega, '\n');
			
		}
		
		LOG.os<>("Modified davidson finished\n");
		
	}
	
	std::vector<smat> ritz_vectors() {
		
		return m_dav.ritz_vectors();
		
	}
	
	std::vector<double> eigval() {
		return m_dav.eigvals();
	}

}; // end class


} // end namespace

#endif
