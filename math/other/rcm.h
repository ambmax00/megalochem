#ifndef MATH_RCM_H
#define MATH_RCM_H

#include <vector>
#include <iostream>
#include <stdexcept>

namespace math {
	
/* RCM: takes in a set of coordinates, and a function, then reduces
 * the bandwidth of the connectivity matrix, and spits out a vector
 * with reordered indices (also offers a reordering tool (?))
 */

template <class Point>
class rcm {
	
using func = std::function<double(Point, Point)>;
	
private:

	const std::vector<Point> m_points;
	const func m_distfunc;
	const double m_cutoff;
	const size_t m_dim;
	
	std::vector<int> m_index;
	std::unique_ptr<double[]> m_distmat;
	std::unique_ptr<int[]> m_conmat;
	
	size_t idx(size_t i, size_t j) {
		return i*m_dim + j;
	}
	
	template <typename T>
	void print(std::unique_ptr<T>& mat) {
		
		
		for (int i = 0; i != m_dim; ++i) {
			for (int j = 0; j != m_dim; ++j) {
				std::cout << mat[idx(i,j)] << " ";
			}
			std::cout << std::endl;
		}
			
	}
		
	void compute_distmat() {
		
		//std::cout << "Sym size " << m_dim*(m_dim+1)/2 << std::endl;
		
		for (int p1 = 0; p1 != m_dim; ++p1) {
		 for (int p2 = p1; p2 != m_dim; ++p2) {
			 
			 m_distmat[idx(p1,p2)] = m_distfunc(m_points[p1], m_points[p2]);
			 m_distmat[idx(p2,p1)] = m_distmat[idx(p1,p2)];
			 
		}}	
	
		print(m_distmat);
	
	}
	
	void compute_conmat() {
		
		for (int i = 0; i != m_dim*m_dim; ++i){
			m_conmat[i] = fabs(m_distmat[i]) >= m_cutoff ? 0 : 1;
		}
		
	}
	
	int degree(int node) {
		
		int d = 0;
	
		for (int i = 0; i != m_dim; ++i) {
			d += m_conmat[idx(node,i)];
		}
	
		return d-1;
		
	}
		
	int start_node() {
	
		// First, get the two points, which are at maximum distance from eachother
		double max_dist = 0;
		int node1 = 0;
		int node2 = 0;
	
		for (int i = 0; i < m_dim; ++i) {
			for (int j = i+1; j < m_dim; ++j) {
			
				if (m_distmat[idx(i,j)] > max_dist) {
					max_dist = m_distmat[idx(i,j)];
					node1 = i;
					node2 = j;
				}
			}
		}
	
		//std::cout << "MAX dist at: " << node1 << " " << node2 << " " << max_dist << std::endl;
	
		// degree of the nodes
		int deg1 = degree(node1);
		int deg2 = degree(node2);
	
		//std::cout << "Degrees: " << deg1 << " " << deg2 << std::endl;
	
		// Return the node with the least amount of bonds (If applicable)
		if (deg1 > deg2) { return node2;} 
		else { return node1; }
	
	}
	
	
public:

	rcm(std::vector<Point> t_points, double t_cutoff, 
		func t_distfunc) : m_points(t_points), m_distfunc(t_distfunc), 
		m_cutoff(t_cutoff), m_index(t_points.size()), m_dim(t_points.size()),
		m_distmat(new double[m_dim*m_dim]), 
		m_conmat(new int[m_dim*m_dim]) {};

	~rcm() {};
	
	void compute() {
		
		compute_distmat();
		compute_conmat();
		
		print(m_distmat);
		print(m_conmat);
		
		int P = start_node();
		//std::cout << "Best node to start: " << P << std::endl;
	
		// PREPARE EMTPY QUEUE AND RESULT ARRAY
		std::vector<int> Q;
		std::vector<int> R;
	
		// SELECT STARTING NODE
		R.push_back(P);
	
		// ADD TO THE QUEUE ALL ADJACENT NODES TO P
		for (int i = 0; i != m_dim; ++i) {
			if ((m_conmat[idx(P,i)] == 1) && (i != P)) {
				Q.push_back(i);
			}
		}
	
		//std::cout << "PRE SORT" << std::endl;
		for (int i = 0; i != Q.size(); ++i) {
			//std::cout << Q[i] << " ";
		}
		//std::cout << std::endl;
	
		// SORT IT ACCORDING TO DEGREE
		auto sortdeg = [&](const int q1, const int q2) 
		-> bool {
				return degree(q1) < degree(q2);
		};	
	
		std::sort(Q.begin(), Q.end(), sortdeg);
	
		//std::cout << "AFTER SORT" << std::endl;
		for (int i = 0; i != Q.size(); ++i) {
			//std::cout << Q[i] << " ";
		}
		//std::cout << std::endl;
	
		int Qi = 0;
	
		while ( Q.size() ) {
	
			std::vector<int>::iterator it = std::find(R.begin(), R.end(), Q[Qi]);
		
			if (it == R.end()) {
			
				//std::cout << Q[Qi] << " not yet in R " << std::endl;
			
				R.push_back(Q[Qi]);
			
				int newP = Q[Qi];
			
				Q.erase(Q.begin() + Qi);
			
				int offset = Q.size();
			
			// ADD ALL ADJACENT POINTS OF newP to Q, that are not in R
				for (int i = 0; i != m_dim; ++i) {
					if ((m_conmat[idx(newP,i)] == 1) && (i != newP) ) {
					
						std::vector<int>::iterator it2;
						it2 = find(R.begin(), R.end(), i);
					
						if (it2 == R.end()) Q.push_back(i);
					}
				}
			
			// SORT
				std::sort(Q.begin()+offset, Q.end(), sortdeg);
			
				//std::cout << "NEW Q: " << std::endl;
				for (int i = 0; i != Q.size(); ++i) {
					std::cout << Q[i] << " ";
				}
				//std::cout << std::endl;
			
			
			} else if (Qi < Q.size()) {
				
				//std::cout << Q[Qi] << "already in R " << std::endl;
				Q.erase(Q.begin() + Qi);
				
				if (Q.size() == 0) {
					//std::cout << "Q is zero.. Looking for unexplored nodes..." << std::endl;
					for (int i = 0; i != m_dim; ++i) {
						std::vector<int>::iterator it;
						it = std::find(R.begin(), R.end(), i);
						if (it == R.end()) {
							Q.push_back(i);
							//std::cout << "Found one..." << std::endl;
							break;
						}
					}
				}
				
			} 
			
	
		} // END WHILE
	
		//std::cout << "Here is R " << std::endl;
		for (int i = 0; i != R.size(); ++i) {
			std::cout << R[i] << " ";
		}
		//std::cout << std::endl;
		
		if (R.size() != m_dim) throw std::runtime_error("Error reordering atoms.");
	
	
		//std::cout << "PREVIOUS CON MAT" << std::endl;
		//print(m_conmat);
	
	// REORDER CMAT
		std::unique_ptr<int[]> CMAT2(new int[m_dim*m_dim]);
	
		for (int j = 0; j != m_dim; ++j) {
		 for (int i = 0; i != m_dim; ++i) {
			CMAT2[idx(i,j)] = m_conmat[idx(R[i], R[j])];	
		 }
		}
		
		m_conmat.reset();
		
		m_conmat = std::move(CMAT2);
		
		std::cout << "REORDERED CON MAT" << std::endl;
		print(m_conmat);
		
		m_index = R;
	
		std::cout << "Reordered atoms: " << std::endl;
		for (int i = 0; i != m_dim; ++i) {
			std::cout << m_index[i] << " ";
		}
	
		std::cout << std::endl;
	
	}
	
	template <typename T>
	void reorder(T& p_in) {
		
		T p_out = p_in;
		for (int i = 0; i != m_points.size(); ++i) {
			p_out[i] = p_in[m_index[i]];
		}
		p_in = p_out;
	}
	
}; // end class

} // end namespace



#endif
