#ifndef MATH_INVERSE_H
#define MATH_INVERSE_H

#include <dbcsr_matrix_ops.hpp>

namespace math {
	
class inverse {
private:

	dbcsr::smat_d m_mat_in;
	
	vec<double> est_eig() {
		
		auto desym = m_mat_in->desymmetrize();
		
		dbcsr::iterator<double> iter(desym);
		int n = m_mat_in->nfullrows_total();
		
		vec<double> rowsums_loc(n,0);
		vec<double> rowsums_tot(n,0);
		
		iter.start();
		
		while (iter.blocks_left()) {
			
			iter.next_block();
			
			int ioff = iter.row_offset();
			int joff = iter.col_offset();
			
			int isize = iter.row_size();
			int jsize = iter.col_size();
			
			for (int i = 0; i != isize; ++i) {
				for (int j = 0; j != jsize; ++j) {
					
					rowsums_loc[i + ioff] += iter(i,j);
					
				}
			}
			
		}
		
		iter.stop();
		
		MPI_Allreduce(rowsums_loc.data(),rowsums_tot.data(),n,MPI_DOUBLE,
			MPI_SUM,m_mat_in->get_world().comm());
			
		
		return rowsums_tot;
		
	}
	
public:

	inverse(dbcsr::smat_d& in) : m_mat_in(in) {}

	void compute() {
		
		// estimate eigenvalues
		auto est = this->est_eig();
		
		
		auto it_max = std::max_element(est.begin(),est.end());
		auto it_min = std::min_element(est.begin(),est.end());
	
	    double max_eval = *it_max;
        double min_eval = std::max(0.0, *it_min);
        
        std::cout << "MIN/MAX: " << *it_min << " " << *it_max << std::endl;
        
		std::cout << "Min eval: " << min_eval << " Max eval: " << max_eval
				<< std::endl;
				
		auto scale = 1.0 / (max_eval + min_eval);
		
		std::cout << "scale. " << scale << std::endl;
		
		dbcsr::smat_d X = std::make_shared<dbcsr::mat_d>(
			dbcsr::mat_d::create_template(*m_mat_in).name("X"));
		
		dbcsr::smat_d inv = std::make_shared<dbcsr::mat_d>(
			dbcsr::mat_d::create_template(*m_mat_in).name("inv"));
			
		dbcsr::smat_d temp1 = std::make_shared<dbcsr::mat_d>(
			dbcsr::mat_d::create_template(*m_mat_in)
			.name("temp").type(dbcsr_type_no_symmetry));
			
		dbcsr::smat_d temp2 = std::make_shared<dbcsr::mat_d>(
			dbcsr::mat_d::create_template(*m_mat_in).name("temp2"));
		
		X->copy_in(*m_mat_in);
		X->scale(scale);
		
		dbcsr::print(*m_mat_in);
		
		dbcsr::print(*X);
		
		dbcsr::multiply('N', 'N', *X, *m_mat_in, *inv).perform();
		
		double trace_ideal = m_mat_in->trace();
		double trace_real = 0;
		
		int iter = 0;
		
		while (iter < 1000 && fabs(trace_ideal-trace_real) >= 1e-10) {
			
			dbcsr::multiply('N', 'N', *m_mat_in, *X, *temp1).perform();
			dbcsr::multiply('N', 'N', *X, *temp1, *temp2).perform();
			
			temp1->clear();
			
			X->add(2.0, -1.0, *temp2);
			
			temp2->clear();
			
			dbcsr::multiply('N', 'N', *X, *m_mat_in, *inv).perform();
			
			dbcsr::print(*inv);
			
			trace_real = inv->trace();
			
			std::cout << "ITER: " << iter << " ERR: " << fabs(trace_ideal-trace_real) << std::endl;
			++iter;
			
		}
			
	
  /*
  Array X, Inv;
  X("i,j") = scale * S("i,j");
  Inv("i,j") = X("i,k") * S("k,j");

  auto iter = 0;
  double trace_ideal = S.trange().tiles_range().extent()[0];
  double trace_real = 0.0;
  while (iter < 1000 && std::abs(trace_real - trace_ideal) >= 1e-10) {
    X("i,j") = 2 * X("i,j") - X("i,k") * S("k,l") * X("l,j");
    Inv("i,j") = X("i,k") * S("k,j");
    trace_real = Inv("i,j").trace();
    std::cout << "Error(" << iter
              << ") = " << std::abs(trace_real - trace_ideal) << std::endl;
    ++iter;
  }
	}*/
	}
	
};

}

#endif
