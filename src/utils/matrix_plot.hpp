#ifndef UTILS_MATRIX_PLOT_H
#define UTILS_MATRIX_PLOT_H

#include <dbcsr_conversions.hpp>
#include <fstream>
//#include "utils/matplotlibcpp.hpp"
#include <cmath>

namespace util {

//namespace plt = matplotlibcpp;

inline void plot(dbcsr::shared_matrix<double> mat_in, double thresh,
	std::string filename) {

	auto eigen_colmaj = dbcsr::matrix_to_eigen(*mat_in);
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_rowmaj 
		= eigen_colmaj.cast<float>();
	
	int nrows = eigen_rowmaj.rows();
	int ncols = eigen_colmaj.cols();
	
	float* ptr = eigen_rowmaj.data();
	
	float zeroval = std::log10(fabs(thresh));
	
	for (size_t idx = 0; idx != nrows * ncols; ++idx) {
		float& val = ptr[idx];
		val = (fabs(val) < thresh) ? zeroval : std::log10(fabs(val));
	}
	
	auto mycart = mat_in->get_cart();
	if (mycart.rank() == 0) {
	
		std::ofstream file(filename + ".dat");
		for (int j = 0; j != ncols; ++j) {
			for (int i = 0; i != nrows; ++i) {
				file << eigen_rowmaj(i,j);
				if (i != nrows-1) file << " ";
			}
			if (j != ncols-1) file << '\n';
		}
		file.close();
		
	}
	
	MPI_Barrier(mycart.comm());
	
}
	
} // end namspace

#endif
