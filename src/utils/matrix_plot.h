#ifndef UTILS_MATRIX_PLOT_H
#define UTILS_MATRIX_PLOT_H

#include <dbcsr_conversions.hpp>
#include "utils/matplotlibcpp.h"
#include <cmath>

namespace util {

namespace plt = matplotlibcpp;

inline void plot(dbcsr::shared_matrix<double> mat_in, double thresh) {

	auto eigen_colmaj = dbcsr::matrix_to_eigen(mat_in);
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
	
	auto myworld = mat_in->get_world();
	if (myworld.rank() == 0) {
	
		const int color = 1;
		
		plt::title("MATRIX");
		plt::imshow(ptr, nrows, ncols, color);
		plt::show();
		
	}
	
	MPI_Barrier(myworld.comm());
	
}
	
} // end namspace

#endif
