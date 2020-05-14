#ifndef DESC_IO_H
#define DESC_IO_H

#include "tensor/dbcsr_conversions.h"
#include <fstream>
#include <cstdio>
#include <vector>
#include <memory>

namespace desc {

template <typename T>
using svector = std::shared_ptr<std::vector<T>>;

bool fexists(const std::string& filename);

template<class Matrix>
void write_binary_mat(const char* filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}
template<class Matrix>
void read_binary_mat(const char* filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
}

void write_2dtensor(dbcsr::stensor<2>& t_in, std::string molname);

void read_2dtensor(dbcsr::stensor<2>& t_in, std::string molname, std::string tensorname, MPI_Comm comm, arrvec<int,2>& blk_sizes);

void write_vector(svector<double>& v_in, std::string molname, std::string vecname, MPI_Comm comm);

void read_vector(svector<double>& v_in, std::string molname, std::string vecname);

} // end namespace

#endif
