#include "io/io.h"
#include <Eigen/Core>

namespace filio {
bool fexists(const std::string& filename) {
		std::ifstream ifile(filename.c_str());
		return (bool)ifile;
}

void write_matrix(dbcsr::shared_matrix<double>& m_in, std::string filename) {
		
		std::ofstream file;
				
		std::string file_name = filename;
		
		if (fexists(file_name) && m_in->get_world().rank() == 0) std::remove(file_name.c_str());
		
		auto eigen_mat = dbcsr::matrix_to_eigen(m_in);
		
		if (m_in->get_world().rank() == 0) write_binary_mat(file_name.c_str(), eigen_mat);
		
}

dbcsr::shared_matrix<double> read_matrix(std::string filename, std::string matname, 
	dbcsr::world wrld, vec<int> rowblksizes, vec<int> colblksizes, dbcsr::type mytype) {
	
	std::ifstream file;
	
	std::string file_name = filename;
	
	if (!fexists(file_name)) throw std::runtime_error("File " + file_name + " does not exist.");
	
	Eigen::MatrixXd eigen_mat;
	
	read_binary_mat(file_name.c_str(), eigen_mat);
	
	return dbcsr::eigen_to_matrix(eigen_mat, wrld, matname, rowblksizes, colblksizes, mytype);
	
}

void write_vector(svector<double>& v_in, std::string filename, MPI_Comm comm) {
	
		std::ofstream file;
		
		int myrank = -1;
		
		MPI_Comm_rank(comm, &myrank);
				
		std::string file_name = filename;
		
		if (fexists(file_name) && myrank == 0) std::remove(file_name.c_str());
		
		Eigen::VectorXd eigen_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v_in->data(), v_in->size());
		
		if (myrank == 0) write_binary_mat(file_name.c_str(), eigen_vec);
		
}	

void read_vector(svector<double>& v_in, std::string filename) {
	
	std::ifstream file;
	
	std::string file_name = filename;
	
	if (!fexists(file_name)) throw std::runtime_error("File " + file_name + " does not exist.");
	
	Eigen::VectorXd eigen_vec;
	
	read_binary_mat(file_name.c_str(), eigen_vec);
	
	v_in = std::make_shared<std::vector<double>>(
		std::vector<double>(eigen_vec.data(), eigen_vec.data() + eigen_vec.size()));
	
	
}	

} // end namesapce 
