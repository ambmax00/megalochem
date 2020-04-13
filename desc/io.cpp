#include "desc/io.h"

namespace desc {

bool fexists(const std::string& filename) {
		std::ifstream ifile(filename.c_str());
		return (bool)ifile;
}

void write_2dtensor(dbcsr::stensor<2>& t_in, std::string molname) {
		
		std::ofstream file;
		
		int myrank = -1;
		
		MPI_Comm_rank(t_in->comm(), &myrank);
				
		std::string file_name = molname + "_" + t_in->name() + ".dat";
		
		if (fexists(file_name) && myrank == 0) std::remove(file_name.c_str());
		
		auto eigen_mat = dbcsr::tensor_to_eigen(*t_in);
		
		if (myrank == 0) write_binary_mat(file_name.c_str(), eigen_mat);
		
}

void read_2dtensor(dbcsr::stensor<2>& t_in, std::string molname, std::string tensorname, MPI_Comm comm, arrvec<int,2>& blk_sizes) {
	
	std::ifstream file;
	
	std::string file_name = molname + "_" + tensorname + ".dat";
	
	if (!fexists(file_name)) throw std::runtime_error("File " + file_name + " does not exist.");
	
	Eigen::MatrixXd eigen_mat;
	
	read_binary_mat(file_name.c_str(), eigen_mat);
	
	dbcsr::pgrid<2> grid2(comm);
	t_in = std::make_shared<dbcsr::tensor<2>>(eigen_to_tensor(eigen_mat, tensorname, grid2, vec<int>{0}, vec<int>{1}, blk_sizes));
	
}

void write_vector(svector<double>& v_in, std::string molname, std::string vecname, MPI_Comm comm) {
	
		std::ofstream file;
		
		int myrank = -1;
		
		MPI_Comm_rank(comm, &myrank);
				
		std::string file_name = molname + "_" + vecname + ".dat";
		
		if (fexists(file_name) && myrank == 0) std::remove(file_name.c_str());
		
		Eigen::VectorXd eigen_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(v_in->data(), v_in->size());
		
		if (myrank == 0) write_binary_mat(file_name.c_str(), eigen_vec);
		
}	

void read_vector(svector<double>& v_in, std::string molname, std::string vecname) {
	
	std::ifstream file;
	
	std::string file_name = molname + "_" + vecname + ".dat";
	
	if (!fexists(file_name)) throw std::runtime_error("File " + file_name + " does not exist.");
	
	Eigen::VectorXd eigen_vec;
	
	read_binary_mat(file_name.c_str(), eigen_vec);
	
	v_in = std::make_shared<std::vector<double>>(
		std::vector<double>(eigen_vec.data(), eigen_vec.data() + eigen_vec.size()));
	
	
}
	

} // end namesapce 
