#include "io/io.hpp"
#include <Eigen/Core>

namespace megalochem {

namespace filio {
	
bool fexists(const std::string& filename) {
		std::ifstream ifile(filename.c_str());
		return (bool)ifile;
}

void write_matrix(dbcsr::shared_matrix<double>& m_in, std::string filename) {
		
		std::ofstream file;
				
		std::string file_name = filename;
		
		if (fexists(file_name) && m_in->get_cart().rank() == 0) std::remove(file_name.c_str());
		
		auto eigen_mat = dbcsr::matrix_to_eigen(*m_in);
		
		if (m_in->get_cart().rank() == 0) write_binary_mat(file_name.c_str(), eigen_mat);
		
}

dbcsr::shared_matrix<double> read_matrix(std::string filename, std::string matname, 
	dbcsr::cart wrld, vec<int> rowblksizes, vec<int> colblksizes, dbcsr::type mytype) {
	
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

bool compare_outputs(std::string filename, std::string ref_filename) {
	
	nlohmann::json data, ref_data;
	std::ifstream file, ref_file;
	
	file.open(filename);
	ref_file.open(ref_filename);
	
	file >> data;
	ref_file >> ref_data;
	
	double ref_prec = 1e-8;
	
	std::cout << "Output: " << std::endl;
	std::cout << data.dump(4) << std::endl;
	
	std::cout << "Reference output: " << std::endl;
	std::cout << ref_data.dump(4) << std::endl;
	
	std::function<bool(nlohmann::json&,nlohmann::json&)> validate;
	
	validate = [ref_prec,&validate](nlohmann::json& sub, nlohmann::json& sub_ref) {
	
		for (auto it = sub_ref.begin(); it != sub_ref.end(); ++it) {

			if (sub.find(it.key()) == sub.end()) {
				return false;
			}

			if (it->is_structured() && !it->is_array()) {
				if (!validate(sub[it.key()],sub_ref[it.key()])) {
					return false;
				}
			} else if (it->type() == nlohmann::json::value_t::number_float) {
				if (!(fabs((double)it.value() - (double)sub[it.key()]) < ref_prec)) {
					return false;
				}
			} else if (it.value() != sub[it.key()]) {
				return false;
			}
	
		}
		
		return true;
		
	};
	
	return validate(data, ref_data);
	
}

} // end namesapce 

} // end namespace megalochem
