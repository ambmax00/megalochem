#ifndef FILIO_DATA_HANDLER_H
#define FILIO_DATA_HANDLER_H

#include <string>
#include <filesystem>
#include <fstream>
#include <H5Cpp.h>

#include "desc/molecule.h"
#include "hf/hf_wfn.h"

#include <dbcsr_matrix.hpp>

namespace filio {
	
enum class create_mode {
	create,
	no_create
};

enum class access_mode {
	read_only,
	rdwr
};

template <typename T>
struct adesc {
	std::vector<T> data;
	std::vector<hsize_t> dims;
};

template <typename T>
struct H5toCpp {

	static constexpr H5::PredType type () { 
		if (std::is_same<int,T>::value) {
			return H5::PredType::NATIVE_INT;
		} else if (std::is_same<double,T>::value) {
			return H5::PredType::NATIVE_DOUBLE;
		} else if (std::is_same<float,T>::value) {
			return H5::PredType::NATIVE_FLOAT;
		} else if (std::is_same<bool,T>::value) {
			return H5::PredType::NATIVE_HBOOL;
		} else if (std::is_same<char,T>::value) {
			return H5::PredType::C_S1;
		} else {
			assert(false);
			return H5::PredType::STD_I8BE;
		}
	}
	
};

class data_handler {
private:
	
	H5::H5File* m_file_ptr;
	std::string m_filename;
	dbcsr::world m_world;

public:
	
	// deletes, then creates file if overwrite = true
	data_handler(dbcsr::world wrd, std::string filename, create_mode mode) 
		: m_filename(filename), m_world(wrd) 
	{
		
		if (m_world.rank() == 0) {
		
			if (mode == create_mode::create && std::filesystem::exists(filename)) {
				throw std::runtime_error("File " + filename + " already exists.");
			} else if (mode == create_mode::create) {
				H5::H5File* file = new H5::H5File(m_filename, H5F_ACC_TRUNC);
				delete file;
			}
			
		}
		
	}
		
	
	void open(access_mode mode) {
		
		if (m_world.rank() == 0) {
			if (mode == access_mode::read_only) {
				m_file_ptr = new H5::H5File(m_filename, H5F_ACC_RDONLY);
			} else {
				m_file_ptr = new H5::H5File(m_filename, H5F_ACC_RDWR);
			}
		}
		
	}
	
	void close() {
		
		if (m_file_ptr) delete m_file_ptr;
		
	}
	
	// reading routines
	template <typename T,
		std::enable_if_t<std::is_same<T,std::string>::value, bool> = true
	>
	adesc<T> read(std::string path) {
		
		H5::DataSet dataset = m_file_ptr->openDataSet(path);
		H5::DataSpace dataspace = dataset.getSpace();
		
		int rank = dataspace.getSimpleExtentNdims();
		 
		std::vector<hsize_t> dims_out(rank);
		dataspace.getSimpleExtentDims(dims_out.data(), nullptr);
		hsize_t ntot = std::accumulate(dims_out.begin(), dims_out.end(), 1, 
			std::multiplies<hsize_t>());
		
		H5::StrType strtype(H5::PredType::C_S1, H5T_VARIABLE);
		std::vector<char*> cstr(ntot);
		dataset.read(cstr.data(), strtype);
		
		std::vector<std::string> data_out(ntot);
		for (int i = 0; i != ntot; ++i) {
			data_out[i] = std::string(cstr[i]);
		}
		
		adesc<std::string> out = {std::move(data_out), dims_out};
		
		return out;
		
	}
	
	template <typename T,
		std::enable_if_t<!(std::is_same<T,std::string>::value), bool> = true
	>
	adesc<T> read(std::string path) {
		
		H5::DataSet dataset = m_file_ptr->openDataSet(path);
		H5::DataSpace dataspace = dataset.getSpace();
		
		int rank = dataspace.getSimpleExtentNdims();
		 
		std::vector<hsize_t> dims_out(rank);
		dataspace.getSimpleExtentDims(dims_out.data(), nullptr);
		hsize_t ntot = std::accumulate(dims_out.begin(), dims_out.end(), 1, 
			std::multiplies<hsize_t>());
		
		auto type_class = H5toCpp<T>::type();
		std::vector<T> data_out(ntot);
		dataset.read(data_out.data(), type_class);
		
		adesc<T> out = {std::move(data_out), dims_out};
		
		return out;
		
	}
	
	template <typename T,
		std::enable_if_t<std::is_same<T,std::string>::value, bool> = true
	>
	void write(std::string path, T* data, std::initializer_list<hsize_t> dims) {
		
		std::vector<hsize_t> vdims(dims);
	
		hsize_t ntot = std::accumulate(vdims.begin(), vdims.end(), 1, 
			std::multiplies<hsize_t>());
		int rank = vdims.size();
		
		std::vector<const char*> c_strings(ntot);
		for (hsize_t i = 0; i != ntot; ++i) {
			c_strings[i] = data[i].c_str();
		}
	
		H5::StrType strtype(H5::PredType::C_S1, H5T_VARIABLE);
		
		H5::DataSpace name_space(rank, vdims.data());
		H5::DataSet name_data(m_file_ptr->createDataSet(
			path, strtype, name_space));
		name_data.write(c_strings.data(), strtype);
		
	}
	
	template <typename T,
		std::enable_if_t<!(std::is_same<T,std::string>::value), bool> = true
	>
	void write(std::string path, T* data, std::initializer_list<hsize_t> dims) {
	
		std::vector<hsize_t> vdims(dims);
		
		hsize_t ntot = std::accumulate(vdims.begin(), vdims.end(), 1, 
			std::multiplies<hsize_t>());
		int rank = vdims.size();
		
		auto type_class = H5toCpp<T>::type();
		H5::DataSpace dataspace(rank, vdims.data());
		H5::DataSet dataset(m_file_ptr->createDataSet(path,
			type_class, dataspace));
		dataset.write(data, type_class);
	
	}
	
	template <typename T>
	void write_matrix(std::string path, dbcsr::shared_matrix<double>& m) {
				
		int rank = 2;
		hsize_t dims[2] = {(hsize_t)m->nfullrows_total(), (hsize_t)m->nfullcols_total()};
		
		auto eigen = dbcsr::matrix_to_eigen(m);
		
		if (m_world.rank() == 0) {
			auto type_class = H5toCpp<T>::type();
			H5::DataSpace dataspace(rank, dims);
			H5::DataSet dataset(m_file_ptr->createDataSet(path,
				type_class, dataspace));
			dataset.write(eigen.data(), type_class);
		}
		
	}
	
	template <typename T>
	dbcsr::shared_matrix<T> read_matrix(std::string path, std::string name, 
		vec<int> rblksizes, vec<int> cblksizes, dbcsr::type mtype)
	{
		int nrowstot = std::accumulate(rblksizes.begin(), rblksizes.end(), 0);
		int ncolstot = std::accumulate(cblksizes.begin(), cblksizes.end(), 0);
		
		H5::DataSet dataset = m_file_ptr->openDataSet(path);
		H5::DataSpace dataspace = dataset.getSpace();
		
		int rank = dataspace.getSimpleExtentNdims();
		
		if (rank != 2) 
			throw std::runtime_error("data_handler, read_matrix: wrong rank");
		 
		std::vector<hsize_t> dims(rank);
		dataspace.getSimpleExtentDims(dims.data(), nullptr);
		
		if (dims[0] != nrowstot || dims[1] != ncolstot)
			throw std::runtime_error("data_handler, read_matrix: given block dimensions incompatible");
		
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eigen(dims[0], dims[1]);
		
		auto type_class = H5toCpp<T>::type();
		dataset.read(eigen.data(), type_class);
		
		return dbcsr::eigen_to_matrix(eigen, m_world, name, rblksizes, cblksizes, mtype);
		
	}
	
	desc::smolecule read_molecule();
	hf::shared_hf_wfn read_hf_wfn(desc::smolecule mol);
	
	void write_molecule(desc::smolecule& mol);
	void write_hf_wfn(hf::shared_hf_wfn& hfwfn);
	
}; // end class

}

#endif
