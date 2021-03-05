#ifndef IO_DATA_HANDLER_H
#define IO_DATA_HANDLER_H
/*
#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <cassert>
#include <vector>
#include <numeric>

extern "C" {
#include <hdf5.h>
}

namespace filio {

enum class create_mode {
	truncate,
	append
};

enum class access_mode {
	rdwr,
	rdonly
};

namespace fs = std::filesystem;

template <typename T>
struct CPPtoHDF5 {

	static constexpr hid_t memtype () {
		
		if (std::is_same<int,T>::value) {
			return H5T_NATIVE_INT;
			
		} else if (std::is_same<double,T>::value) {
			return H5T_NATIVE_DOUBLE;
			
		} else if (std::is_same<float,T>::value) {
			return H5T_NATIVE_FLOAT;
			
		} else if (std::is_same<bool,T>::value) {
			return H5T_NATIVE_HBOOL;
			
		} else if (std::is_same<char,T>::value) {
			return H5T_NATIVE_CHAR;
			
		} else if (std::is_same<signed char,T>::value) {
			return H5T_NATIVE_SCHAR;
			
		} else if (std::is_same<unsigned char,T>::value) {
			return H5T_NATIVE_UCHAR;
		
		} else if (std::is_same<short,T>::value) {
			return H5T_NATIVE_SHORT;
			
		} else if (std::is_same<unsigned short,T>::value) {
			return H5T_NATIVE_USHORT;
			
		} else if (std::is_same<unsigned,T>::value) {
			return H5T_NATIVE_UINT;
			
		} else if (std::is_same<long,T>::value) {
			return H5T_NATIVE_LONG;
			
		} else if (std::is_same<unsigned long,T>::value) {
			return H5T_NATIVE_ULONG;
			
		} else if (std::is_same<long long,T>::value) {
			return H5T_NATIVE_LLONG;
			
		} else if (std::is_same<long double,T>::value) {
			return H5T_NATIVE_LDOUBLE;
			
		} else if (std::is_same<hsize_t,T>::value) {
			return H5T_NATIVE_HSIZE;
			
		} else if (std::is_same<hssize_t,T>::value) {
			return H5T_NATIVE_HSSIZE;
			
		} else if (std::is_same<herr_t,T>::value) {
			return H5T_NATIVE_HERR;
			
		} else {
			
			assert(false);
			return 0;
		}
	}

	static constexpr hid_t filetype () {
		
		if (std::is_same<int,T>::value) {
			return H5T_STD_I32LE;
			
		} else if (std::is_same<double,T>::value) {
			return H5T_IEEE_F64LE;
			
		} else if (std::is_same<float,T>::value) {
			return H5T_IEEE_F32LE;
			
		} else if (std::is_same<bool,T>::value) {
			return H5T_STD_U8LE;
			
		} else if (std::is_same<char,T>::value) {
			return H5T_STD_I8LE;
			
		} else if (std::is_same<signed char,T>::value) {
			return H5T_STD_I8LE;
			
		} else if (std::is_same<unsigned char,T>::value) {
			return H5T_STD_U8LE;
		
		} else if (std::is_same<short,T>::value) {
			return H5T_STD_I16LE;
			
		} else if (std::is_same<unsigned short,T>::value) {
			return H5T_STD_U16LE;
			
		} else if (std::is_same<unsigned,T>::value) {
			return H5T_STD_U32LE;
			
		} else if (std::is_same<long,T>::value) {
			return H5T_STD_I32LE;
			
		} else if (std::is_same<unsigned long,T>::value) {
			return H5T_STD_U32LE;
			
		} else if (std::is_same<long long,T>::value) {
			return H5T_STD_I64LE;
			
		} else if (std::is_same<long double,T>::value) {
			return H5T_IEEE_F64LE;
			
		} else if (std::is_same<hsize_t,T>::value) {
			return H5T_NATIVE_HSIZE;
			
		} else {
			
			assert(false);
			return 0;
		}
	}
	
};

template <typename T>
struct data_array {
	std::vector<hsize_t> dims;
	std::vector<T> data;
};

class data_handler {
private:

	std::string _filename, _abs_filename;
	MPI_Comm _comm;
	int _rank, _size;
	
	hid_t _plist_id, _file_id;
	bool _is_open;
	
	static const size_t _max_strlength = 128;

	struct alignas(_max_strlength) fixed_string {
		char data[_max_strlength];
	};

public:

	data_handler(std::string filename, create_mode cmode, MPI_Comm comm) 
		: _filename(filename), _abs_filename(fs::absolute(filename)), 
		_is_open(false), _comm(comm) 
	{

		_plist_id = H5Pcreate(H5P_FILE_ACCESS);
		H5Pset_fapl_mpio(_plist_id, _comm, MPI_INFO_NULL);
		
		MPI_Comm_rank(_comm, &_rank);
		MPI_Comm_size(_comm, &_size);

		if (cmode == create_mode::truncate) {
			
			if (_rank == 0 && fs::exists(_filename)) fs::remove(_filename);
			auto file_id = H5Fcreate(_abs_filename.c_str(), H5F_ACC_TRUNC, 
				H5P_DEFAULT, _plist_id);
			H5Fclose(file_id);

		} else {

			if (!fs::exists(_filename)) 
				throw std::runtime_error(
				"Datahandler: file " + _abs_filename + 
			       	" does not exist!");	       
		}
	
	}
	
	void open(access_mode amode) {
		
		if (_is_open) {
			throw std::runtime_error("File " + _abs_filename + " already opened.");
		}
		
		unsigned h5mode = (amode == access_mode::rdonly) ? 
			H5F_ACC_RDONLY : H5F_ACC_RDWR;
			
		_file_id = H5Fopen(_abs_filename.c_str(), h5mode, _plist_id);
		_is_open = true;
		
	} 
	
	void close() {
		if (_is_open) {
			H5Fclose(_file_id);
			_is_open = false;
		}
	}
	
	void create_group(std::string gname) {
		auto group = H5Gcreate(_file_id, gname.c_str(),
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
		H5Gclose(group);
	}
	
	template <typename T>
	void write(std::string vname, T val, int wrank = 0) {
		write(vname, &val, {1}, wrank);		
	}
	
	template <typename T,
		std::enable_if_t<!(std::is_same<T,std::string>::value), bool> = true
	>
	void write(std::string vname, T* data, std::vector<hsize_t> dims, int wrank = 0) {
		
		auto space = H5Screate_simple(dims.size(),dims.data(),NULL);
		
		auto memtype = CPPtoHDF5<T>::memtype();
		auto filetype = CPPtoHDF5<T>::filetype();
		
		hid_t data_plist = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(data_plist, H5FD_MPIO_INDEPENDENT);
		
		auto dset = H5Dcreate(_file_id, vname.c_str(), filetype, space,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			
		if (wrank == _rank) {
			auto err = H5Dwrite(dset, memtype, H5S_ALL, H5S_ALL,
				data_plist, data);
		}
		
		H5Dclose(dset);
		H5Sclose(space);
			
		H5Pclose(data_plist);
		
	}
	
	template <typename T,
		std::enable_if_t<(std::is_same<T,std::string>::value), bool> = true
	>
	void write(std::string vname, T* data, std::vector<hsize_t> dims, int wrank = 0) {
		
		auto strtype = H5Tcopy(H5T_C_S1);
		auto status = H5Tset_size(strtype, _max_strlength);
		
		auto space = H5Screate_simple(dims.size(),dims.data(),NULL);
				
		hid_t data_plist = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(data_plist, H5FD_MPIO_INDEPENDENT);
		
		hsize_t ntot = std::accumulate(dims.begin(), dims.end(), 1, 
			std::multiplies<hsize_t>());
		
		std::vector<fixed_string> c_strings(ntot);
		
		for (hsize_t ii = 0; ii != ntot; ++ii) {
			
			std::copy(data[ii].begin(), data[ii].end(), c_strings[ii].data);
			std::cout << c_strings[ii].data << std::endl;
 			c_strings[ii].data[data[ii].size()] = '\0';
						
			if (data[ii].size() > _max_strlength-1) {
				throw std::runtime_error(
				"Datahandler: string exceeds size limit.");
			}
		}
		
		auto dset = H5Dcreate(_file_id, vname.c_str(), strtype, space,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
			
		if (wrank == _rank) {
			auto err = H5Dwrite(dset, strtype, H5S_ALL, H5S_ALL,
				data_plist, c_strings.data());
		}
		
		H5Dclose(dset);
		H5Sclose(space);
			
		H5Pclose(data_plist);
		
	}
	
	template <typename T> 
	T read_single(std::string vname) {
		auto darray = read<T>(vname);
		return darray.data[0];
	}
	
	template <typename T,
		std::enable_if_t<!(std::is_same<T,std::string>::value), bool> = true
	>
	data_array<T> read(std::string name) {
		
		hid_t dataset = H5Dopen(_file_id, name.c_str(), H5P_DEFAULT);
		hid_t space = H5Dget_space(dataset);
		
		int rank = H5Sget_simple_extent_ndims(space);
		
		std::vector<hsize_t> dims(rank);
		auto status = H5Sget_simple_extent_dims(space, dims.data(), NULL);
		
		hid_t data_plist = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(data_plist, H5FD_MPIO_COLLECTIVE);
	
		hsize_t ntot = std::accumulate(dims.begin(), dims.end(), 1,
			std::multiplies<hsize_t>());
	
		std::vector<T> buf(ntot);
	
		auto memtype = CPPtoHDF5<T>::memtype();
	
		auto err = H5Dread(dataset, memtype, H5S_ALL, H5S_ALL, 
			data_plist, buf.data());
		
		H5Dclose(dataset);
		H5Sclose(space);
		
		return data_array<T>{std::move(dims), std::move(buf)};
		
	}
	
	template <typename T,
		std::enable_if_t<(std::is_same<T,std::string>::value), bool> = true
	>
	data_array<T> read(std::string name) {
		
		hid_t dataset = H5Dopen(_file_id, name.c_str(), H5P_DEFAULT);
		hid_t space = H5Dget_space(dataset);
		
		int rank = H5Sget_simple_extent_ndims(space);
		
		std::vector<hsize_t> dims(rank);
		auto status = H5Sget_simple_extent_dims(space, dims.data(), NULL);
		
		hid_t data_plist = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(data_plist, H5FD_MPIO_COLLECTIVE);
	
		hsize_t ntot = std::accumulate(dims.begin(), dims.end(), 1,
			std::multiplies<hsize_t>());
	
		std::vector<T> outbuf(ntot);
		std::vector<fixed_string> readbuf(ntot);
	
		auto strtype = H5Tcopy(H5T_C_S1);
		status = H5Tset_size(strtype, _max_strlength);
	
		auto err = H5Dread(dataset, strtype, H5S_ALL, H5S_ALL, 
			data_plist, readbuf.data());
		
		H5Dclose(dataset);
		H5Sclose(space);
		
		for (hsize_t ii = 0; ii != ntot; ++ii) {
			outbuf[ii] = readbuf[ii].data;
		}
		
		return data_array<T>{std::move(dims), std::move(outbuf)};
		
	}
	
	bool exists(std::string name) {
		return H5Lexists(_file_id, name.c_str(), H5P_DEFAULT);
	}

	~data_handler() {
		H5Pclose(_plist_id);
		if (_is_open) close();
	}	

};

} // end namespace*/

#endif

