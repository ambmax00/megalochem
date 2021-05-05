#ifndef DESC_WFN_H
#define DESC_WFN_H

#include "desc/molecule.hpp"
#include "io/data_handler.hpp"

#include <dbcsr_matrix.hpp>
#include <dbcsr_conversions.hpp>
#include <mpi.h>

#include <string>
#include <fstream>
#include <filesystem>

namespace megalochem {

namespace desc {
	
class hf_wavefunction {
private:
	
	dbcsr::smatrix<double> m_c_bo_A;
	dbcsr::smatrix<double> m_c_bv_A;

	dbcsr::smatrix<double> m_c_bo_B;
	dbcsr::smatrix<double> m_c_bv_B;
	
	svector<double> m_eps_occ_A;
	svector<double> m_eps_occ_B;
	svector<double> m_eps_vir_A;
	svector<double> m_eps_vir_B;
	
	double m_scf_energy;
	double m_nuc_energy;
	double m_wfn_energy;
		
public:

	hf_wavefunction(
		dbcsr::smatrix<double> c_bo_A,
		dbcsr::smatrix<double> c_bo_B,
		dbcsr::smatrix<double> c_bv_A,
		dbcsr::smatrix<double> c_bv_B,
		svector<double> eps_occ_A,
		svector<double> eps_occ_B,
		svector<double> eps_vir_A,
		svector<double> eps_vir_B,
		double scf_energy,
		double nuc_energy,
		double wfn_energy
	) : 
		m_c_bo_A(c_bo_A), m_c_bv_A(c_bv_A), m_c_bo_B(c_bo_B), 
		m_c_bv_B(c_bv_B), m_eps_occ_A(eps_occ_A), 
		m_eps_occ_B(eps_occ_B), m_eps_vir_A(eps_vir_A), 
		m_eps_vir_B(eps_vir_B), m_scf_energy(scf_energy),
		m_nuc_energy(nuc_energy), m_wfn_energy(wfn_energy)
	{}
				
	dbcsr::smatrix<double> c_bo_A() { return m_c_bo_A; }
	dbcsr::smatrix<double> c_bv_A() { return m_c_bv_A; }
	
	dbcsr::smatrix<double> c_bo_B() { return m_c_bo_B; }
	dbcsr::smatrix<double> c_bv_B() { return m_c_bv_B; }
	
	svector<double> eps_occ_A() { return m_eps_occ_A; }
	svector<double> eps_vir_A() { return m_eps_vir_A; }
	svector<double> eps_occ_B() { return m_eps_occ_B; }
	svector<double> eps_vir_B() { return m_eps_vir_B; }
	
	double scf_energy() { return m_scf_energy; }
	double nuc_energy() { return m_nuc_energy; }
	double wfn_energy() { return m_wfn_energy; }
	
};

class mp_wavefunction {
protected:

	double m_mp_os_energy;
	double m_mp_ss_energy;
	double m_mp_energy; 
	
public:

	mp_wavefunction(
		double mp_os_energy,
		double mp_ss_energy,
		double mp_energy) :
		m_mp_os_energy(mp_os_energy),
		m_mp_ss_energy(mp_ss_energy),
		m_mp_energy(mp_energy)
	{}
	
	double mp_os_energy() { return m_mp_os_energy; }
	double mp_ss_energy() { return m_mp_ss_energy; }
	double mp_energy() { return m_mp_energy; }
	
	~mp_wavefunction() {}
	
};

class adc_wavefunction {
private:

	std::vector<double> m_dav_eigvals;
	std::vector<dbcsr::shared_matrix<double>> m_dav_eigvecs;
	
	bool m_blocked; // indicates if all roots are converged
	
public:

	adc_wavefunction(
		bool blocked,
		std::vector<double> dav_eigvals,
		std::vector<dbcsr::shared_matrix<double>> dav_eigvecs) :
		m_blocked(blocked),
		m_dav_eigvals(dav_eigvals),
		m_dav_eigvecs(dav_eigvecs)
	{}
	
	bool blocked() { return m_blocked; }
	
	std::vector<double> davidson_eigenvalues() {
		return m_dav_eigvals;
	}
	
	std::vector<dbcsr::shared_matrix<double>> davidson_eigenvectors() {
		return m_dav_eigvecs;
	}
		
	~adc_wavefunction() {}
	
	
};

using shared_adc_wavefunction = std::shared_ptr<adc_wavefunction>;
using shared_mp_wavefunction = std::shared_ptr<mp_wavefunction>;
using shared_hf_wavefunction = std::shared_ptr<hf_wavefunction>;

struct wavefunction {

	shared_molecule mol;
	shared_hf_wavefunction hf_wfn;
	shared_mp_wavefunction mp_wfn;
	shared_adc_wavefunction adc_wfn;
	
};

using shared_wavefunction = std::shared_ptr<wavefunction>;

inline void write_hfwfn(std::string name, hf_wavefunction& hfwfn, filio::data_handler& dh) {
	
	dh.open(filio::access_mode::rdwr);
	dh.create_group(name);
		
	// write matrices
	auto write_mat = [&](auto mat, std::string name) {
		if (mat) {
			hsize_t nrows = mat->nfullrows_total();
			hsize_t ncols = mat->nfullcols_total();
			auto eigen = dbcsr::matrix_to_eigen<double,Eigen::RowMajor>(*mat);
			dh.write<double>(name, eigen.data(), {nrows,ncols}, 0);
		}
	};
	
	auto write_vec = [&](auto vec, std::string name) {
		if (vec) {
			hsize_t ntot = vec->size();
			dh.write<double>(name, vec->data(), {ntot}, 0);
		}
	}; 
	
	write_mat(hfwfn.c_bo_A(), name + "/c_bo_A");
	write_mat(hfwfn.c_bo_B(), name + "/c_bo_B");
	write_mat(hfwfn.c_bv_A(), name + "/c_bv_A");
	write_mat(hfwfn.c_bv_B(), name + "/c_bv_B");
	write_vec(hfwfn.eps_occ_A(), name + "/eps_occ_A");
	write_vec(hfwfn.eps_vir_A(), name + "/eps_vir_A");
	write_vec(hfwfn.eps_occ_B(), name + "/eps_occ_B");
	write_vec(hfwfn.eps_vir_B(), name + "/eps_vir_B");
	
	dh.write<double>(name + "/scf_energy", hfwfn.scf_energy());
	dh.write<double>(name + "/nuc_energy", hfwfn.nuc_energy());
	dh.write<double>(name + "/wfn_energy", hfwfn.wfn_energy());
	
	dh.close();

}

inline desc::shared_hf_wavefunction read_hfwfn(std::string name, desc::shared_molecule mol, 
	world w, filio::data_handler& dh) 
{
	
	dh.open(filio::access_mode::rdonly);
	
	// read matrix
	auto read_mat = [&](std::string name, std::vector<int> r, 
		std::vector<int> c, bool throw_if_not_present) 
	{
		
		std::cout << "Looking for " << name << std::endl;
		
		int nrows = std::accumulate(r.begin(), r.end(), 0);
		int ncols = std::accumulate(c.begin(), c.end(), 0);
		
		if (throw_if_not_present && !dh.exists(name)) {
			throw std::runtime_error("Read hfwfn: could not find " + name + ".");
		} else if (!dh.exists(name)) {
			std::cout << name << " not present" << std::endl;
			dbcsr::shared_matrix<double> out = nullptr;
			return out;
		}
		
		auto darray = dh.read<double>(name);
		
		if (nrows != darray.dims[0] || ncols != darray.dims[1]) {
			throw std::runtime_error("Read hfwfn: incompatible matrix dimensions.");
		}
		
		Eigen::Map<MatrixX<double,Eigen::RowMajor>>
			emap(darray.data.data(), nrows, ncols);
		
		auto mat = dbcsr::eigen_to_matrix(emap, w.dbcsr_grid(), name, r, c, 
			dbcsr::type::no_symmetry);
		
		return mat;
		
	};
	
	auto read_vec = [&](std::string name, bool throw_if_not_present) {
		
		std::cout << "Looking for " << name << std::endl;
		
		if (throw_if_not_present && !dh.exists(name)) {
			throw std::runtime_error("Read hfwfn: could not find " + name + ".");
		} else if (!dh.exists(name)) {
			std::cout << name << " not present" << std::endl;
			std::shared_ptr<std::vector<double>> out = nullptr;
			return out;
		}
		
		auto darray = dh.read<double>(name);
		return std::make_shared<std::vector<double>>(darray.data);
		
	};
	
	auto b = mol->dims().b();
	auto oa = mol->dims().oa();
	auto ob = mol->dims().ob();
	auto va = mol->dims().va();
	auto vb = mol->dims().vb();
	
	auto c_bo_A = read_mat(name + "/c_bo_A", b, oa, true);
	auto c_bo_B = read_mat(name + "/c_bo_B", b, ob, false);
	auto c_bv_A = read_mat(name + "/c_bv_A", b, va, true);
	auto c_bv_B = read_mat(name + "/c_bv_B", b, vb, false);
	
	auto eps_occ_A = read_vec(name + "/eps_occ_A", true);
	auto eps_occ_B = read_vec(name + "/eps_occ_B", false);
	auto eps_vir_A = read_vec(name + "/eps_vir_A", true);
	auto eps_vir_B = read_vec(name + "/eps_vir_B", false);
	
	double scf_energy = dh.read_single<double>(name + "/scf_energy");
	double nuc_energy = dh.read_single<double>(name + "/nuc_energy");
	double wfn_energy = dh.read_single<double>(name + "/wfn_energy");
	
	auto hfwfn = std::make_shared<desc::hf_wavefunction>(
		c_bo_A, c_bo_B, c_bv_A, c_bv_B, eps_occ_A, eps_occ_B,
		eps_vir_A, eps_vir_B, scf_energy, nuc_energy, wfn_energy
	);
		
	return hfwfn;
	
}

} // end namespace hf 

} // end namespace megalochem
	
#endif
	
	