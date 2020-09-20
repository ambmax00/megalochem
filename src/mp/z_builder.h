#ifndef MP_Z_BUILDER_H
#define MP_Z_BUILDER_H

#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>
#include <dbcsr_conversions.hpp>
#include <dbcsr_btensor.hpp>
#include <Eigen/Core>
#include "utils/mpi_time.h"
#include "utils/registry.h"
#include "desc/options.h"
#include "desc/molecule.h"

namespace mp {

using SMatrixXi = std::shared_ptr<Eigen::MatrixXi>;
SMatrixXi get_shellpairs(dbcsr::sbtensor<3,double> eri_batched);

class Z {
protected:

	dbcsr::world m_world;
	desc::smolecule m_mol;
	desc::options m_opt;
	util::registry m_reg;
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
	dbcsr::shared_matrix<double> m_zmat;
	dbcsr::shared_tensor<2,double> m_zmat_01;
	
	dbcsr::shared_matrix<double> m_pocc;
	dbcsr::shared_matrix<double> m_pvir;
	dbcsr::shared_matrix<double> m_locc;
	dbcsr::shared_matrix<double> m_lvir;
	
	SMatrixXi m_shellpair_info;

public:

	Z(dbcsr::world& w, desc::smolecule smol, desc::options opt, std::string mname) : 
		m_world(w),
		m_mol(smol),
		m_opt(opt),
		LOG(m_world.comm(), m_opt.get<int>("print", 0)),
		TIME (m_world.comm(), mname) {}

	Z& set_occ_density(dbcsr::smat_d& pocc) {
		m_pocc = pocc;
		return *this;
	}
	
	Z& set_vir_density(dbcsr::smat_d& pvir) {
		m_pvir = pvir;
		return *this;
	}
	
	Z& set_occ_coeff(dbcsr::smat_d& locc) {
		m_locc = locc;
		return *this;
	}
	
	Z& set_vir_coeff(dbcsr::smat_d& lvir) {
		m_lvir = lvir;
		return *this;
	}
	
	Z& set_reg(util::registry& reg) {
		m_reg = reg;
		return *this;
	}
	
	Z& set_shellpair_info(SMatrixXi& spinfo) {
		m_shellpair_info = spinfo;
		return *this;
	}
	 
	virtual void init_tensors() = 0;
	virtual void compute() = 0;
	
	virtual ~Z() {}
	
	dbcsr::shared_matrix<double> zmat() {
		return m_zmat;
	}
	
	void print_info() {
		TIME.print_info();
	}
	
};

class LLMP_FULL_Z : public Z {
private:	
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	dbcsr::sbtensor<3,double> m_z_xbb_batched;
	
	dbcsr::shared_tensor<2,double> m_locc_01;
	dbcsr::shared_tensor<2,double> m_pvir_01;
	
public:

	LLMP_FULL_Z(dbcsr::world& w, desc::smolecule smol, desc::options& opt) :
		Z(w,smol,opt,"LLMP_FULL") {}

	void init_tensors() override;
	void compute() override;
	
	~LLMP_FULL_Z() override {}
	
	
};

class LLMP_MEM_Z : public Z {
private:	
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	
	dbcsr::shared_tensor<2,double> m_locc_01;
	dbcsr::shared_tensor<2,double> m_pvir_01;
	
public:

	LLMP_MEM_Z(dbcsr::world& w, desc::smolecule smol, desc::options& opt) :
		Z(w,smol,opt,"LLMP_MEM") {}

	void init_tensors() override;
	void compute() override;
	
	~LLMP_MEM_Z() override {}
	
	
};

/*
class LLMP_ASYM_Z : public Z {
private:	
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	dbcsr::sbtensor<3,double> m_t_batched;
	
	dbcsr::shared_tensor<2,double> m_locc_01;
	dbcsr::shared_tensor<2,double> m_pvir_01;
	
public:

	LLMP_ASYM_Z(dbcsr::world& w, desc::smolecule& smol, desc::options& opt) :
		Z(w,smol,opt,"LLMP_MEM") {}

	void init_tensors() override;
	void compute() override;
	
	~LLMP_ASYM_Z() override {}
	
	
};
*/

inline std::shared_ptr<Z> get_Z(
	std::string name, dbcsr::world& w, desc::smolecule smol, desc::options opt) {
	
	std::shared_ptr<Z> out;
	Z* ptr = nullptr;
	
	if (name == "LLMPFULL") {
		ptr = new LLMP_FULL_Z(w,smol,opt);
	} else if (name == "LLMPMEM") {
		ptr = new LLMP_MEM_Z(w,smol,opt);
	}
	
	if (!ptr) {
		throw std::runtime_error("INVALID Z BUILDER SPECIFIED");
	}
	
	out.reset(ptr);
	return out;

}

} // end namespace

#endif
