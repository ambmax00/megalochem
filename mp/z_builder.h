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

namespace mp {

class Z {
protected:

	dbcsr::world m_world;
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
	
	Eigen::MatrixXi get_shellpairs(dbcsr::sbtensor<3,double> eri_batched);

public:

	Z(dbcsr::world& w, desc::options opt, std::string mname) : 
		m_world(w),
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
	
	Eigen::MatrixXi m_shell_idx;
	
	bool m_force_sparsity;
	
public:

	LLMP_FULL_Z(dbcsr::world& w, desc::options& opt) :
		Z(w,opt,"LLMP_FULL") {}

	void init_tensors() override;
	void compute() override;
	
	~LLMP_FULL_Z() override {}
	
	
};

class LLMP_MEM_Z : public Z {
private:	
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	
	dbcsr::shared_tensor<2,double> m_locc_01;
	dbcsr::shared_tensor<2,double> m_pvir_01;
	
	Eigen::MatrixXi m_shell_idx;
	
	bool m_force_sparsity;
	
public:

	LLMP_MEM_Z(dbcsr::world& w, desc::options& opt) :
		Z(w,opt,"LLMP_MEM") {}

	void init_tensors() override;
	void compute() override;
	
	~LLMP_MEM_Z() override {}
	
	
};

inline std::shared_ptr<Z> get_Z(
	std::string name, dbcsr::world& w, desc::options opt) {
	
	std::shared_ptr<Z> out;
	Z* ptr = nullptr;
	
	if (name == "LLMPFULL") {
		ptr = new LLMP_FULL_Z(w,opt);
	} else if (name == "LLMPMEM") {
		ptr = new LLMP_MEM_Z(w,opt);
	}
	
	if (!ptr) {
		throw std::runtime_error("INVALID Z BUILDER SPECIFIED");
	}
	
	out.reset(ptr);
	return out;

}

} // end namespace

#endif
