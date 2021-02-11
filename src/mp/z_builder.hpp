#ifndef MP_Z_BUILDER_H
#define MP_Z_BUILDER_H

#include <dbcsr_matrix_ops.hpp>
#include <dbcsr_tensor_ops.hpp>
#include <dbcsr_conversions.hpp>
#include <dbcsr_btensor.hpp>
#include <Eigen/Core>
#include "utils/mpi_time.hpp"
#include "utils/ppdirs.hpp"
#include "desc/molecule.hpp"

namespace mp {

enum class zmethod {
	llmp_full,
	llmp_mem,
	llmp_asym
};

inline zmethod str_to_zmethod(std::string str) {
	if (str == "llmp_full") {
		return zmethod::llmp_full;
	} else if (str == "llmp_mem") {
		return zmethod::llmp_mem;
	} else if (str == "llmp_asym") {
		return zmethod::llmp_asym;
	} else {
		throw std::runtime_error("Invalid zbuilder mathod.");
	}
}

using SMatrixXi = std::shared_ptr<Eigen::MatrixXi>;
SMatrixXi get_shellpairs(dbcsr::sbtensor<3,double> eri_batched);

class Z {
protected:

	dbcsr::world m_world;
	desc::shared_molecule m_mol;
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

	Z(dbcsr::world w, desc::shared_molecule smol, int nprint, std::string mname) : 
		m_world(w),
		m_mol(smol),
		LOG(m_world.comm(), nprint),
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
	
	Z& set_shellpair_info(SMatrixXi& spinfo) {
		m_shellpair_info = spinfo;
		return *this;
	}
	 
	virtual void init() = 0;
	virtual void compute() = 0;
	
	virtual ~Z() {}
	
	dbcsr::shared_matrix<double> zmat() {
		return m_zmat;
	}
	
	void print_info() {
		TIME.print_info();
	}
	
};

class create_LLMP_FULL_Z_base;

class LLMP_FULL_Z : public Z {
private:	
	
	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::btype m_intermeds;
	
	dbcsr::sbtensor<3,double> m_FT3c2e_batched;
	dbcsr::shared_tensor<2,double> m_locc_01;
	dbcsr::shared_tensor<2,double> m_pvir_01;
	
	friend class create_LLMP_FULL_Z_base;
	
public:

	LLMP_FULL_Z(dbcsr::world w, desc::shared_molecule smol, int nprint) :
		Z(w,smol,nprint,"LLMP_FULL") {}

	void init() override;
	void compute() override;
	
	~LLMP_FULL_Z() override {}
	
	
};

MAKE_STRUCT(
	LLMP_FULL_Z, Z,
	(
		(world, (dbcsr::world)),
		(mol, (desc::shared_molecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(intermeds, (dbcsr::btype), required, val)
	)
)

#if 0
class create_LL_Z_base;

class LL_Z : public Z {
private:	
	
	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	dbcsr::btype m_intermeds;
	
	dbcsr::shared_tensor<2,double> m_locc_01;
	dbcsr::shared_tensor<2,double> m_pvir_01;
	
	friend class create_LL_Z_base;
	
public:

	LL_Z(dbcsr::world w, desc::shared_molecule smol, int nprint) :
		Z(w,smol,nprint,"LL") {}

	void init() override;
	void compute() override;
	
	~LL_Z() override {}
	
	
};

MAKE_STRUCT(
	LL_Z, Z,
	(
		(world, (dbcsr::world)),
		(mol, (desc::shared_molecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val),
		(intermeds, (dbcsr::btype), required, val)
	)
)
#endif

class create_LLMP_MEM_Z_base;

class LLMP_MEM_Z : public Z {
private:	
	
	dbcsr::sbtensor<3,double> m_eri3c2e_batched;
	
	dbcsr::shared_tensor<2,double> m_locc_01;
	dbcsr::shared_tensor<2,double> m_pvir_01;
	
	friend class create_LLMP_MEM_Z_base;
	
public:

	LLMP_MEM_Z(dbcsr::world w, desc::shared_molecule smol, int nprint) :
		Z(w,smol,nprint,"LLMP_MEM") {}

	void init() override;
	void compute() override;
	
	~LLMP_MEM_Z() override {}
	
	
};

MAKE_STRUCT(
	LLMP_MEM_Z, Z,
	(
		(world, (dbcsr::world)),
		(mol, (desc::shared_molecule)),
		(print, (int))
	),
	(
		(eri3c2e_batched, (dbcsr::sbtensor<3,double>), required, val)
	)
)

class create_LLMP_ASYM_Z_base;

class LLMP_ASYM_Z : public Z {
private:	
	
	dbcsr::sbtensor<3,double> m_t3c2e_right_batched;
	dbcsr::sbtensor<3,double> m_t3c2e_left_batched;
	
	dbcsr::shared_tensor<2,double> m_locc_01;
	dbcsr::shared_tensor<2,double> m_pvir_01;
	
	friend class create_LLMP_ASYM_Z_base;
	
public:

	LLMP_ASYM_Z(dbcsr::world w, desc::shared_molecule& smol, int nprint) :
		Z(w,smol,nprint,"LLMP_MEM") {}

	void init() override;
	void compute() override;
	
	~LLMP_ASYM_Z() override {}
	
};

MAKE_STRUCT(
	LLMP_ASYM_Z, Z,
	(
		(world, (dbcsr::world)),
		(mol, (desc::shared_molecule)),
		(print, (int))
	),
	(
		(t3c2e_left_batched, (dbcsr::sbtensor<3,double>), required, val),
		(t3c2e_right_batched, (dbcsr::sbtensor<3,double>), required, val)
	)
)

} // end namespace

#endif
