#ifndef FOCK_JK_BUILDER_H
#define FOCK_JK_BUILDER_H

#include "ints/aofactory.h"
#include "utils/registry.h"
#include "desc/molecule.h"
#include "utils/mpi_time.h"
#include <dbcsr_btensor.hpp>
#include <dbcsr_matrix.hpp>

namespace fock {
	
using ilist = std::initializer_list<int>;
	
class JK {
protected:
	
	desc::smolecule m_mol;
	dbcsr::world m_world;
	
	util::registry m_reg;
	
	util::mpi_log LOG;
	util::mpi_time TIME;
	
	bool m_SAD_iter;
	int m_SAD_rank;
	
	bool m_sym = true;
	
	dbcsr::shared_matrix<double> m_mat_A;
	dbcsr::shared_matrix<double> m_mat_B;
	
	void init();
	virtual void init_tensors() = 0;
	
public:
	
	JK(dbcsr::world& w, desc::smolecule smol, util::registry reg, 
		int print, std::string name) : 
		m_world(w),
		LOG(m_world.comm(),print),
		TIME(m_world.comm(),name) 
	{
			init();
	}
	
	void set_sym(bool sym) { m_sym = sym; }
	
	void set_SAD(bool SAD, int rank) { 
		m_SAD_iter = SAD; 
		m_SAD_rank = rank;
	}
	
	virtual ~JK() {}
	
	virtual void batch_init() = 0;
	virtual void batch_finalize() = 0;
	
	void print_info() { TIME.print_info(); }
	
	virtual void compute_block(ilist idx) = 0;
	
	dbcsr::shared_matrix<double> get_A() { return m_mat_A; }
	dbcsr::shared_matrix<double> get_B() { return m_mat_B; }
	
};

/*
class EXACT_J : public JK {
private:

	dbcsr::sbtensor<4,double> m_eri_batched;
	dbcsr::shared_tensor<3,double> m_J_bbd;
	dbcsr::shared_tensor<3,double> m_ptot_bbd;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;

public:
	
	EXACT_J(dbcsr::world& w, desc::smolecule smol, int nprint) :
		JK(w,mol,nprint,"EXACT_J") 
	{
		init_tensors();
	}
	
	void batch_init() override;
	void batch_finalize() override;
		
	void compute_Jblock(ilist idx) override;
	void init_tensors() override;
	
};

class EXACT_K : public JK {
private:

	dbcsr::sbtensor<4,double> m_eri_batched;
	dbcsr::shared_tensor<3,double> m_K_bbd;
	dbcsr::shared_tensor<3,double> m_p_bbd;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
	
public:

	EXACT_K(dbcsr::world& w, desc::smolecule smol, int nprint) :
		JK(w,mol,nprint,"EXACT_K")
	{
		init_tensors();
	}
	
	void batch_init() override;
	void batch_finalize() override;
	
	void compute_Kblock(ilist idx) override;
	void init_tensors() override;
	
};*/

class BATCHED_DF_J : public JK {
private:

	dbcsr::sbtensor<3,double> m_eri_batched;
	
	dbcsr::shared_tensor<3,double> m_J_bbd;
	dbcsr::shared_tensor<3,double> m_ptot_bbd;
	dbcsr::shared_tensor<2,double> m_gp_xd;
	dbcsr::shared_tensor<2,double> m_gq_xd;
	dbcsr::shared_tensor<2,double> m_inv;
	
	dbcsr::shared_pgrid<3> m_spgrid_bbd;
	dbcsr::shared_pgrid<2> m_spgrid_xd;

public:

	BATCHED_DF_J(dbcsr::world& w, desc::smolecule mol, 
		util::registry reg, int nprint) :
		JK(w,mol,reg,nprint,"BATCHED_DF_J") 
	{
		init_tensors();
	}
	
	void batch_init() override;
	void batch_finalize() override;
	
	void compute_block(ilist idx) override;
	void init_tensors() override;
	
	~BATCHED_DF_J() {}
	
};

/*
class BATCHED_DFMO_K : public JK {
private:
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	
	dbcsr::shared_tensor<3,double> m_HT1_xmb_02_1;
	dbcsr::shared_tensor<3,double> m_HT1_xmb_0_12;
	dbcsr::shared_tensor<3,double> m_HT2_xmb_0_12;
	dbcsr::shared_tensor<3,double> m_HT2_xmb_01_2;
	
	dbcsr::shared_tensor<2,double> m_c_bm;
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_invsqrt;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
public:

	BATCHED_DFMO_K(dbcsr::world& w, desc::smolecule mol, int nprint) :
		JK(w,mol,nprint,"BATCHED_DFMO_K") 
	{
		init_tensors();
	}
	
	void batch_init() override;
	void batch_finalize() override;
	
	void compute_Kblock(ilist idx) override;
	void init_tensors() override;
	
	~BATCHED_DFMO_K() {}
	
};

class BATCHED_DFAO_K : public JK {
private:
	
	dbcsr::sbtensor<3,double> m_eri_batched;
	dbcsr::sbtensor<3,double> m_c_xbb_batched;
		
	dbcsr::shared_tensor<3,double> m_c_xbb_1_02;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_01_2;
	dbcsr::shared_tensor<3,double> m_cbar_xbb_1_02;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
public:

	BATCHED_DFAO_K(dbcsr::world& w, desc::smolecule mol, int nprint) :
		JK(w,mol,nprint,"BATCHED_DFAO_K") 
	{
		init_tensors();
	}
	
	void batch_init() override;
	void batch_finalize() override;
	
	void compute_Kblock(ilist idx) override;
	void init_tensors() override;
	
	~BATCHED_DFAO_K() {}
	
};

class BATCHED_PARI_K : public JK {
private:

	dbcsr::sbtensor<3,double> m_eri_batched;
	dbcsr::shared_tensor<3,double> m_cfit_xbb;
	
	dbcsr::shared_tensor<2,double> m_K_01;
	dbcsr::shared_tensor<2,double> m_p_bb;
	dbcsr::shared_tensor<2,double> m_s_xx;
	
	dbcsr::shared_pgrid<2> m_spgrid2;
	
public:

	BATCHED_PARI_K(dbcsr::world& w, desc::smolecule mol, int nprint) :
		JK(w,mol,nprint,"BATCHED_PARI_K") 
	{
		init_tensors();
	}
	
	void batch_init() override;
	void batch_finalize() override;
	
	void compute_Kblock(int ix) override;
	void init_tensors() override;
	
	~BATCHED_PARI_K() {}
	
};*/

inline std::shared_ptr<JK> get_JK(
	std::string name, dbcsr::world& w, desc::smolecule mol, util::registry reg, int nprint) {
	
	std::shared_ptr<JK> out;
	JK* ptr = nullptr;
	
	if (name == "exact_J") {
		ptr = nullptr; //new EXACT_J(w,opt,mol,nprint);
	} else if (name == "batchdf_J") {
		ptr = new BATCHED_DF_J(w,mol,reg,nprint);
	} else if (name == "exact_K") {
		ptr = nullptr; //new EXACT_K(w,mol,nprint);
	} else if (name == "batchdfao_K") {
		ptr = nullptr; //new BATCHED_DFAO_K(w,mol,nprint);
	} else if (name == "batchdfmo_K") {
		ptr = nullptr; //new BATCHED_DFMO_K(w,mol,nprint);
	} else if (name == "batchpari_K") {
		ptr = nullptr; //new BATCHED_PARI_K(w,mol,nprint);
	}
	
	if (!ptr) {
		throw std::runtime_error("INVALID J/K BUILDER SPECIFIED");
	}
	
	out.reset(ptr);
	return out;

}

} // end namespace

#endif 
