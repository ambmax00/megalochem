#ifndef INTS_INTEGRALS_H
#define INTS_INTEGRALS_H

#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>
#include "utils/pool.h"
#include "utils/params.hpp"
#include "desc/basis.h"
#include "ints/screening.h"
#include <vector>
#include <mpi.h>

// =====================================================================
//                          LIBINT2
// =====================================================================

#ifdef USE_LIBINT
#include <libint2.hpp>

namespace ints {

using cluster_vec = std::vector<const std::vector<std::vector<libint2::Shell>>*>;

void calc_ints(dbcsr::mat_d& m_out, util::ShrPool<libint2::Engine>& engine,
	cluster_vec& basvec);

void calc_ints(dbcsr::tensor<2>& t_out, util::ShrPool<libint2::Engine>& eng,
	cluster_vec& basvec);
	
void calc_ints(dbcsr::tensor<3>& t_out, util::ShrPool<libint2::Engine>& eng,
	cluster_vec& basvec, screener* scr = nullptr);
	
void calc_ints(dbcsr::tensor<4>& t_out, util::ShrPool<libint2::Engine>& eng,
	cluster_vec& basvec);
	
void calc_ints_schwarz_mn(dbcsr::mat_d& m_out, util::ShrPool<libint2::Engine>& engine,
	cluster_vec& basvec);
	
void calc_ints_schwarz_x(dbcsr::mat_d& m_out, util::ShrPool<libint2::Engine>& engine,
	cluster_vec& basvec);

} // end namespace

#endif // USE_LIBINT

// =====================================================================
//                       LIBCINT
// =====================================================================

#ifdef USE_LIBCINT
extern "C" {
#include <cint.h>

int cint1e_ovlp_sph(double *out, const FINT *shls,
	const FINT *atm, FINT natm, const FINT *bas, FINT nbas, const double *env,
	const CINTOpt *opt);

int cint1e_nuc_sph(double *out, const FINT *shls,
	const FINT *atm, FINT natm, const FINT *bas, FINT nbas, const double *env,
	const CINTOpt *opt);


int cint1e_kin_sph(double *out, const FINT *shls,
	const FINT *atm, FINT natm, const FINT *bas, FINT nbas, const double *env,
	const CINTOpt *opt);
	
int cint3c2e_sph(double *out, const FINT *shls,
	const FINT *atm, FINT natm, const FINT *bas, FINT nbas, const double *env,
	const CINTOpt *opt);
	
int cint2e_coulerf_sph(double *out, const FINT *shls,
	const FINT *atm, FINT natm, const FINT *bas, FINT nbas, const double *env,
	const CINTOpt *opt);


}

typedef FINT (*CINTIntegralFunction)(double *out, const FINT *shls,
		const FINT *atm, FINT natm, const FINT *bas, FINT nbas, const double *env,
		const CINTOpt *opt);
		
namespace ints {
		
void calc_ints(dbcsr::mat_d& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		FINT *atm, FINT natm, FINT* bas, FINT nbas, double* env, int max_l);
		
void calc_ints(dbcsr::tensor<3,double>& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		FINT *atm, FINT natm, FINT* bas, FINT nbas, double* env, int max_l);
		
void calc_ints(dbcsr::tensor<4,double>& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		FINT *atm, FINT natm, FINT* bas, FINT nbas, double* env, int max_l);
		
void calc_ints_xx(dbcsr::mat_d& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		FINT *atm, FINT natm, FINT* bas, FINT nbas, double* env, int max_l);

void calc_ints_schwarz_mn(dbcsr::mat_d& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		FINT *atm, FINT natm, FINT* bas, FINT nbas, double* env, int max_l);
		
void calc_ints_schwarz_x(dbcsr::mat_d& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		FINT *atm, FINT natm, FINT* bas, FINT nbas, double* env, int max_l);

} // end namespace

#endif // USE_LIBCINT
#endif // INTS_INTEGRALS_H
