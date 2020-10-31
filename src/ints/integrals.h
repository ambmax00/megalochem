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
//                       LIBCINT
// =====================================================================

#ifdef USE_LIBCINT
extern "C" {
#include <cint.h>

int cint1e_ovlp_sph(double *out, const int *shls,
	const int *atm, int natm, const int *bas, int nbas, const double *env,
	const CINTOpt *opt);

int cint1e_nuc_sph(double *out, const int *shls,
	const int *atm, int natm, const int *bas, int nbas, const double *env,
	const CINTOpt *opt);

int cint1e_r_sph(double *out, const int *shls,
	const int *atm, int natm, const int *bas, int nbas, const double *env,
	const CINTOpt *opt);

int cint1e_kin_sph(double *out, const int *shls,
	const int *atm, int natm, const int *bas, int nbas, const double *env,
	const CINTOpt *opt);
	
int cint2c2e_sph(double *out, const int *shls,
	const int *atm, int natm, const int *bas, int nbas, const double *env,
	const CINTOpt *opt);
	
int cint3c2e_sph(double *out, const int *shls,
	const int *atm, int natm, const int *bas, int nbas, const double *env,
	const CINTOpt *opt);
	
int cint2e_coulerf_sph(double *out, const int *shls,
	const int *atm, int natm, const int *bas, int nbas, const double *env,
	const CINTOpt *opt);

int cint3c1e_sph(double *out, const int *shls,
	const int *atm, int natm, const int *bas, int nbas, const double *env,
	const CINTOpt *opt);
	
int cint4c1e_sph(double *out, const int *shls,
	const int *atm, int natm, const int *bas, int nbas, const double *env,
	const CINTOpt *opt);
	
void cint3c1e_sph_optimizer(CINTOpt **opt, const int *atm, const int natm,
    const int *bas, const int nbas, const double *env);
	
}

typedef int (*CINTIntegralFunction)(double *out, const int *shls,
		const int *atm, int natm, const int *bas, int nbas, const double *env,
		const CINTOpt *opt);
		
namespace ints {
		
void calc_ints(dbcsr::mat_d& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l);
		
void calc_ints(dbcsr::mat_d& m_x, dbcsr::mat_d& m_y, dbcsr::mat_d& m_z, 
		std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l);
		
void calc_ints(dbcsr::tensor<3,double>& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l);
		
void calc_ints(dbcsr::tensor<4,double>& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l);

void calc_ints_schwarz_mn(dbcsr::mat_d& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l);
		
void calc_ints_schwarz_x(dbcsr::mat_d& m_out, std::vector<std::vector<int>*> shell_offsets, 
		std::vector<std::vector<int>*> shell_sizes, CINTIntegralFunction& int_func,
		int *atm, int natm, int* bas, int nbas, double* env, int max_l);

} // end namespace

#endif // USE_LIBCINT
#endif // INTS_INTEGRALS_H
