#ifndef INTS_INTEGRALS_H
#define INTS_INTEGRALS_H

#include <mpi.h>
#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>
#include <vector>
#include "desc/basis.hpp"
#include "ints/screening.hpp"

// =====================================================================
//                       LIBCINT
// =====================================================================

extern "C" {
#include <cint.h>
#include <cint_funcs.h>

CINTIntegralFunction int3c2e_sph;

CINTIntegralFunction int3c1e_sph;
#ifdef WITH_4C1E
CINTIntegralFunction int4c1e_sph;
#endif
CINTIntegralFunction int2c2e_sph;
}

namespace megalochem {

namespace ints {

typedef void (*GeneralFunctionPtr)(void*);

void calc_ints(
    dbcsr::matrix<double>& m_out,
    std::vector<std::vector<int>>& shell_offsets,
    std::vector<std::vector<int>>& shell_sizes,
    CINTIntegralFunction* int_func,
    int* atm,
    int natm,
    int* bas,
    int nbas,
    double* env,
    int max_l);

void calc_ints(
    dbcsr::matrix<double>& m_x,
    dbcsr::matrix<double>& m_y,
    dbcsr::matrix<double>& m_z,
    std::vector<std::vector<int>>& shell_offsets,
    std::vector<std::vector<int>>& shell_sizes,
    CINTIntegralFunction* int_func,
    int* atm,
    int natm,
    int* bas,
    int nbas,
    double* env,
    int max_l);

void calc_ints(
    dbcsr::tensor<3, double>& m_out,
    std::vector<std::vector<int>>& shell_offsets,
    std::vector<std::vector<int>>& shell_sizes,
    CINTIntegralFunction* nt_func,
    int* atm,
    int natm,
    int* bas,
    int nbas,
    double* env,
    int max_l);

void calc_ints(
    dbcsr::tensor<4, double>& m_out,
    std::vector<std::vector<int>>& shell_offsets,
    std::vector<std::vector<int>>& shell_sizes,
    CINTIntegralFunction* int_func,
    int* atm,
    int natm,
    int* bas,
    int nbas,
    double* env,
    int max_l);

void calc_ints_schwarz_mn(
    dbcsr::matrix<double>& m_out,
    std::vector<std::vector<int>>& shell_offsets,
    std::vector<std::vector<int>>& shell_sizes,
    CINTIntegralFunction* int_func,
    int* atm,
    int natm,
    int* bas,
    int nbas,
    double* env,
    int max_l);

void calc_ints_schwarz_x(
    dbcsr::matrix<double>& m_out,
    std::vector<std::vector<int>>& shell_offsets,
    std::vector<std::vector<int>>& shell_sizes,
    CINTIntegralFunction* int_func,
    int* atm,
    int natm,
    int* bas,
    int nbas,
    double* env,
    int max_l);

}  // namespace ints

}  // namespace megalochem

#endif  // INTS_INTEGRALS_H
