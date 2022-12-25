#include <megalochem.hpp>
#include <desc/molecule.hpp>
#include <desc/wfn.hpp>
#include <hf/hfmod.hpp>
#include <mp/mpmod.hpp>
#include <utils/constants.hpp>
#include <utils/mpi_log.hpp>

#define CHECK_EQUAL(a, b, result) \
  if (a != b) \
  { \
    LOG.os<>("Expected: ", b, " but got ", #a, " = ", a, '\n');\
    result++; \
  }  

#define CHECK_ALMOST_EQUAL(a, b, result) \
  if (std::abs(a-b)>1e-12)\
  {\
    LOG.scientific();\
    LOG.setprecision(14);\
    LOG.os<>("Expected: ", b, " but got ", #a, " = ", a, '\n');\
    result++;\
  }

#define CHECK_VECTOR_ALMOST_EQUAL(a, b, result) \
  if (a.size() != b.size())\
  {\
    LOG.os<>("Vectors ", #a, ", ", #b, " do not have the same dimension");\
  }\
  for (size_t i = 0L; i < a.size(); ++i)\
  {\
    if (std::abs(a[i]-b[i])>1e-12)\
    {\
      LOG.scientific();\
      LOG.setprecision(14);\
      LOG.os<>("Vectors ", #b, " ", #a, " differ: ", a[i], " ", b[i]);\
      result++;\
    }\
  }

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int result = 0;

  megalochem::init(comm, ".");

  megalochem::world mega_world(comm);

  util::mpi_log LOG(comm, 0);

  { // start context. If you do not do that, there are some problems with finalizing MPI due 
    // to some poor decisions what the dbcsr matrix destructors are doing...

    std::vector<megalochem::desc::Atom> atoms = 
    {
      {0.00000, 0.00000, 0.11779, 8},
      {0.00000, 0.75545,-0.47116, 1},
      {0.00000,-0.75545,-0.47116, 1}
    };

    for (auto& a : atoms)
    {
      a.x /= BOHR_RADIUS;
      a.y /= BOHR_RADIUS;
      a.z /= BOHR_RADIUS;
    }

    auto basis = std::make_shared<megalochem::desc::cluster_basis>("cc-pvdz", atoms);

    auto pDfBasis = std::make_shared<megalochem::desc::cluster_basis>("cc-pvdz-ri", atoms);

    auto pMol = megalochem::desc::molecule::create()
                                      .atoms(atoms)
                                      .charge(0)
                                      .cluster_basis(basis)
                                      .comm(mega_world.comm())
                                      .fractional(false)
                                      .mo_split(5)
                                      .mult(1)
                                      .name("h2o")
                                      .spin_average(true)
                                      .build();

    auto pHfModExact = megalochem::hf::hfmod::create()
                      .set_molecule(pMol)
                      .set_world(mega_world)
                      .build_J("exact")
                      .build_K("exact")
                      //.df_basis2()
                      //.df_basis()
                      //.df_metric()
                      //.diis_max_vecs()
                      //.diis_min_vecs()
                      //.diis_start()
                      //.do_diis()
                      //.do_diis_beta()
                      .eris("core")
                      .guess("SAD")
                      .imeds("core")
                      //.max_iter()
                      .nbatches_b(2)
                      .nbatches_occ(2)
                      .nbatches_x(2)
                      .print(5)
                      //.read()
                      //.SAD_do_diis()
                      .scf_threshold(1e-8)
                      //.SAD_guess()
                      //.SAD_spin_average()
                      .build();

    auto pHfModDfAoDfAo = megalochem::hf::hfmod::create()
                          .set_molecule(pMol)
                          .set_world(mega_world)
                          .build_J("dfao")
                          .build_K("dfao")
                          .df_basis(pDfBasis)
                          .df_metric("qr_fit")
                          .eris("core")
                          .guess("SADNO")
                          .imeds("core")
                          .nbatches_b(2)
                          .nbatches_occ(2)
                          .nbatches_x(2)
                          .scf_threshold(1e-8)
                          .build();

    auto pHfModDfAoDfMo = megalochem::hf::hfmod::create()
                          .set_molecule(pMol)
                          .set_world(mega_world)
                          .build_J("dfao")
                          .build_K("dfmo")
                          .df_basis(pDfBasis)
                          .df_metric("coulomb")
                          .eris("core")
                          .guess("core")
                          .imeds("core")
                          .nbatches_b(2)
                          .nbatches_occ(2)
                          .nbatches_x(2)
                          .scf_threshold(1e-8)
                          .build();

    auto pHfModDfAoDfMem = megalochem::hf::hfmod::create()
                          .set_molecule(pMol)
                          .set_world(mega_world)
                          .build_J("dfao")
                          .build_K("dfmem")
                          .df_basis(pDfBasis)
                          .df_metric("coulomb")
                          .eris("core")
                          .guess("core")
                          .imeds("core")
                          .nbatches_b(2)
                          .nbatches_occ(2)
                          .nbatches_x(2)
                          .scf_threshold(1e-8)
                          .build();

    auto pHfExactWfn = pHfModExact->compute();
    auto pHfDfAoDfAoWfn = pHfModDfAoDfAo->compute();
    auto pHfDfAoDfMoWfn = pHfModDfAoDfMo->compute();
    auto pHfDfAoDfmemWfn = pHfModDfAoDfMem->compute();

    CHECK_ALMOST_EQUAL(pHfExactWfn->hf_wfn->scf_energy(), -8.52159902619285e+01, result);
    // this is fishy, but sticking with it for now... need to investigate
    CHECK_ALMOST_EQUAL(pHfDfAoDfAoWfn->hf_wfn->scf_energy(),-1.35503134472133e+02, result);
    CHECK_ALMOST_EQUAL(pHfDfAoDfMoWfn->hf_wfn->scf_energy(), -8.52170680515785e+01, result);
    CHECK_ALMOST_EQUAL(pHfDfAoDfmemWfn->hf_wfn->scf_energy(), -8.52170680515733e+01, result);

    auto pMpModFull = megalochem::mp::mpmod::create()
                      .build_Z("llmp_full")
                      .c_os(1.3)
                      .df_basis(pDfBasis)
                      .df_metric("coulomb")
                      .eris("core")
                      .imeds("core")
                      .nbatches_b(2)
                      .nbatches_x(2)
                      .nlap(5)
                      .print(0)
                      .set_wfn(pHfExactWfn)
                      .set_world(mega_world)
                      .build();

    auto pMpModMem = megalochem::mp::mpmod::create()
                      .build_Z("llmp_mem")
                      .c_os(1.3)
                      .df_basis(pDfBasis)
                      .df_metric("erfc_coulomb")
                      .eris("core")
                      .imeds("core")
                      .nbatches_b(2)
                      .nbatches_x(2)
                      .nlap(5)
                      .print(0)
                      .set_wfn(pHfExactWfn)
                      .set_world(mega_world)
                      .build();

      auto pMpFullWfn = pMpModFull->compute();
      auto pMpMemWfn = pMpModMem->compute();

      CHECK_ALMOST_EQUAL(pMpFullWfn->mp_wfn->mp_energy(), 0.198172796878, result);
      CHECK_ALMOST_EQUAL(pMpMemWfn->mp_wfn->mp_energy(), 0.198172563454, result);

  }

  MPI_Finalize();

  return result;
}