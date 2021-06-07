#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/SparseCore>
#include <bitset>
#include "extern/lapack.hpp"
#include "ints/fitting.hpp"
#include "math/solvers/hermitian_eigen_solver.hpp"
#include "utils/constants.hpp"
#include "utils/scheduler.hpp"

//#define _USE_FLOAT
#define _USE_QR
#define _CHUNK_SIZE 2

namespace megalochem {

namespace ints {

static constexpr bool use_qr = true;
static constexpr int chunk_size = 2;
static constexpr double radius_eps = 1e-12;
static constexpr double radius_step = 0.2;
static constexpr int radius_max_step = 10000;

struct block_info {
  double alpha;
  double radius;
  std::array<double, 3> pos;
};

std::vector<block_info> get_block_info(desc::cluster_basis& cbas)
{
  std::vector<block_info> blkinfo(cbas.size());
  auto radii = cbas.radii(radius_eps, radius_step, radius_max_step);
  auto min_alphas = cbas.min_alpha();

  for (size_t ic = 0; ic != cbas.size(); ++ic) {
    blkinfo[ic].pos = cbas[ic].O;
    blkinfo[ic].radius = radii[ic];
    blkinfo[ic].alpha = min_alphas[ic];
  }

  return blkinfo;
}

dbcsr::sbtensor<3, double> dfitting::compute_qr_new(
    dbcsr::shared_matrix<double> s_bb,
    dbcsr::shared_matrix<double> s_xx_inv,
    dbcsr::shared_matrix<double> m_xx,
    dbcsr::shared_pgrid<3> spgrid3_xbb,
    std::array<int, 3> bdims,
    dbcsr::btype mytype)
{
  TIME.start();

  auto& prep_time = TIME.sub("Preparations for QRFIT");
  auto& work_time = TIME.sub("Worker function");
  auto& comp_time = TIME.sub("Compression");

  auto aofac = std::make_shared<aofactory>(m_mol, m_world);

  double qr_theta = global::qr_theta;
  double qr_rho = global::qr_rho;

  prep_time.start();

  auto x = m_mol->dims().x();
  auto b = m_mol->dims().b();

  auto xoff = m_xx->row_blk_offsets();
  auto boff = b;
  int off = 0;
  for (size_t i = 0; i != b.size(); ++i) {
    boff[i] = off;
    off += b[i];
  }

  arrvec<int, 2> xx = {x, x};
  arrvec<int, 3> xbb = {x, b, b};

  // =========== ALLOCATE TENSORS ====================================

  auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
  auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
  auto blktype_b = m_mol->c_basis()->shell_types();

  auto spgrid2_local = dbcsr::pgrid<2>::create(MPI_COMM_SELF).build();
  auto spgrid3_local = dbcsr::pgrid<3>::create(MPI_COMM_SELF).build();

  auto prs_xbb_local = dbcsr::tensor<3>::create()
                           .name("prs_xbb_local")
                           .set_pgrid(*spgrid3_local)
                           .blk_sizes(xbb)
                           .map1({0})
                           .map2({1, 2})
                           .build();

  auto c_xbb_local = dbcsr::tensor<3>::create()
                         .name("c_xbb_local")
                         .set_pgrid(*spgrid3_local)
                         .blk_sizes(xbb)
                         .map1({0})
                         .map2({1, 2})
                         .build();

  auto c_xbb_task = dbcsr::tensor<3>::create()
                        .name("c_xbb_task")
                        .set_pgrid(*spgrid3_local)
                        .blk_sizes(xbb)
                        .map1({0})
                        .map2({1, 2})
                        .build();

  auto ovlp_xbb_local = dbcsr::tensor<3>::create()
                            .name("eri_local")
                            .set_pgrid(*spgrid3_local)
                            .blk_sizes(xbb)
                            .map1({0})
                            .map2({1, 2})
                            .build();

  auto eri_local = dbcsr::tensor<3>::create()
                       .name("eri_local")
                       .set_pgrid(*spgrid3_local)
                       .blk_sizes(xbb)
                       .map1({0})
                       .map2({1, 2})
                       .build();

  auto s_xx_inv_local = dbcsr::tensor<2>::create()
                            .name("s_xx_inv_local")
                            .set_pgrid(*spgrid2_local)
                            .blk_sizes(xx)
                            .map1({0})
                            .map2({1})
                            .build();

  auto c_xbb_global = dbcsr::tensor<3>::create()
                          .name("c_xbb_global_1bb")
                          .set_pgrid(*spgrid3_xbb)
                          .blk_sizes(xbb)
                          .map1({0})
                          .map2({1, 2})
                          .build();

  arrvec<int, 3> blkmaps = {blkmap_x, blkmap_b, blkmap_b};

  auto c_xbb_batched = dbcsr::btensor<3>::create()
                           .name("cqr_xbb_batched")
                           .set_pgrid(spgrid3_xbb)
                           .blk_sizes(xbb)
                           .blk_maps(blkmaps)
                           .batch_dims(bdims)
                           .btensor_type(mytype)
                           .print(0)
                           .build();

  dbcsr::cart single_world(MPI_COMM_SELF);

  auto s_xx_inv_eigen = dbcsr::matrix_to_eigen(*s_xx_inv);
  auto s_xx_local_mat = dbcsr::eigen_to_matrix(
      s_xx_inv_eigen, single_world, "temp", x, x, dbcsr::type::symmetric);
  dbcsr::copy_matrix_to_tensor(*s_xx_local_mat, *s_xx_inv_local);
  s_xx_inv_local->filter(dbcsr::global::filter_eps);

  s_xx_inv_eigen.resize(0, 0);
  s_xx_local_mat->release();

  /* NOT FILTERED BECAUSE IT IS USED IN QR */
  auto m_xx_eigen = dbcsr::matrix_to_eigen(*m_xx);

  // =============== CREATE FRAGMENT BLOCKS ==========================

  auto is_diff = m_mol->c_basis()->diffuse();

  // make sure that each process has one atom block, but diffuse
  // and tight blocks are separated
  auto blkmap_frag = blkmap_b;
  int natoms = (int)m_mol->atoms().size();
  
  for (size_t i = 0; i != blkmap_b.size(); ++i) {
    blkmap_frag[i] = (!is_diff[i]) ? blkmap_frag[i] : blkmap_frag[i] + natoms;
  }

  for (auto ele : blkmap_frag) {
	  LOG.os<>(ele, " ");
  } LOG.os<>('\n');

  std::vector<std::vector<int>> frag_blocks;
  std::vector<int> frag_blk_sizes;

  std::vector<int> sub_block;
  int sub_size = 0;
  int prev_centre = -1;

  for (size_t i = 0; i != b.size(); ++i) {
    int current_centre = blkmap_frag[i];
    if (prev_centre != -1 && current_centre != prev_centre) {
      frag_blocks.push_back(sub_block);
      frag_blk_sizes.push_back(sub_size);
      sub_block.clear();
      sub_size = 0;
    }

    sub_block.push_back(i);
    sub_size += b[i];

    if (i == b.size() - 1) {
      frag_blocks.push_back(sub_block);
      frag_blk_sizes.push_back(sub_size);
    }

    prev_centre = current_centre;
  }

  for (size_t ifrag = 0; ifrag != frag_blocks.size(); ++ifrag) {
    LOG.os<1>("Fragment ", ifrag, " (size: ", frag_blk_sizes[ifrag], ")\n");
    for (auto ff : frag_blocks[ifrag]) { LOG.os<1>(ff, " "); }
    LOG.os<1>('\n');
  }

  int nbatches = c_xbb_batched->nbatches(2);
  std::vector<std::vector<int>> frag_bounds(nbatches);
  for (int inu = 0; inu != nbatches; ++inu) {
    auto nbds = c_xbb_batched->blk_bounds(2, inu);

    std::vector<int> blkbounds = {std::numeric_limits<int>::max(), 0};

    // we are guaranteed that each atom block is entirely in one
    // batch, so we just check the first function
    for (int i = 0; i != (int)frag_blocks.size(); ++i) {
      if (frag_blocks[i][0] < nbds[0] || frag_blocks[i][0] > nbds[1])
        continue;
      blkbounds[0] = std::min(blkbounds[0], i);
      blkbounds[1] = std::max(blkbounds[1], i);
    }

    frag_bounds[inu] = blkbounds;
  }

  LOG.os<1>("FRAG BLOCK BOUNDS: \n");
  for (auto bds : frag_bounds) {
    LOG.os<1>(bds[0],  " ", bds[1], '\n');
  }

  // ============== CREATE BLOCK INFO ================================

  auto b_basis = m_mol->c_basis();
  auto x_basis = m_mol->c_dfbasis();
  int nxblks = x.size();

  // get minimum exponent and position for each block
  auto blkinfo_x = get_block_info(*x_basis);
  auto blkinfo_b = get_block_info(*b_basis);

#if 0
	if (m_cart.rank() == 0) {
	std::cout << "X BASIS INFO: " << std::endl;
	for (auto s : blkinfo_x) {
		std::cout << "EXP: " << s.alpha << std::endl;
		std::cout << "RADIUS: " << s.radius << std::endl;
		std::cout << "POS: " << s.pos[0] << " " << s.pos[1]
			<< " " << s.pos[2] << std::endl;
	}
	}
#endif

  // ===================== OTHER STUFF ==============================

  auto dist = [](std::array<double, 3>& p1, std::array<double, 3>& p2) {
    return sqrt(
        pow(p1[0] - p2[0], 2.0) + pow(p1[1] - p2[1], 2.0) +
        pow(p1[2] - p2[2], 2.0));
  };

  auto blknorms = dbcsr::block_norms(*s_bb);

  // ================== LOOP OVER NU BATCHES ==============

  c_xbb_batched->compress_init({2}, vec<int>{0}, vec<int>{1, 2});

  prep_time.finish();

  std::vector<int> blkprev;
  int ncounter = 0;

  for (int ibatch_nu = 0; ibatch_nu != c_xbb_batched->nbatches(2);
       ++ibatch_nu) {
    LOG.os<>("BATCH: ", ibatch_nu, '\n');
    LOG.os<>("Creating tasks\n");

    auto nu_blkbounds = c_xbb_batched->blk_bounds(2, ibatch_nu);
    auto frag_blkbounds = frag_bounds[ibatch_nu];

    // =================== CREATE TASKS ============================

    using task_list = std::vector<std::vector<std::pair<int, int>>>;

    task_list global_tasks;

    std::vector<std::pair<int, int>> chunk;

    for (int ifrag = 0; ifrag != (int)frag_blocks.size(); ++ifrag) {
      for (int jfrag = frag_blkbounds[0]; jfrag != frag_blkbounds[1] + 1;
           ++jfrag) {
        if (ifrag > jfrag)
          continue;

        std::pair<int, int> p = {ifrag, jfrag};

        chunk.push_back(p);
        if (chunk.size() >= chunk_size) {
          global_tasks.push_back(chunk);
          chunk.clear();
        }
      }
    }

    if (chunk.size() != 0)
      global_tasks.push_back(chunk);

    /*LOG.os<>("GLOBAL TASKS: \n");
    for (int itask = 0; itask != global_tasks.size(); ++itask) {
            LOG.os<>("TASK: ", itask, '\n');
            for (auto c : global_tasks[itask]) {
                    LOG.os<>(c.first, " ", c.second, '\n');
            }
    }

    std::cout << "NTASKS: " << global_tasks.size() << std::endl;
    ntasks += global_tasks.size();*/

    /* =============================================================
     *            TASK FUNCTION FOR SCHEDULER
     * ============================================================*/

    std::deque<std::vector<int>> blkbuffer;

    std::function<void(int64_t)> task_func = [&](int64_t itask) {
      // std::cout << "PROC: " << m_cart.rank() << " -> TASK ID: " << itask <<
      // std::endl;

      // === COMPUTE 3c1e overlap integrals
      // 1. allocate blocks

      auto& ovlp_time = work_time.sub("Computing ovlp integrals");
      auto& coul_time = work_time.sub("Computing coul integrals");
      auto& dgels_time = work_time.sub("DGELS");
      auto& setup_time = work_time.sub("Setting up");
      auto& move_time = work_time.sub("Moving data");

      work_time.start();

      arrvec<int, 3> blkidx, blkidx_full;

      for (auto chunk : global_tasks[itask]) {
        int ifrag = chunk.first;
        int jfrag = chunk.second;

        for (auto imu : frag_blocks[ifrag]) {
          for (auto inu : frag_blocks[jfrag]) {
            bool is_same_type = (blktype_b[imu] == blktype_b[inu]);

            if (is_same_type &&
                blknorms(imu, inu) < dbcsr::global::filter_eps) {
              continue;
            }

            for (size_t ix = 0; ix != x.size(); ++ix) {
              blkidx[0].push_back(ix);
              blkidx[1].push_back(imu);
              blkidx[2].push_back(inu);
            }
          }
        }
      }

      // 2. Compute

      ovlp_time.start();

      ovlp_xbb_local->reserve(blkidx);

      aofac->ao_3c1e_ovlp_setup();
      aofac->ao_3c_fill(ovlp_xbb_local);
      ovlp_xbb_local->filter(dbcsr::global::filter_eps);

      vec<vec<int>> mn_bounds = {
          c_xbb_batched->full_bounds(1), c_xbb_batched->bounds(2, ibatch_nu)};

      // 3. Contract

      if (ovlp_xbb_local->num_blocks() == 0) {
        std::cout << "RANK: " << m_cart.rank() << "COMPLETE SCREEN"
                  << std::endl;
        ovlp_time.finish();
        work_time.finish();
        return;
      }

      dbcsr::contract(
          1.0, *s_xx_inv_local, *ovlp_xbb_local, 0.0, *prs_xbb_local)
          .filter(qr_theta)
          .bounds3(mn_bounds)
          .perform("XY, Ymn -> Xmn");

      ovlp_xbb_local->clear();

      if (prs_xbb_local->num_blocks() == 0) {
        std::cout << "RANK: " << m_cart.rank() << "COMPLETE SCREEN"
                  << std::endl;
        ovlp_time.finish();
        work_time.finish();
        return;
      }

      ovlp_time.finish();

      // === LOOP OVER CHUNKS OF FRAGMENTS

      for (auto chunk : global_tasks[itask]) {
        setup_time.start();

        int ifrag = chunk.first;
        int jfrag = chunk.second;

        // std::cout << "CHUNK: " << ifrag << " " << jfrag << std::endl;

        auto blk_mu = frag_blocks[ifrag];
        auto blk_nu = frag_blocks[jfrag];

        // std::cout << "SIZE: " << blk_mu.size() << " " << blk_nu.size()
        //	<< std::endl;

        vec<bool> blk_P_bool(x.size(), false);

        std::array<int, 3> idx3 = {0, 0, 0};
        std::array<int, 3> size3 = {0, 0, 0};

        for (auto imu : blk_mu) {
          for (auto inu : blk_nu) {
            for (int ix = 0; ix != (int)x.size(); ++ix) {
              idx3 = {ix, imu, inu};
              size3 = {x[ix], b[imu], b[inu]};
              bool found = true;

              auto blk3 = prs_xbb_local->get_block(idx3, size3, found);
              if (!found)
                continue;

              auto max_iter = std::max_element(
                  blk3.data(), blk3.data() + blk3.ntot(),
                  [](double a, double b) {
                    return fabs(a) < fabs(b);
                  });

              if (fabs(*max_iter) > qr_theta)
                blk_P_bool[ix] = true;
            }
          }
        }

        // prs_xbb_local->clear();

        vec<int> blk_P;
        blk_P.reserve(x.size());
        for (int ix = 0; ix != (int)x.size(); ++ix) {
          if (blk_P_bool[ix])
            blk_P.push_back(ix);
        }

        // check if in buffer
        auto blkbuffer_it =
            std::find(blkbuffer.begin(), blkbuffer.end(), blk_P);
        if (blkbuffer_it != blkbuffer.end()) {
          ncounter++;
        }

        if (blkbuffer.size() > 10) {
          blkbuffer.pop_front();
        }

        blkbuffer.push_back(blk_P);

        int nblkp = blk_P.size();
        // std::cout << "FUNCS: " << nblkp << "/" << x.size() << std::endl;

        if (nblkp == 0) {
          setup_time.finish();
          std::cout << "RANK: " << m_cart.rank() << "QRRHO SCREEN" << std::endl;
          continue;
        }

        // ==== Get all Q functions ====
        std::vector<bool> blk_Q_bool(
            x.size(), false);  // whether block x is involved

        for (int ix = 0; ix != nxblks; ++ix) {
          for (auto ip : blk_P) {
            auto& pos_x = blkinfo_x[ix].pos;
            auto& pos_p = blkinfo_x[ip].pos;
            double alpha_x = blkinfo_x[ix].alpha;
            double alpha_p = blkinfo_x[ip].alpha;

            double f = (alpha_x * alpha_p) / (alpha_x + alpha_p) *
                pow(dist(pos_x, pos_p), 2.0);

            if (f < qr_rho)
              blk_Q_bool[ix] = true;
          }
        }

        std::vector<int> blk_Q;
        blk_Q.reserve(x.size());
        for (int ix = 0; ix != (int)x.size(); ++ix) {
          if (blk_Q_bool[ix])
            blk_Q.push_back(ix);
        }

        // === Compute coulomb integrals
        arrvec<int, 3> coul_blk_idx;
        coul_blk_idx[0] = blk_Q;
        coul_blk_idx[1] = blk_mu;
        coul_blk_idx[2] = blk_nu;

        int nb = 0;  // number of total nb functions
        int mstride = 0;
        int nstride = 0;

        for (auto imu : blk_mu) { mstride += b[imu]; }

        for (auto inu : blk_nu) { nstride += b[inu]; }

        nb = nstride * mstride;

        setup_time.finish();

        coul_time.start();

        // generate integrals
        aofac->ao_3c2e_setup(metric::coulomb);
        aofac->ao_3c_fill_idx(eri_local, coul_blk_idx, nullptr);
        /* NOT FILTERED BECAUSE IT IS USED IN QR */

        coul_time.finish();

        // dbcsr::print(*eri_local);

        // how many x functions?
        int nq = 0;
        int np = 0;

        for (auto iq : blk_Q) { nq += x[iq]; }

        for (auto ip : blk_P) { np += x[ip]; }

        std::cout << "RANK: " << m_cart.rank() << " NP,NQ,NB: " << np << "/"
                  << nq << "/" << nb << std::endl;

        // ==== Prepare matrices for QR decomposition ====

        move_time.start();

#ifdef _USE_FLOAT
        Eigen::MatrixXf eris_eigen = Eigen::MatrixXf::Zero(nq, nb);
        Eigen::MatrixXf m_qp_eigen = Eigen::MatrixXf::Zero(nq, np);
#else
        Eigen::MatrixXd eris_eigen = Eigen::MatrixXd::Zero(nq, nb);
        Eigen::MatrixXd m_qp_eigen = Eigen::MatrixXd::Zero(nq, np);
#endif

        int poff = 0;
        int qoff = 0;
        int moff = 0;
        int noff = 0;

        // Copy integrals to matrix
        for (auto iq : blk_Q) {
          for (auto imu : blk_mu) {
            for (auto inu : blk_nu) {
              std::array<int, 3> idx = {iq, imu, inu};
              std::array<int, 3> size = {x[iq], b[imu], b[inu]};
              bool found = true;
              auto blk = eri_local->get_block(idx, size, found);
              if (!found)
                continue;

              for (int qq = 0; qq != size[0]; ++qq) {
                for (int mm = 0; mm != size[1]; ++mm) {
                  for (int nn = 0; nn != size[2]; ++nn) {
                    eris_eigen(qq + qoff, mm + moff + (nn + noff) * mstride) =
                        blk(qq, mm, nn);
                  }
                }
              }
              noff += b[inu];
            }
            noff = 0;
            moff += b[imu];
          }
          moff = 0;
          qoff += x[iq];
        }

        // copy metric to eigen
        for (auto ip : blk_P) {
          int sizep = x[ip];
          int poff_m = xoff[ip];

          qoff = 0;

          for (auto iq : blk_Q) {
            int sizeq = x[iq];
            int qoff_m = xoff[iq];

            for (int qq = 0; qq != sizeq; ++qq) {
              for (int pp = 0; pp != sizep; ++pp) {
                m_qp_eigen(qq + qoff, pp + poff) =
                    m_xx_eigen(qq + qoff_m, pp + poff_m);
              }
            }

            qoff += sizeq;
          }
          poff += sizep;
        }

        eri_local->clear();

        // ===== Compute QR decomposition ====

        move_time.finish();

        dgels_time.start();

/*
int info = 0;
int MN = std::min(nq,np);
int lwork = std::max(1, MN + std::max(MN, nb)) * 2;
double* work = new double[lwork];

c_dgels('N', nq, np, nb, m_qp_eigen.data(), nq, eris_eigen.data(),
        nq, work, lwork, &info);

delete[] work;

*/
#ifdef _USE_FLOAT
        Eigen::MatrixXf c_eigen;
#else
        Eigen::MatrixXd c_eigen;
#endif

#ifdef _USE_QR
        c_eigen = m_qp_eigen.householderQr().solve(eris_eigen);
#else
        c_eigen = (m_qp_eigen.transpose() * m_qp_eigen)
                      .ldlt()
                      .solve(m_qp_eigen.transpose() * eris_eigen);
#endif

        dgels_time.finish();

        move_time.start();

        // ==== Transfer fitting coefficients to tensor

        // transfer it to c_xbb
        arrvec<int, 3> cfit_idx;

        for (auto ip : blk_P) {
          for (auto imu : blk_mu) {
            for (auto inu : blk_nu) {
              cfit_idx[0].push_back(ip);
              cfit_idx[1].push_back(imu);
              cfit_idx[2].push_back(inu);
            }
          }
        }

        c_xbb_task->reserve(cfit_idx);

        poff = 0;
        moff = 0;
        noff = 0;

        for (auto ip : blk_P) {
          for (auto imu : blk_mu) {
            for (auto inu : blk_nu) {
              std::array<int, 3> idx = {ip, imu, inu};
              std::array<int, 3> size = {x[ip], b[imu], b[inu]};

              dbcsr::block<3, double> blk(size);
              for (int pp = 0; pp != size[0]; ++pp) {
                for (int mm = 0; mm != size[1]; ++mm) {
                  for (int nn = 0; nn != size[2]; ++nn) {
                    blk(pp, mm, nn) =
                        c_eigen(pp + poff, mm + moff + (nn + noff) * mstride);
                  }
                }
              }
              c_xbb_task->put_block(idx, blk);
              noff += b[inu];
            }
            noff = 0;
            moff += b[imu];
          }
          moff = 0;
          poff += x[ip];
        }

        c_xbb_task->filter(dbcsr::global::filter_eps);
        if (ifrag == jfrag)
          c_xbb_task->scale(0.5);

        dbcsr::copy(*c_xbb_task, *c_xbb_local)
            .move_data(true)
            .sum(true)
            .perform();

        move_time.finish();

      }  // end loop over chunks

      prs_xbb_local->clear();

      work_time.finish();
    };  // end task function

    util::basic_scheduler tasks(m_cart.comm(), global_tasks.size(), task_func);

    tasks.run();

    comp_time.start();

    dbcsr::copy_local_to_global(*c_xbb_local, *c_xbb_global);

    c_xbb_local->clear();

    c_xbb_batched->compress({ibatch_nu}, c_xbb_global);
    c_xbb_global->clear();

    comp_time.finish();

  }  // end loop over batches

  c_xbb_batched->compress_finalize();

  auto& time_desym = TIME.sub("Desymmetrizing tensor");
  time_desym.start();

  auto c_xbb_batched_full = dbcsr::btensor<3>::create()
                                .name("cqr_xbb_batched_full")
                                .set_pgrid(spgrid3_xbb)
                                .blk_sizes(xbb)
                                .blk_maps(blkmaps)
                                .batch_dims(bdims)
                                .btensor_type(mytype)
                                .print(0)
                                .build();

  auto c_xbb_n = c_xbb_batched->get_template("c_xbb_n", {0}, {1, 2});
  auto c_xbb_t = c_xbb_batched->get_template("c_xbb_t", {0}, {1, 2});

  c_xbb_batched_full->compress_init({0}, {0}, {1, 2});
  c_xbb_batched->decompress_init({0}, {0}, {1, 2});

  for (int ix = 0; ix != c_xbb_batched->nbatches(0); ++ix) {
    c_xbb_batched->decompress({ix});
    auto c_xbb = c_xbb_batched->get_work_tensor();

    vec<vec<int>> xbounds = {
        c_xbb_batched->bounds(0, ix), c_xbb_batched->full_bounds(1),
        c_xbb_batched->full_bounds(2)};

    dbcsr::copy(*c_xbb, *c_xbb_n).bounds(xbounds).perform();

    dbcsr::copy(*c_xbb, *c_xbb_t).bounds(xbounds).order({0, 2, 1}).perform();

    dbcsr::copy(*c_xbb_t, *c_xbb_n)
        .bounds(xbounds)
        .sum(true)
        .move_data(true)
        .perform();

    c_xbb_batched_full->compress({ix}, c_xbb_n);
  }

  c_xbb_batched->decompress_finalize();
  c_xbb_batched_full->compress_finalize();

  time_desym.finish();

  double occupation = c_xbb_batched_full->occupation() * 100;

  if (occupation > 100)
    throw std::runtime_error("Fitting coefficients occupation more than 100%");

  TIME.finish();
  TIME.print_info();

  return c_xbb_batched_full;
}

// for C_Xμν computes sparsity between X and ν
std::shared_ptr<Eigen::MatrixXi> dfitting::compute_idx(
    dbcsr::sbtensor<3, double> cfit_xbb)
{
  auto x = m_mol->dims().x();
  auto b = m_mol->dims().b();

  Eigen::MatrixXi idx_local = Eigen::MatrixXi::Zero(x.size(), b.size());
  Eigen::MatrixXi idx_global = Eigen::MatrixXi::Zero(x.size(), b.size());

  cfit_xbb->decompress_init({2}, vec<int>{0}, vec<int>{1, 2});
  int nbatches =
      cfit_xbb->nbatches(2);  // (cfit_xbb->get_type() == dbcsr::btype::core) ?
                              // 1 : cfit_xbb->nbatches(2);

  double tresh = 1e-6;

  for (int inu = 0; inu != nbatches; ++inu) {
    cfit_xbb->decompress({inu});
    auto cfit = cfit_xbb->get_work_tensor();

    dbcsr::iterator_t<3> iter(*cfit);
    iter.start();

    while (iter.blocks_left()) {
      iter.next();
      auto& idx = iter.idx();
      auto& size = iter.size();

      int ix = idx[0];
      int inu = idx[2];

      bool found = true;
      auto blk = cfit->get_block(idx, size, found);
      double maxele = 0.0;

      for (int i = 0; i != blk.ntot(); ++i) {
        maxele = std::max(maxele, fabs(blk.data()[i]));
      }

      idx_local(ix, inu) = (maxele > tresh) ? 1 : idx_local(ix, inu);
    }

    iter.stop();
  }

  cfit_xbb->decompress_finalize();

  MPI_Allreduce(
      idx_local.data(), idx_global.data(), x.size() * b.size(), MPI_INT,
      MPI_LOR, m_cart.comm());

  int nblks = 0;

  for (int ix = 0; ix != (int)x.size(); ++ix) {
    for (int inu = 0; inu != (int)b.size(); ++inu) {
      if (idx_global(ix, inu))
        ++nblks;
    }
  }

  LOG.os<>("BLOCKS: ", nblks, "/", x.size() * b.size(), '\n');

  auto out_ptr = std::make_shared<Eigen::MatrixXi>(std::move(idx_global));
  return out_ptr;
}

}  // namespace ints

}  // namespace megalochem
