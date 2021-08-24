#ifndef DBCSR_CONVERSIONS_HPP
#define DBCSR_CONVERSIONS_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <dbcsr_matrix.hpp>
#include <dbcsr_tensor.hpp>
#include <limits>
#include "extern/scalapack.hpp"

template <typename T, int StorageOrder>
using MatrixX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>;

namespace dbcsr {

template <typename T = double, int StorageOrder = Eigen::ColMajor>
MatrixX<T, StorageOrder> matrix_to_eigen(matrix<T>& mat_in)
{
  int prow = -1;
  int pcol = -1;

  auto mat_copy =
      dbcsr::matrix<T>::copy(mat_in).name("Copy of " + mat_in.name()).build();
  decltype(mat_copy) mat_desym;

  if (mat_copy->has_symmetry()) {
    mat_desym = mat_copy->desymmetrize();
  }
  else {
    mat_desym = mat_copy;
  }

  if (prow < 0 || pcol < 0) {
    // replicate on all ranks

    int row = mat_in.nfullrows_total();
    int col = mat_in.nfullcols_total();

    mat_desym->replicate_all();

    MatrixX<T, StorageOrder> eigenmat =
        MatrixX<T, StorageOrder>::Zero(row, col);

#pragma omp parallel
    {
      iterator<T> iter(*mat_desym);

      iter.start();

      while (iter.blocks_left()) {
        iter.next_block();
        for (int j = 0; j != iter.col_size(); ++j) {
          for (int i = 0; i != iter.row_size(); ++i) {
            eigenmat(i + iter.row_offset(), j + iter.col_offset()) = iter(i, j);
          }
        }
      }

      iter.stop();
      mat_desym->finalize();
    }

    return eigenmat;
  }
  else {
    // eigen matrix only on one rank
    auto w = mat_in.get_cart();

    int nblkrow = mat_in.nblkrows_total();
    int nblkcol = mat_in.nblkcols_total();

    std::vector<int> rowdist(nblkrow, prow);
    std::vector<int> coldist(nblkcol, pcol);

    auto locdist =
        dist::create().set_cart(w).row_dist(rowdist).col_dist(coldist).build();

    auto rblksizes = mat_in.row_blk_sizes();
    auto cblksizes = mat_in.col_blk_sizes();

    auto locmat = matrix<T>::create()
                      .set_dist(*locdist)
                      .name("locmat")
                      .row_blk_sizes(rblksizes)
                      .col_blk_sizes(cblksizes)
                      .matrix_type(dbcsr::type::no_symmetry)
                      .build();

    locmat->complete_redistribute(*mat_desym);
    MatrixX<T, StorageOrder>* eigenmat;

    if ((prow == w.myprow()) && (pcol == w.mypcol())) {
      int nrows = mat_in.nfullrows_total();
      int ncols = mat_in.nfullcols_total();

      eigenmat = new MatrixX<T, StorageOrder>(
          MatrixX<T, StorageOrder>::Zero(nrows, ncols));

#pragma omp parallel
      {
        iterator<T> iter(*locmat);

        iter.start();

        while (iter.blocks_left()) {
          iter.next_block();
          for (int j = 0; j != iter.col_size(); ++j) {
            for (int i = 0; i != iter.row_size(); ++i) {
              (*eigenmat)(i + iter.row_offset(), j + iter.col_offset()) =
                  iter(i, j);
            }
          }
        }

        iter.stop();
        locmat->finalize();
      }
    }
    else {
      eigenmat =
          new MatrixX<T, StorageOrder>(MatrixX<T, StorageOrder>::Zero(0, 0));
    }

    return *eigenmat;
  }
}

template <typename Derived, typename T = double>
shared_matrix<typename Derived::Scalar> eigen_to_matrix(
    const Eigen::MatrixBase<Derived>& mat,
    cart w,
    std::string name,
    vec<int>& row_blk_sizes,
    vec<int>& col_blk_sizes,
    type mtype)
{
  int prow = -1;
  int pcol = -1;

  if (prow < 0 || pcol < 0) {
    auto out = matrix<T>::create()
                   .name(name)
                   .set_cart(w)
                   .row_blk_sizes(row_blk_sizes)
                   .col_blk_sizes(col_blk_sizes)
                   .matrix_type(mtype)
                   .build();

    if (out->has_symmetry()) {
      out->reserve_sym();
    }
    else {
      out->reserve_all();
    }

#pragma omp parallel
    {
      iterator<T> iter(*out);
      iter.start();

      while (iter.blocks_left()) {
        iter.next_block();

        for (int j = 0; j != iter.col_size(); ++j) {
          for (int i = 0; i != iter.row_size(); ++i) {
            iter(i, j) = mat(i + iter.row_offset(), j + iter.col_offset());
          }
        }
      }
      iter.stop();
      out->finalize();
    }

    return out;
  }
  else {
    std::vector<int> rowdist(row_blk_sizes.size(), prow),
        coldist(col_blk_sizes.size(), pcol);

    auto locdist =
        dist::create().set_cart(w).row_dist(rowdist).col_dist(coldist).build();

    auto locmat = matrix<T>::create()
                      .name(name)
                      .set_dist(*locdist)
                      .row_blk_sizes(row_blk_sizes)
                      .col_blk_sizes(col_blk_sizes)
                      .matrix_type(mtype)
                      .build();

    auto out = matrix<T>::create()
                   .name(name)
                   .set_cart(w)
                   .row_blk_sizes(row_blk_sizes)
                   .col_blk_sizes(col_blk_sizes)
                   .matrix_type(mtype)
                   .build();

    if (locmat->has_symmetry()) {
      locmat->reserve_sym();
    }
    else {
      locmat->reserve_all();
    }

    if (prow == w.myprow() && pcol == w.mypcol()) {
#pragma omp parallel
      {
        iterator<T> iter(*locmat);
        iter.start();

        while (iter.blocks_left()) {
          iter.next_block();

          for (int j = 0; j != iter.col_size(); ++j) {
            for (int i = 0; i != iter.row_size(); ++i) {
              iter(i, j) = mat(i + iter.row_offset(), j + iter.col_offset());
            }
          }
        }
        iter.stop();
        locmat->finalize();
      }
    }

    out->complete_redistribute(*locmat);

    return out;
  }
}

//#ifdef USE_SCALAPACK

template <typename T = double>
scalapack::distmat<T> matrix_to_scalapack(
    shared_matrix<T> mat_in,
    scalapack::grid sgrid,
    int nsplitrow,
    int nsplitcol,
    int ori_row,
    int ori_col)
{
  cart mcart = mat_in->get_cart();

  int nrows = mat_in->nfullrows_total();
  int ncols = mat_in->nfullcols_total();

  // make distvecs
  vec<int> rowsizes = split_range(nrows, nsplitrow);
  vec<int> colsizes = split_range(ncols, nsplitcol);

  vec<int> rowdist = cyclic_dist(rowsizes.size(), mcart.dims()[0]);
  vec<int> coldist = cyclic_dist(colsizes.size(), mcart.dims()[1]);

  auto scaldist = dist::create()
                      .set_cart(mcart)
                      .row_dist(rowdist)
                      .col_dist(coldist)
                      .build();

  shared_matrix<T> mat_out = matrix<T>::create()
                                 .name(mat_in->name() + " redist.")
                                 .set_dist(*scaldist)
                                 .row_blk_sizes(rowsizes)
                                 .col_blk_sizes(colsizes)
                                 .matrix_type(type::no_symmetry)
                                 .build();

  if (mat_in->has_symmetry()) {
    auto mat_in_nosym = mat_in->desymmetrize();
    mat_out->complete_redistribute(*mat_in_nosym, false);
    // mat_in_nosym.release();
  }
  else {
    mat_out->complete_redistribute(*mat_in, false);
  }

  // dbcsr::print(mat_out);
  scalapack::distmat<double> scamat(
      sgrid, nrows, ncols, nsplitrow, nsplitcol, ori_row, ori_col);

#pragma omp parallel
  {
    dbcsr::iterator iter(*mat_out);
    iter.start();

    while (iter.blocks_left()) {
      iter.next_block();

      int ioff = iter.row_offset();
      int joff = iter.col_offset();

      for (int j = 0; j != iter.col_size(); ++j) {
        for (int i = 0; i != iter.row_size(); ++i) {
          // use global indices for scamat
          // std::cout << iter(i,j) << std::endl;
          scamat.global_access(i + ioff, j + joff) = iter(i, j);
        }
      }
    }

    iter.stop();
    mat_out->finalize();
  }
  mat_out->release();
  scaldist->release();

  // scamat.print();

  return scamat;
}

template <typename T = double>
shared_matrix<T> scalapack_to_matrix(
    scalapack::distmat<T>& sca_mat_in,
    cart cart_in,
    std::string nameint,
    vec<int>& rowblksizes,
    vec<int>& colblksizes,
    std::string type = "")
{
  // form block-cyclic distribution

  bool sym = (type == "symmetric") ? true : false;
  bool lowtriang = (type == "lowtriang") ? true : false;

  int nfullrow = std::accumulate(
      rowblksizes.begin(), rowblksizes.end(), 0, std::plus<int>());
  int nfullcol = std::accumulate(
      colblksizes.begin(), colblksizes.end(), 0, std::plus<int>());

  int nrows = sca_mat_in.nrowstot();
  int ncols = sca_mat_in.ncolstot();

  if (nfullrow != nrows || nfullcol != ncols) {
    throw std::runtime_error(
        "Number of rows/cols of distmat != rows/cols of dbcsr matrix!");
  }

  // make distvecs
  vec<int> rowcycsizes = split_range(nfullrow, sca_mat_in.rowblk_size());
  vec<int> colcycsizes = split_range(nfullcol, sca_mat_in.colblk_size());

  vec<int> rowdist = cyclic_dist(rowcycsizes.size(), cart_in.dims()[0]);
  vec<int> coldist = cyclic_dist(colcycsizes.size(), cart_in.dims()[1]);

  auto cycdist = dist::create()
                     .set_cart(cart_in)
                     .row_dist(rowdist)
                     .col_dist(coldist)
                     .build();

  auto mat_cyclic = matrix<T>::create()
                        .name(nameint + "cyclic")
                        .set_dist(*cycdist)
                        .row_blk_sizes(rowcycsizes)
                        .col_blk_sizes(colcycsizes)
                        .matrix_type(type::no_symmetry)
                        .build();

  mat_cyclic->reserve_all();

#pragma omp parallel
  {
    dbcsr::iterator iter(*mat_cyclic);
    iter.start();

    while (iter.blocks_left()) {
      iter.next_block();

      int row = iter.row();
      int col = iter.col();

      int ioff = iter.row_offset();
      int joff = iter.col_offset();

      if (lowtriang && col > row)
        continue;

      if (lowtriang && row == col) {
        for (int i = 0; i != iter.row_size(); ++i) {
          for (int j = 0; j != i + 1; ++j) {
            iter(i, j) = sca_mat_in.global_access(i + ioff, j + joff);
          }
        }
      }
      else {
        for (int j = 0; j != iter.col_size(); ++j) {
          for (int i = 0; i != iter.row_size(); ++i) {
            // use global indices for scamat
            iter(i, j) = sca_mat_in.global_access(i + ioff, j + joff);
          }
        }
      }
    }

    iter.stop();
    mat_cyclic->finalize();
  }

  // make new matrix
  shared_matrix<T> mat_out =
      matrix<T>::create()
          .name(nameint)
          .set_cart(cart_in)
          .row_blk_sizes(rowblksizes)
          .col_blk_sizes(colblksizes)
          .matrix_type((sym) ? type::symmetric : type::no_symmetry)
          .build();

  bool keep_sp = false;

  if (sym) {
    mat_out->reserve_sym();
    keep_sp = true;
  }

  mat_out->complete_redistribute(*mat_cyclic, keep_sp);

  if (sym)
    mat_out->filter(1e-16);

  mat_cyclic->release();
  cycdist->release();

  return mat_out;
}

/*
//#endif // USE_SCALAPACK

template <typename T = double>
MatrixX<T> tensor_to_eigen(dbcsr::tensor<2,T>& array, int l = 0) {

        int myrank, mpi_size;

        MPI_Comm comm = array.comm();

        MPI_Comm_rank(comm, &myrank);
        MPI_Comm_size(comm, &mpi_size);

        arr<int,2> tsize = array.nfull_total();

        MatrixX<T> m_out(tsize[0],tsize[1]);

        // * we loop over each process, from which we broadcast
        // * each local block to all the other processes


        iterator_t<2,T> iter(array);
        iter.start();

        for (int p = 0; p != mpi_size; ++p) {

                int numblocks = -1;

                if (p == myrank) numblocks = array.num_blocks();

                MPI_Bcast(&numblocks,1,MPI_INT,p,comm);

                for (int iblk = 0; iblk != numblocks; ++iblk) {

                        // needed: blocksize, blockoffset, block
                        index<2> idx;
                        index<2> blkoff;
                        index<2> blksize;

                        if (p == myrank) {
                                iter.next();
                                idx = iter.idx();
                                blkoff = iter.offset();
                                blksize = iter.size();

                        }

                        MPI_Bcast(blkoff.data(),2,MPI_INT,p,comm);
                        MPI_Bcast(blksize.data(),2,MPI_INT,p,comm);

                        block<2,T> blk(blksize);

                        if (p == myrank) {
                                bool found;
                                blk = array.get_block(idx,blksize,found);
                        }

                        MPI_Bcast(blk.data(),blk.ntot(),MPI_DOUBLE,p,comm);

                        m_out.block(blkoff[0], blkoff[1], blksize[0],
blksize[1]) = Eigen::Map<MatrixX<T>>(blk.data(), blksize[0], blksize[1]);

                }

        }

        iter.stop();

        return m_out;

}


template <typename T = double>
dbcsr::tensor<2,T> eigen_to_tensor(MatrixX<T>& M, std::string name,
        dbcsr::pgrid<2>& grid, vec<int> map1, vec<int> map2, arrvec<int,2>
blk_sizes, double eps = global::filter_eps) {

        dbcsr::tensor<2,T> out = typename
dbcsr::tensor<2,T>::create().name(name).ngrid(grid)
                .map1(map1).map2(map2).blk_sizes(blk_sizes);

        out.reserve_all();

        //#pragma omp parallel
        //{
                dbcsr::iterator_t<2> iter(out);
                iter.start();

                while (iter.blocks_left()) {

                        iter.next();

                        auto& idx = iter.idx();
                        auto& size = iter.size();
                        auto& off = iter.offset();

                        MatrixX<T> eigenblk =
M.block(off[0],off[1],size[0],size[1]);

                        dbcsr::block<2> blk(size,eigenblk.data());

                        out.put_block(idx,blk);

                }

                out.finalize();
                iter.stop();
        //}

        return out;

}
*/

template <typename T>
void copy_tensor_to_matrix(
    tensor<2, T>& t_in, matrix<T>& m_out, std::optional<bool> summation)
{
  c_dbcsr_t_copy_tensor_to_matrix(
      t_in.m_tensor_ptr, m_out.m_matrix_ptr,
      (summation) ? &*summation : nullptr);
  return;
}

template <typename T>
void copy_matrix_to_tensor(
    matrix<T>& m_in, tensor<2, T>& t_out, std::optional<bool> summation)
{
  c_dbcsr_t_copy_matrix_to_tensor(
      m_in.m_matrix_ptr, t_out.m_tensor_ptr,
      (summation) ? &*summation : nullptr);
  return;
}

template <typename T>
Eigen::MatrixXd block_norms(matrix<T>& m_in, norm norm_type = norm::frobenius)
{
  // returns an eigen matrix with block norms
  int nrows = m_in.nblkrows_total();
  int ncols = m_in.nblkcols_total();

  Eigen::MatrixXd eigen_out(nrows, ncols);

  m_in.replicate_all();

  dbcsr::iterator iter(m_in);

  type mtype = m_in.matrix_type();

  //#pragma omp parallel
  //{

  iter.start();

  while (iter.blocks_left()) {
    iter.next_block();

    int iblk = iter.row();
    int jblk = iter.col();

    bool found = true;
    auto blk2 = m_in.get_block_p(iblk, jblk, found);

    eigen_out(iblk, jblk) = blk2.norm(norm_type);
    if (mtype == type::symmetric)
      eigen_out(jblk, iblk) = eigen_out(iblk, jblk);
  }

  iter.stop();
  m_in.finalize();
  //}//end parallel region

  m_in.distribute();
  return eigen_out;
}

template <typename T>
Eigen::SparseMatrix<double> sparse_block_norms(dbcsr::matrix<T>& m_in, 
  double eps, dbcsr::norm norm_type = dbcsr::norm::frobenius) 
{
  
  typedef Eigen::Triplet<double> triplet;
  
  auto dcart = m_in.get_cart();
  std::vector<triplet> sdata;
  //sdata.reserve(nblks);
  
  //auto eigen = dbcsr::matrix_to_eigen(m_in);
  
  /*if (dcart.rank() == 0) {
    std::cout << eigen << std::endl;
  }*/
  
  auto print = [&](auto v) {
    for (int ip = 0; ip != dcart.size(); ++ip) {
      if (ip == dcart.rank()) {
        std::cout << "RANK: " << ip << std::endl;
        for (auto e : v) {
          std::cout << e.row() << " " << e.col() << " : " << e.value() << std::endl;
        } std::cout << std::endl;
      }
      MPI_Barrier(dcart.comm());
    }
  };
  
  int nblks = 0;
  
  dbcsr::matrix<double>* mat_ptr = nullptr;
  if (m_in.matrix_type() == dbcsr::type::symmetric) {
    auto desym = m_in.desymmetrize();
    mat_ptr = new dbcsr::matrix<double>(std::move(*desym));
  } else {
    mat_ptr = &m_in;
  }
  
  dbcsr::iterator iter(*mat_ptr);
  iter.start();
    
  while (iter.blocks_left()) {
    
    iter.next_block();
    int irow = iter.row();
    int icol = iter.col();
    
    bool found = true;
    auto blk2 = mat_ptr->get_block_p(irow, icol, found);
    
    double val = blk2.norm(norm_type);
    
    if (val > eps) {
      sdata.push_back(triplet(irow, icol, val));
      ++nblks;
    }
  }
  
  iter.stop();
  
  if (m_in.matrix_type() == dbcsr::type::symmetric) {
    delete mat_ptr;
  }
  
  //print(sdata);
  
  int tripletsize = sizeof(triplet);
  
  int blklengths[1] = {tripletsize};
  MPI_Aint disps[1] = {0};
  MPI_Datatype types[1] = {MPI_BYTE};
  MPI_Datatype MPI_TRIPLET;
  
  MPI_Type_create_struct(1, blklengths, disps, types, &MPI_TRIPLET);
  MPI_Type_commit(&MPI_TRIPLET);
  
  // we now send all triplets to everyone
  
  int nblk_send = nblks;
  std::vector<int> nblk_recv(dcart.size(), 0);
  std::vector<int> nblk_recv_offset(dcart.size(), 0);
  
  MPI_Allgather(&nblks, 1, MPI_INT, nblk_recv.data(), 1, MPI_INT, dcart.comm());
  
  int nblkstot = std::accumulate(nblk_recv.begin(), nblk_recv.end(), 0);
  int offset = 0;
  for (int ii = 0; ii != nblk_recv.size(); ++ii) {
    nblk_recv_offset[ii] = offset;
    offset += nblk_recv[ii];
  }
  
  std::vector<triplet> rdata(nblkstot);
  
  MPI_Allgatherv(sdata.data(), nblk_send, MPI_TRIPLET,
    rdata.data(), nblk_recv.data(), nblk_recv_offset.data(), 
    MPI_TRIPLET, dcart.comm());
  
  sdata.clear();
  //print(rdata);
  
  Eigen::SparseMatrix<double> smat(m_in.nblkrows_total(), m_in.nblkcols_total());
  smat.setFromTriplets(rdata.begin(), rdata.end());
  
  return smat;
  
}
  
}  // namespace dbcsr
  
#endif
