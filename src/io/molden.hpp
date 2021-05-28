#ifndef IO_MOLDEN_HPP
#define IO_MOLDEN_HPP

#include <fstream>
#include "desc/wfn.hpp"
#include "megalochem.hpp"
#include "utils/ele_to_int.hpp"

namespace megalochem {

namespace io {

inline void write_molden(
    std::string filename,
    world w,
    desc::molecule& mol,
    dbcsr::matrix<double>& c_bo,
    dbcsr::matrix<double>& c_bv,
    std::vector<double>& eps_occ,
    std::vector<double>& eps_vir)
{
  auto c_bo_eigen = dbcsr::matrix_to_eigen(c_bo);
  auto c_bv_eigen = dbcsr::matrix_to_eigen(c_bv);

  if (w.rank() == 0) {
    std::ofstream file(filename);

    file << std::setprecision(6);
    file << std::scientific;

    file << "[Molden Format]\n";

    file << "[Atoms] (AU)\n";

    auto atoms = mol.atoms();

    for (size_t ii = 0; ii != atoms.size(); ++ii) {
      file << util::int_to_ele[atoms[ii].atomic_number] << " ";
      file << ii + 1 << " " << atoms[ii].atomic_number << " ";
      file << atoms[ii].x << " " << atoms[ii].y << " " << atoms[ii].z << '\n';
    }

    file << "[GTO]\n";

    auto cbas = mol.c_basis();
    auto blkmap = cbas->block_to_atom(atoms);

    std::vector<char> labels = {'s', 'p', 'd', 'f', 'g'};

    for (size_t ii = 0; ii != cbas->size(); ++ii) {
      int iatom = blkmap[ii];
      for (auto s : cbas->at(ii).shells) {
        file << iatom + 1 << " " << 0 << '\n';
        file << labels[s.l] << " " << s.ncontr() << " " << 1.0 << '\n';

        for (size_t jj = 0; jj != s.ncontr(); ++jj) {
          file << s.alpha[jj] << " " << s.coeff[jj] << '\n';
        }
        file << '\n';
      }
    }

    file << "[5D]\n[7F]\n[9G]\n\n";

    file << "[MO]\n";

    std::vector<int> smap = {0};
    std::vector<int> pmap = {0, 1, 2};
    std::vector<int> dmap = {2, 3, 1, 4, 0};
    std::vector<int> fmap = {3, 4, 2, 5, 1, 6, 0};
    std::vector<int> gmap = {4, 5, 3, 6, 2, 7, 1, 8, 0};

    auto write_coeff = [&](auto cmat, auto eps, bool occ) {
      int norb = cmat.cols();

      std::string occup = (occ) ? "1" : "0";

      for (int iorb = 0; iorb != norb; ++iorb) {
        file << "Sym= C1\n";
        file << "Ene= " << eps[iorb] << '\n';
        file << "Spin= "
             << "Alpha" << '\n';
        file << "Occup= " << occup << '\n';

        int noff = 0;

        for (size_t icluster = 0; icluster != cbas->size(); ++icluster) {
          auto& c = cbas->at(icluster);

          for (size_t ishell = 0; ishell != c.shells.size(); ++ishell) {
            auto s = c.shells[ishell];
            int size = s.size();

            std::vector<int> map;

            switch (s.l) {
              case 0:
                map = smap;
                break;
              case 1:
                map = pmap;
                break;
              case 2:
                map = dmap;
                break;
              case 3:
                map = fmap;
                break;
              case 4:
                map = gmap;
                break;
            }

            for (int ii = 0; ii != size; ++ii) {
              file << noff + ii + 1 << " " << cmat(noff + map[ii], iorb)
                   << '\n';
            }

            noff += size;

          }  // end for ishell
        }  // end for icluster

      }  // end for iorb
    };  // end lambda

    write_coeff(c_bo_eigen, eps_occ, true);
    write_coeff(c_bv_eigen, eps_vir, false);

    file.close();
  }

  MPI_Barrier(w.comm());
}

}  // namespace io

}  // namespace megalochem

#endif
