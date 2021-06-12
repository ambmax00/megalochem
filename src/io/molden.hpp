#ifndef IO_MOLDEN_HPP
#define IO_MOLDEN_HPP

#include <fstream>
#include "desc/wfn.hpp"
#include "megalochem.hpp"
#include "utils/ele_to_int.hpp"
#include <optional>

namespace megalochem {

namespace io {

inline double gaussian_int(int l, double alpha)
{
	double l1 = (l + 1) * 0.5;
	double res = tgamma(l1) / (2.0 * pow(alpha, l1));
	return res;
}

inline double gto_norm(int l, double e)
{
	return 1.0 / sqrt(gaussian_int(l * 2 + 2, 2 * e));
}

inline void write_molden(
    std::string filename,
    world w,
    desc::molecule& mol,
    dbcsr::shared_matrix<double> c_bo_A,
    dbcsr::shared_matrix<double> c_bv_A,
    std::vector<double> eps_occ_A,
    std::vector<double> eps_vir_A,
    dbcsr::shared_matrix<double> c_bo_B,
    dbcsr::shared_matrix<double> c_bv_B,
    std::optional<std::vector<double>> eps_occ_B,
    std::optional<std::vector<double>> eps_vir_B)
{

  std::ofstream file;
  auto cbas = mol.c_basis();
  auto atoms = mol.atoms();
  auto blkmap = cbas->block_to_atom(atoms);
  
  
  std::vector<int> smap = {0};
    std::vector<int> pmap = {0, 1, 2};
    std::vector<int> dmap = {2, 3, 1, 4, 0};
    std::vector<int> fmap = {3, 4, 2, 5, 1, 6, 0};
    std::vector<int> gmap = {4, 5, 3, 6, 2, 7, 1, 8, 0};

  if (w.rank() == 0) {
	  
    file.open(filename);

    file << std::setprecision(6);
    file << std::scientific;

    file << "[Molden Format]\n";

    file << "[Atoms] (AU)\n";

    

    for (size_t ii = 0; ii != atoms.size(); ++ii) {
      file << util::int_to_ele[atoms[ii].atomic_number] << " ";
      file << ii + 1 << " " << atoms[ii].atomic_number << " ";
      file << atoms[ii].x << " " << atoms[ii].y << " " << atoms[ii].z << '\n';
    }

    file << "[GTO]\n";

    std::vector<char> labels = {'s', 'p', 'd', 'f', 'g'};

    for (size_t ii = 0; ii != cbas->size(); ++ii) {
      int iatom = blkmap[ii];
      for (auto s : cbas->at(ii).shells) {
        file << iatom + 1 << " " << 0 << '\n';
        file << labels[s.l] << " " << s.ncontr() << " " << 1.0 << '\n';

        for (size_t jj = 0; jj != s.ncontr(); ++jj) {
		  double fac = 1.0; //1.0 / gto_norm(s.l, s.alpha[jj]);
          file << s.alpha[jj] << " " << fac * s.coeff[jj] << '\n';
        }
        file << '\n';
      }
    }

    file << "[5D]\n[7F]\n[9G]\n\n";

    file << "[MO]\n";
    
  }

    auto write_coeff = [&](auto cmat, auto eps, bool occ, std::string spin) {
      
      auto cmat_eigen = dbcsr::matrix_to_eigen(*cmat);
	  int norb = cmat_eigen.cols();
	  
	  if (w.rank() == 0) {

		  std::string occup = (occ) ? "2" : "0";

		  for (int iorb = 0; iorb != norb; ++iorb) {
			file << "Sym= C1\n";
			file << "Ene= " << eps[iorb] << '\n';
			file << "Spin= "
				 << spin << '\n';
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
				  file << noff + ii + 1 << " " << cmat_eigen(noff + map[ii], iorb)
					   << '\n';
				}

				noff += size;

			  }  // end for ishell
			}  // end for icluster

		  }  // end for iorb
	  } // endif
    };  // end lambda

    write_coeff(c_bo_A, eps_occ_A, true, "Alpha");
    write_coeff(c_bv_A, eps_vir_A, false, "Alpha");
    if (c_bo_B) write_coeff(c_bo_B, *eps_occ_B, true, "Beta");
    if (c_bv_B) write_coeff(c_bv_B, *eps_vir_B, false, "Beta");

    if (w.rank() == 0) file.close();

  MPI_Barrier(w.comm());
}

}  // namespace io

}  // namespace megalochem

#endif
