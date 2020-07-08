#include "fock/jkbuilder.h"
#include "fock/fock_defaults.h"
#include "math/linalg/LLT.h"
#include "ints/screening.h"
#include <dbcsr_tensor_ops.hpp>
#include <libxsmm.h>

namespace fock {

/*
CADF_K::CADF_K(dbcsr::world& w, desc::options& opt) : K(w,opt) {}

void CADF_K::compute_fit() {
	
	

void CADF_K::compute_L3() {
	
	// ============ BLOCKING INFO
	
	// get block_to_atom mapping
	auto x = m_mol->dims().x();
	auto b = m_mol->dims().b();
	
	auto cbas = m_mol->c_basis();
	auto xbas = *m_mol->c_dfbasis();
	
	int nbasb = cbas.nbf();
	int nbasx = xbas.nbf();
	
	auto atoms = m_mol->atoms();
	int natoms = atoms.size();
	
	auto oncentre = [](libint2::Shell& s, libint2::Atom& a) {
		double r = sqrt(pow(s.O[0] - a.x, 2) + pow(s.O[1] - a.y,2) + pow(s.O[2] - a.z,2));
		if (r < std::numeric_limits<double>::epsilon()) return true;
		return false;
	};
	
	auto blk_to_atom = [&](desc::cluster_basis& cbas, vec<int>& dim) 
	{
		vec<int> out(dim.size());
		for (int i = 0; i != out.size(); ++i) {
			auto& shell = cbas[i][0];
			for (int a = 0; a != natoms; ++a) {
				auto atom = atoms[a];
				if (oncentre(shell,atom)) {
					out[i] = a;
					break;
				}
			}
		}
		return out;
	};
	
	auto atom_to_blk = [](vec<int>& blk_to_atom, int dim) 
	{
		vec<vec<int>> out(dim);
		for (int i = 0; i != blk_to_atom.size(); ++i) {
			 out[blk_to_atom[i]].push_back(i);
		}
		return out;
	};
	
	m_blk_to_atom_x = blk_to_atom(xbas,x); // how each block in x maps to atoms
	m_blk_to_atom_b = blk_to_atom(cbas,b);
	
	m_atom_to_blk_x = atom_to_blk(m_blk_to_atom_x,x.size());
	m_atom_to_blk_b = atom_to_blk(m_blk_to_atom_b,b.size());
	
	// ============ SET UP TENSORS
	
	m_eri_batched->set_batch_dim(vec<int>{1});
	auto eri = m_eri_batched->get_stensor();
	
	int nsigbatches = m_eri_batched->nbatches();
	
	dbcsr::stensor3_d INTS_02_1 = dbcsr::make_stensor<3>(
		dbcsr::tensor3_d::create_template(*eri).name("INTS_02_1")
		.map1({0,2}).map2({1}));
		
	dbcsr::stensor3_d C_0_12 = dbcsr::make_stensor<3>(
		dbcsr::tensor3_d::create_template(*eri).name("C_0_12"));
		
	m_scr = m_reg.get_screener(m_mol->name() + "_schwarz_screener");
	
	if (!m_scr) {
		std::cout << "NOT HERE." << std::endl;
		exit(-1);
	}
	
	// ============= START BATCHING
	
	// C_X_SIGMA_NU
	
	Eigen::MatrixXd sum_ccd = Eigen::MatrixXd::Zero(nbasx,nbasb);
	Eigen::MatrixXd sum_ced = Eigen::MatrixXd::Zero(nbasx,nbasb);
	Eigen::MatrixXd sum_ccdcd = Eigen::MatrixXd::Zero(nbasx,nbasb);
	Eigen::MatrixXd sum_ced_global = Eigen::MatrixXd::Zero(nbasx,nbasb);
		
	for (int SIGBATCH = 0; SIGBATCH != nsigbatches; ++SIGBATCH) {
		
		// get the integrals
		m_factory->fetch_ao_3c2e(m_eri_batched, SIGBATCH, false, nullptr, nullptr);
		
		// form C_xsn = eri_ysn * inv_yx
		C_0_12->reserve_template(*eri);
		dbcsr::contract(*eri, *m_s_xx_inv, *C_0_12).retain_sparsity(true)
			.perform("YMN, YX -> XMN");
			
		// loop over blocks
		dbcsr::iterator_t iter(*C_0_12);
		
		iter.start();
		
		while(iter.blocks_left()) {
			
			iter.next();
			
			auto& idx = iter.idx();
			auto& off = iter.offset();
			auto& size = iter.size();
			
			int x_c = m_blk_to_atom_x[idx[0]];
			int sig_c = m_blk_to_atom_b[idx[1]];
			int nu_c = m_blk_to_atom_b[idx[2]];
			
			int xoff = off[0];
			int sigoff = off[1];
			int nuoff = off[2];
			
			// C_Xsn_cdc
			bool found = false;
			auto blk3 = C_0_12->get_block(idx,size,found);
			
			if (x_c == sig_c && x_c == nu_c) {
				
				// surivive three terms: ccd, ced delta(cd), ccd delta(cd)
				for (int ix = 0; ix != size[0]; ++ix) {
					for (int isig = 0; isig != size[1]; ++isig) {
						double sum = 0.0;
						for (int inu = 0; inu != size[2]; ++inu) {
							sum += pow(blk3(ix,isig,inu),2);
						}
						sum_ced(ix + xoff, isig + sigoff) += sum;
						sum_ccd(ix + xoff, isig + sigoff) += sum;
						sum_ccdcd(ix + xoff, isig + sigoff) += sum;
					}
				}
				
			} else if (x_c != sig_c && x_c == nu_c) {
				
				// survive one term: ccd
				for (int ix = 0; ix != size[0]; ++ix) {
					for (int isig = 0; isig != size[1]; ++isig) {
						double sum = 0.0;
						for (int inu = 0; inu != size[2]; ++inu) {
							sum += pow(blk3(ix,isig,inu),2);
						}
						sum_ccd(ix + xoff, isig + sigoff) += sum;
					}
				}
				
			} else if (x_c == sig_c && x_c != nu_c) {
				
				//survive one term: ced delta(ccd)
				for (int ix = 0; ix != size[0]; ++ix) {
					for (int isig = 0; isig != size[1]; ++isig) {
						double sum = 0.0;
						for (int inu = 0; inu != size[2]; ++inu) {
							sum += pow(blk3(ix,isig,inu),2);
						}
						sum_ced(ix + xoff, isig + sigoff) += sum;
					}
				}
				
			}
			
		} // end block loop
		
		MPI_Allreduce(sum_ced.data(),sum_ced_global.data(),nbasx*nbasb,
			MPI_DOUBLE, MPI_SUM, m_world.comm());
			
		for (size_t i = 0; i != nbasx*nbasb; ++i) {
			sum_ced.data()[i] = sqrt(sum_ced.data()[i]);
			sum_ccd.data()[i] = sqrt(sum_ccd.data()[i]);
			sum_ccdcd.data()[i] = sqrt(sum_ccdcd.data()[i]);
		}
		
		
		sum_ced.resize(0,0);
		sum_ccd += std::move(sum_ced_global) - std::move(sum_ccdcd);
		
		auto s_xx_diag = m_s_xx->get_diag();
		
		for (int i = 0; i != nbasx; ++i) {
		 for (int j = 0; j != nbasb; ++j) {
			sum_ccd(i,j) *= sqrt(s_xx_diag[i]);
		}}
		
		std::cout << "CBAR: " << std::endl;
		std::cout << sum_ccd << std::endl;
		
		eri->clear();
		
	}
	
	dbcsr::mat_d c_temp = dbcsr::eigen_to_matrix(sum_ccd, m_world, "Cbar_xb", x, b, dbcsr_type_no_symmetry);
	dbcsr::smat_d Cbar_xb = c_temp.get_smatrix();
	sum_ccd.resize(0,0);	
	
	dbcsr::smat_d dbar_xb = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create_template(*Cbar_xb).name("dbar_xb"));
		
	dbcsr::multiply('N', 'N', *Cbar_xb, *m_p_A, *dbar_xb).perform();
	
	dbar_xb->apply(dbcsr_func_spread_from_zero);
	
	dbcsr::print(*dbar_xb);
	
	auto dbar_xb_blknorms = dbcsr::block_norms(*dbar_xb);
	
	std::array<int,3> I = {0,0,0};
	
	while (I[0] < x.size()) {
		while (I[1] < b.size()) {
			while (I[2] < b.size()) {
				if (eri->proc(I) == m_world.rank()) {
					m_L3[0].push_back(I[0]);
					m_L3[1].push_back(I[1]);
					m_L3[2].push_back(I[2]);
				}
				++I[2];
			}
			I[2] = 0;
			++I[1];
		}
		I[1] = 0;
		++I[0];
	}
	
	std::cout << "L3" << std::endl;
	for (int n = 0; n != m_L3[0].size(); ++n) {
		for (int i = 0; i != 3; ++i) {
			std::cout << m_L3[i][n] << " ";
		} std::cout << std::endl;
	}
	
}

void CADF_K::compute_LB() {
	
	auto eri = m_eri_batched->get_stensor();
	auto x = m_mol->dims().x();
	auto b = m_mol->dims().b();
	
	std::array<int,3> I = {0,0,0};
	
	while (I[0] < x.size()) {
		while (I[1] < b.size()) {
			while (I[2] < b.size()) {
				if (eri->proc(I) == m_world.rank()) {
					m_LB[0].push_back(I[0]);
					m_LB[1].push_back(I[1]);
					m_LB[2].push_back(I[2]);
				}
				++I[2];
			}
			I[2] = 0;
			++I[1];
		}
		I[1] = 0;
		++I[0];
	}
	
	std::cout << "LB" << std::endl;
	for (int n = 0; n != m_LB[0].size(); ++n) {
		for (int i = 0; i != 3; ++i) {
			std::cout << m_LB[i][n] << " ";
		} std::cout << std::endl;
	}
	
}

void CADF_K::init_tensors() {
	
	m_s_xx_inv = m_reg.get_tensor<2,double>(m_mol->name() + "_s_xx_inv_(0|1)");
	
	m_s_xx = m_reg.get_matrix<double>(m_mol->name() + "_s_xx");
	
	m_eri_batched = m_reg.get_btensor<3,double>(m_mol->name() + "_i_xbb_(0|12)_batched");
	
	compute_fit();
	
}

void CADF_K::compute_K() {
	
	compute_L3();
	
	compute_LB();
	
	m_eri_batched->set_batch_dim(vec<int>{0});
	auto eri = m_eri_batched->get_stensor();
	int nbatches_x = m_eri_batched->nbatches();
	
	arrvec<int,3> m_batch_L3;
	arrvec<int,3> m_batch_LB;
	arrvec<int,3> blkbatch;
	
	int blkoff = 0;
	int nblk = 0;
	
	m_s_xx->replicate_all();
	
	for (int XBATCH = 0; XBATCH != nbatches_x; ++XBATCH) {
		
		std::cout << "BATCH: " << XBATCH << std::endl;
		
		blkbatch[0] = m_eri_batched->bounds_blk(XBATCH,0);
		blkbatch[1] = m_eri_batched->bounds_blk(XBATCH,1);
		blkbatch[2] = m_eri_batched->bounds_blk(XBATCH,2);
		
		int nblk = 0;
		while (m_L3[0][blkoff+nblk] <= blkbatch[0][1]) {
			 ++nblk;
		}
		
		blkoff += nblk;
		 
		m_batch_L3[0].insert(m_batch_L3[0].end(),
			m_L3[0].begin()+blkoff,m_L3[0].begin()+nblk+blkoff);
		m_batch_L3[1].insert(m_batch_L3[1].end(),
			m_L3[1].begin()+blkoff,m_L3[1].begin()+nblk+blkoff);
		m_batch_L3[2].insert(m_batch_L3[2].end(),
			m_L3[2].begin()+blkoff,m_L3[2].begin()+nblk+blkoff);
		
		eri->reserve(m_L3);

		m_factory->ao_3c2e_setup();
		m_factory->ao_3c2e_fill(eri);

		// LOOP OVER BLOCKS OF TENSOR
		dbcsr::iterator_t iter(*eri);

#pragma omp parallel
{

		iter.start();

		while(iter.blocks_left()) {
		 
			iter.next();
			auto& idx = iter.idx();
			auto& size = iter.size();
			auto& off = iter.offset();

			int x_c = m_blk_to_atom_x[idx[0]];
			int mu_c = m_blk_to_atom_b[idx[1]];
			int lam_c = m_blk_to_atom_b[idx[2]];

			bool found = false;
			auto blk3_x_ml = eri->get_block(idx,size,found);

			vec<int>& blks_a = m_atom_to_blk_x[mu_c];
			vec<int>& blks_b = m_atom_to_blk_x[lam_c];

			std::cout << "For block " << idx[0] << " " << idx[1]
					<< " " << idx[2] << std::endl;
						
			for (auto a : blks_a) {
				std::cout << a << " ";
			} std::cout << std::endl;
			
			for (auto a : blks_b) {
				std::cout << a << " ";
			} std::cout << std::endl;
			
			int m = size[1] * size[2];
			int k = size[0];
			int n = size[0];
			
			std::array<int,3> cidx = {0,0,0};
			
			for (auto A : blks_a) {
				auto blk3A_xbb = m
				auto blk2A_xy = m_s_xx->get_block_p(A, idx[0], found);
				libxsmm_dgemm('T', 'N', 
			
			libxsmm_dgemm();
			
			
		}
		
		iter.finish();
		
} // end omp parallel

	}
		
		
}
	
void CADF_K::init_tensors() {
	
	m_inv = m_reg.get_matrix<double>(m_mol->name() + "_s_xx");
	
	// taken from D.C. Ghosh et al./Journal of Molecular Structure: THEOCHEM 865 (2008) 60–67
	std::vector<double> atomic_radii =
	{ 
	1.0000, 												0.5883,
	3.0770, 2.0513, 1.5384, 1.2308, 1.0257, 0.8791, 0.7693, 0.6837
	};
	
	auto atoms = m_mol->atoms();
	int natoms = atoms.size();
	auto xbas = *m_mol->c_dfbasis();
	auto x = m_mol->dims().x();
	int nxbf = xbas.nbf();

	auto distAX = [](libint2::Atom& a1, libint2::Atom& a2) 
	{
		return sqrt(pow(a1.x - a2.x,2) + pow(a1.y + a2.y,2) + pow(a1.z + a2.z,2));
	};
	
	auto oncentre = [](libint2::Shell& s, libint2::Atom& a) {
		double r = sqrt(pow(s.O[0] - a.x, 2) + pow(s.O[1] - a.y,2) + pow(s.O[2] - a.z,2));
		if (r < std::numeric_limits<double>::epsilon()) return true;
		return false;
	};
	
	auto bumpfunc = [](double r, double r0, double r1) {
		if (r <= r0) return 1.0;
		if (r >= r1) return 0.0;
		
		return 1.0/(1+ exp((r1 - r0)/(r1 - r) - (r1 - r0)/(r-r0)));
		
	};
	
	vec<int> mappings(x.size()); // how each block in x maps to atoms
	vec<int> nshellblks(natoms,0); // how many shell blocks on atom
	
	for (int i = 0; i != mappings.size(); ++i) {
		auto& shell = xbas[i][0];
		for (int a = 0; a != natoms; ++a) {
			auto atom = atoms[a];
			if (oncentre(shell,atom)) {
				nshellblks[a]++;
				mappings[i] = a;
				break;
			}
		}
	}
	
	for (auto i : mappings) {
		std::cout << i << " ";
	} std::cout << std::endl;				
	
	// set up PQ (off)diagonals
	dbcsr::smat_d inv_D = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create_template(*m_inv).name("Metric Diag"));
		
	dbcsr::smat_d inv_OD = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create_template(*m_inv).name("Metric Offdiag"));

	dbcsr::smat_d temp = std::make_shared<dbcsr::mat_d>(
		dbcsr::mat_d::create_template(*m_inv).name("TEMP"));

	// block diagonal
	
	vec<int> resrow_diag;
	vec<int> rescol_diag;
	
	int off = 0;
	for (int a = 0; a != natoms; ++a) {
		for (int jx = 0; jx != nshellblks[a]; ++jx) {
			for (int ix = 0; ix <= jx; ++ix) {
				resrow_diag.push_back(ix + off);
				rescol_diag.push_back(jx + off);
			}
		}
		off += nshellblks[a];
	}
				
	inv_D->reserve_blocks(resrow_diag,rescol_diag);
	
	inv_D->copy_in(*m_inv, true);
	inv_D->filter();
	
	inv_OD->copy_in(*m_inv,false);
	inv_OD->add(1.0, -1.0, *inv_D);
	inv_OD->filter();
	
	dbcsr::print(*m_inv);
	dbcsr::print(*inv_D);
	dbcsr::print(*inv_OD);
	

	for (int ax = 0; ax != natoms; ++ax) {
		
		std::vector<double> bfacs(natoms);
		
		auto ax_atom = atoms[ax];
		int ax_atnum = ax_atom.atomic_number;
		
		for (int aa = 0; aa != natoms; ++aa) {
			
			auto aa_atom = atoms[aa];
			int aa_atnum = aa_atom.atomic_number;
			
			double r0 = 2 * atomic_radii[ax_atnum] 
				+ 2*atomic_radii[aa_atnum];
				
			double r1 = r0 + 1.0;
			
			double r = distAX(ax_atom,aa_atom);
				
			bfacs[aa] = bumpfunc(r,r0,r1);
			
		}
		
		for (auto b : bfacs) {
			std::cout << b << " ";
		} std::cout << std::endl;
		
		std::vector<double> bfacs_all(nxbf);
		
		off = 0;
		for (int i = 0; i != x.size(); ++i) {
			double ifac = bfacs[mappings[i]];
			for (int j = 0; j != x[i]; ++j) {
				bfacs_all[j + off] = ifac;
			}
			off += x[i];
		}
		
		for (auto b : bfacs_all) {
			std::cout << b << " ";
		} std::cout << std::endl;
		
		// BX * inv_OD * BX
		temp->copy_in(*inv_OD);
		temp->scale(bfacs_all,"right");
		temp->scale(bfacs_all,"left");
		
		dbcsr::print(*temp);
		
		temp->add(1.0,1.0,*inv_D);
		
		dbcsr::print(*temp);
		
		math::LLT inverter(temp,1);
		
		inverter.compute();
		
		auto PQinv = inverter.inverse(x);
		
		dbcsr:print(*PQinv);
		
		PQinv->scale(bfacs_all,"right");
		PQinv->scale(bfacs_all,"left");
		
		dbcsr::print(*PQinv);
		
		m_bumpmats.push_back(PQinv);
		
	}
	
	exit(0);
	
}

void CADF_K::compute_K() {
	
	TIME.start();
	
	exit(0);
	
	TIME.finish();
		
}*/
	
} // end namespace
