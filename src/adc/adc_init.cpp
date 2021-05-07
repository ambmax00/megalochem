#include "adc/adcmod.hpp"
#include "ints/aoloader.hpp"
#include "ints/screening.hpp"
#include "math/linalg/LLT.hpp"
#include "locorb/locorb.hpp"
#include "extern/scalapack.hpp"
#include "extern/lapack.hpp"
#include "utils/scheduler.hpp"
#include "math/solvers/hermitian_eigen_solver.hpp"
#include <Eigen/Eigenvalues>

#include <type_traits>

namespace megalochem {

namespace adc {
	
std::tuple<dbcsr::shared_matrix<double>, dbcsr::sbtensor<3,double>> 
adcmod::test_fitting(std::vector<bool> atom_idx) {
	
	// let X,Y,Z -> all auxiliary functions
	// let R, S, T -> auxiliary functions in domain
	
	LOG.os<>("Computing TEST FITTING\n");
	
	// ============== create (X|R), to scalapack =======================
	
	LOG.os<>("Setting up metric matrix.\n");
	
	LOG.os<>("1\n");
	
	auto mol = m_wfn->mol;
	
	ints::aofactory aofac(mol, m_world);
	
	auto b = mol->dims().b();
	auto x = mol->dims().x();
	auto atoms = mol->atoms();
	auto cbas = mol->c_basis();
	auto xbas = mol->c_dfbasis();
	
	std::vector<int> r; // block sizes for reduced aux
	std::vector<int> rx; // mapping r -> x
	
	LOG.os<>("1\n");
	
	auto blkatom_X = xbas->block_to_atom(atoms);
	
	for (int ix = 0; ix != (int)x.size(); ++ix) {
		int iatom = blkatom_X[ix];
		if (atom_idx[iatom]) {
			r.push_back(x[ix]);
			rx.push_back(ix);
		}
	}
	
	LOG.os<>("2\n");
	
	auto metric_XY = aofac.ao_2c2e(ints::metric::coulomb);
	
	auto metric_XY_nosym = metric_XY->desymmetrize();
	metric_XY_nosym->replicate_all();
	
	LOG.os<>("3\n");
	
	auto metric_out = dbcsr::matrix<double>::create()
		.set_cart(m_world.dbcsr_grid())
		.name("metric_local")
		.row_blk_sizes(x)
		.col_blk_sizes(x)
		.matrix_type(dbcsr::type::symmetric)
		.build();
	
	auto metric_XR = dbcsr::matrix<double>::create()
		.set_cart(m_world.dbcsr_grid())
		.name("metric_XR")
		.row_blk_sizes(x)
		.col_blk_sizes(r)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	LOG.os<>("4\n");
		
	metric_XR->reserve_all();
	
	dbcsr::iterator iter(*metric_XR);
	iter.start();
	
	while (iter.blocks_left()) {
		
		iter.next_block();
		
		int iblk_x = iter.row();
		int iblk_r = iter.col();
	
		int iblk_rx = rx[iblk_r];
		
		bool found = true;
		auto blk = metric_XY_nosym->get_block_p(iblk_x, iblk_rx, found);
	
		if (!found) continue;
		
		std::copy(blk.begin(), blk.end(), iter.data());
		
	}
	
	iter.stop();
	
	metric_out->reserve_sym();
	
	for (int ir = 0; ir != r.size(); ++ir) {
		for (int jr = 0; jr != r.size(); ++jr) {
			
			auto ix = rx[ir];
			auto jx = rx[jr];
			
			if (metric_out->proc(ix,jx) != m_world.rank()) continue;
			
			bool found = false;
			
			auto blkin = metric_XY->get_block_p(ix, jx, found);
			
			if (!found) continue;
			
			auto blkout = metric_out->get_block_p(ix, jx, found);
	
			std::copy(blkin.begin(), blkin.end(), blkout.begin());
			
		}
	}
	
	auto sgrid = m_world.scalapack_grid();
	
	LOG.os<>("5\n");
	
	auto mscala_XR = dbcsr::matrix_to_scalapack(metric_XR, sgrid, 
		"mscala_XR", 16, 16, 0, 0);

	//metric_XY->release();
	metric_XR->release();
	metric_XY_nosym->release();
	
	LOG.os<>("6\n");
	
	// ============= solve QR decompostion of (X|R) ====================
	
	LOG.os<>("Performing QR decomposition.\n");
	
	int nx = mscala_XR.nrowstot();
	int nr = mscala_XR.ncolstot();
	
	LOG.os<>("Problem size: ", nx, " x ", nr, '\n');
	
	// make row grids
	
	/*
	int* usermap = new int[sgrid.npcol()];
	
	for (int iprow = 0; iprow != sgrid.nprow(); ++iprow) {
		if (iprow == sgrid.myprow()) {
			std::cout << "USERMAP for " << iprow << " with ";
			for (int ipcol = 0; ipcol != sgrid.npcol(); ++ipcol) {
				usermap[ipcol] = sgrid.get_pnum(iprow,ipcol);
				std::cout << usermap[ipcol] << " ";
			}	
			std::cout << std::endl;
		}
	} 
	
	int row_ctx = sgrid.ctx();
	
	c_blacs_gridmap(&row_ctx, usermap, 1, 1, sgrid.npcol()); 
	
	scalapack::grid row_grid(row_ctx);
	
	LOG.os<>("GRID: ", row_grid.nprow(), " ", row_grid.npcol(), '\n');
	*/
	scalapack::distmat<double> tau_dist(sgrid, 1, nr, 16, 16, 0, 0);
	
	double* tau = new double[nx]();
	double* work = nullptr;
	double work_query = 0;
	int lwork = 0;
	int info = 0;

	// query work size
	c_pdgeqrf(nx, nr, mscala_XR.data(), 0, 0, mscala_XR.desc().data(),
		tau_dist.data(), &work_query, -1, &info);
	
	lwork = (int)work_query;
	work = new double[lwork];	
	
	// perform computation
	c_pdgeqrf(nx, nr, mscala_XR.data(), 0, 0, mscala_XR.desc().data(),
		tau_dist.data(), work, lwork, &info);
		
	LOG.os<>("pdgeqrf exited with ", info, '\n');
	
	if (info != 0) {
		throw std::runtime_error("QR decomposition failed.\n");
	}
	
	/*
	for (int iproc = 0; iproc != m_world.size(); ++iproc) {
		if (iproc == m_world.rank()) {
			std::cout << "PROC " << iproc << " " << sgrid.myprow() << " " << sgrid.mypcol() << std::endl;
			for (int ii = 0; ii != nx; ++ii) {
				std::cout << tau[ii] << " ";
			} std::cout << std::endl;
		}
		MPI_Barrier(m_world.comm());
	}*/
	
	LOG.os<>("TAU DIST\n");
	tau_dist.print();
	
	delete [] work;
	
	// ===================== Redistribute to all processes =============
	
	std::cout << "1" << std::endl;
	scalapack::grid grid_self(MPI_COMM_SELF, 'A', 1, 1);
	
	std::cout << "2" << std::endl;
	
	scalapack::distmat<double> tau_self(grid_self, 1, nr, 1, nr, 0, 0);
	
	scalapack::distmat<double> mscala_XR_self(grid_self, nx, nr, nx, nr, 0, 0);
	
	LOG.os<>("Copying over tau...\n");
	
	auto desc_single = tau_self.desc();
	desc_single[1] = (m_world.rank() == 0) ? desc_single[1] : -1;
	
	c_pdgemr2d(1, nr, tau_dist.data(), 0, 0, tau_dist.desc().data(), tau_self.data(),
		0, 0, desc_single.data(), m_world.scalapack_grid().ctx());
		
	desc_single = mscala_XR_self.desc();
	desc_single[1] = (m_world.rank() == 0) ? desc_single[1] : -1;
	
	LOG.os<>("Copying over metric...\n");
	
	c_pdgemr2d(nx, nr, mscala_XR.data(), 0, 0, mscala_XR.desc().data(), 
		mscala_XR_self.data(), 0, 0, desc_single.data(), 
		m_world.scalapack_grid().ctx());
	
	//std::cout << "TAU ALL: " << std::endl;
	//tau_dist.print();
	
	MPI_Bcast(tau_self.data(), nr, MPI_DOUBLE, 0, m_world.comm());
	
	MPI_Bcast(mscala_XR_self.data(), nx * nr, MPI_DOUBLE, 0, m_world.comm());
	
	MPI_Barrier(m_world.comm());
	
	if (m_world.rank() == 0) {
		std::cout << "PROC 0" << std::endl;
		tau_self.print();
	}
	
	// ==================== SETUP TENSORS ==============================
	
	auto spgrid3 = dbcsr::pgrid<3>::create(m_world.comm()).build();
	auto spgrid3_self = dbcsr::pgrid<3>::create(MPI_COMM_SELF).build();
	
	arrvec<int,3> xbb = {x,b,b};
	std::array<int,3> bdims = {5,5,5};
	arrvec<int,3> blkmaps = {
		xbas->block_to_atom(atoms),
		cbas->block_to_atom(atoms),
		cbas->block_to_atom(atoms)
	};
	
	auto cfit_xbb = dbcsr::tensor<3>::create()
		.set_pgrid(*spgrid3)
		.name("cfit_xbb")
		.map1({0})
		.map2({1,2})
		.blk_sizes(xbb)
		.build();
	
	auto cfit_xbb_self = dbcsr::tensor<3>::create()
		.set_pgrid(*spgrid3_self)
		.name("cfit_xbb")
		.map1({0})
		.map2({1,2})
		.blk_sizes(xbb)
		.build();
		
	auto cfit_xbb_task = dbcsr::tensor<3>::create()
		.set_pgrid(*spgrid3_self)
		.name("cfit_task")
		.map1({0})
		.map2({1,2})
		.blk_sizes(xbb)
		.build();
	
	auto eri_self = dbcsr::tensor<3>::create()
		.set_pgrid(*spgrid3_self)
		.name("eris_xbb")
		.map1({0})
		.map2({1,2})
		.blk_sizes(xbb)
		.build();
	
	auto cfit_xbb_batched = dbcsr::btensor<3>::create()
		.name("cfit_batched")
		.set_pgrid(spgrid3)
		.blk_sizes(xbb)
		.blk_maps(blkmaps)
		.batch_dims(bdims)
		.btensor_type(dbcsr::btype::core)
		.print(LOG.global_plev())
		.build();
	
	aofac.ao_3c2e_setup(ints::metric::coulomb);
	
	// ==================== LOOP OVER BATCHES, SOLVE QR ================
	
	LOG.os<>("Starting loop over batches...\n");
	
	int off = 0;
	
	cfit_xbb_batched->compress_init({1}, {0}, {1,2});
	
	for (int ibatch_mu = 0; ibatch_mu != cfit_xbb_batched->nbatches(1); ++ibatch_mu) {
		
		LOG.os<>("BATCH ", ibatch_mu, '\n');
		
		auto bbounds = cfit_xbb_batched->blk_bounds(1,ibatch_mu);
		int nblk = bbounds[1] - bbounds[0] + 1;
		
		int64_t ntasks = (int64_t)nblk * (int64_t)b.size();
		
		std::function<void(int64_t)> task_function = [&](int64_t itask) {
			
			arrvec<int,3> res;
			
			int imu = (itask + off) / b.size();
			int inu = (itask + off) % b.size();
			
			std::cout << "TASK " << itask << " MU/NU: " << imu << " " << inu << '\n';
			
			for (int ix = 0; ix != x.size(); ++ix) {
				res[0].push_back(ix);
				res[1].push_back(imu);
				res[2].push_back(inu);
			}
			
			eri_self->reserve(res);
			
			aofac.ao_3c_fill(eri_self);
			
			int locnblks = b[imu] * b[inu];
			int mstride = b[imu];
			
			Eigen::MatrixXd eris_eigen = Eigen::MatrixXd::Zero(nx, locnblks);
			
			int qoff = 0;
			
			for (int ix = 0; ix != x.size(); ++ix) {
				
				std::array<int,3> idx = {ix,imu,inu};
				std::array<int,3> size = {x[ix], b[imu], b[inu]};
				
				bool found = false;
				
				auto blk3 = eri_self->get_block(idx, size, found);
				
				if (!found) {
					qoff += x[ix];
					continue;
				}
				
				for (int qq = 0; qq != size[0]; ++qq) {
					for (int mm = 0; mm != size[1]; ++mm) {
						for (int nn = 0; nn != size[2]; ++nn) {
							eris_eigen(qq + qoff, mm + (nn)*mstride)
								= blk3(qq,mm,nn);
				}}}
				
				qoff += x[ix];
				
			}
			
			double local_work_query = 0;
			double* local_work = nullptr;
			int local_lwork = 0;
			int local_info = 0;
		
			// query
			
			c_dormqr('L', 'T', nx, locnblks, nr, mscala_XR_self.data(), 
				nx, tau_self.data(), eris_eigen.data(), nx, &local_work_query, 
				-1, &local_info);
				
			// allocate
			
			local_lwork = (int)local_work_query;
			local_work = new double[local_lwork];
			
			// do computation
			
			c_dormqr('L', 'T', nx, locnblks, nr, mscala_XR_self.data(), 
				nx, tau_self.data(), eris_eigen.data(), nx, local_work, 
				local_lwork, &local_info);
				
			c_dtrtrs('U', 'N', 'N', nr, locnblks, mscala_XR_self.data(),
				nx, eris_eigen.data(), nx, &local_info);
				
			// transfer
			
			qoff = 0;
			
			cfit_xbb_task->reserve(res);
			
			for (int ir = 0; ir != r.size(); ++ir) {
					
				int ix = rx[ir];
					
				std::array<int,3> idx = {ix, imu, inu};
				std::array<int,3> size = {x[ix], b[imu], b[inu]};
				
				dbcsr::block<3,double> blk(size);
				
				for (int pp = 0; pp != size[0]; ++pp) {
					for (int mm = 0; mm != size[1]; ++mm) {
						for (int nn = 0; nn != size[2]; ++nn) {
							blk(pp,mm,nn) = eris_eigen(pp + qoff, 
								mm + (nn)*mstride);
				}}}
				
				cfit_xbb_task->put_block(idx, blk);
				qoff += x[ix];
			}
			
			cfit_xbb_task->filter(dbcsr::global::filter_eps);
			dbcsr::copy(*cfit_xbb_task, *cfit_xbb_self)
				.sum(true)
				.move_data(true)
				.perform();
			
			eri_self->clear();
			
		};
		
		util::basic_scheduler task_master(m_world.comm(), ntasks, task_function);
		task_master.run();
		
		off += ntasks;
		
		dbcsr::copy_local_to_global(*cfit_xbb_self, *cfit_xbb);
				
		cfit_xbb_batched->compress({ibatch_mu}, cfit_xbb);
		
		cfit_xbb_self->clear();
		cfit_xbb->clear();
		
	}
			
	cfit_xbb_batched->compress_finalize();	
			
	return std::make_tuple(metric_out, cfit_xbb_batched);
	
}
		
void adcmod::init()
{
	
	LOG.banner("ADC MODULE",50,'*');
	
	m_wfn->mol->set_cluster_dfbasis(m_df_basis);
		
	dbcsr::btype btype_e = dbcsr::get_btype(m_eris);
		
	dbcsr::btype btype_i = dbcsr::get_btype(m_imeds);
	
	m_aoloader = ints::aoloader::create()
		.set_world(m_world)
		.set_molecule(m_wfn->mol)
		.print(LOG.global_plev())
		.nbatches_b(m_nbatches_b)
		.nbatches_x(m_nbatches_x)
		.btype_eris(btype_e)
		.btype_intermeds(btype_i)
		.build();
	
	m_adcmethod = str_to_adcmethod(m_method);
	
	init_ao_tensors();
	
	LOG.os<>("--- Ready for launching computation. --- \n\n");
	
}

void adcmod::init_ao_tensors() {
	
	LOG.os<>("Setting up AO integral tensors.\n");
	
	switch (m_adcmethod) {
		case adcmethod::ri_ao_adc1: 
		{
			auto jmet_adc1 = fock::str_to_jmethod(m_build_J);
			auto kmet_adc1 = fock::str_to_kmethod(m_build_K);
			auto metr_adc1 = ints::str_to_metric(m_df_metric);
			
			fock::load_jints(jmet_adc1, metr_adc1, *m_aoloader);
			fock::load_kints(kmet_adc1, metr_adc1, *m_aoloader);
		
			break;
		}
		case adcmethod::sos_cd_ri_adc2:
		{
			auto jmet_adc2 = fock::str_to_jmethod(m_build_J);
			auto kmet_adc2 = fock::str_to_kmethod(m_build_K);
			auto zmet_adc2 = mp::str_to_zmethod(m_build_Z);
			auto metr_adc2 = ints::str_to_metric(m_df_metric);
		
			fock::load_jints(jmet_adc2, metr_adc2, *m_aoloader);
			fock::load_kints(kmet_adc2, metr_adc2, *m_aoloader);
			
			break;
		}
	}
	
	m_aoloader->request(ints::key::ovlp_bb, true);

	m_aoloader->compute();
	
	int natoms = m_wfn->mol->atoms().size();
	std::vector<bool> atidx(natoms, true);
	//std::iota(atidx.begin(), atidx.end(), 0);
	
	//m_fit = test_fitting(atidx);
	
}

std::shared_ptr<MVP> adcmod::create_adc1() {
	
	dbcsr::shared_matrix<double> v_xx;
	dbcsr::sbtensor<3,double> eri3c2e, fitting;
	
	auto jmeth = fock::str_to_jmethod(m_build_J);
	auto kmeth = fock::str_to_kmethod(m_build_K);
	auto metr = ints::str_to_metric(m_df_metric);
		
	auto aoreg = m_aoloader->get_registry();
	
	auto get = [&aoreg](auto& tensor, ints::key aokey) {
		if (aoreg.present(aokey)) {
			tensor = aoreg.get<typename std::remove_reference<decltype(tensor)>::type>(aokey);
		} else {
			//std::cout << "NOT PRESENT :" << static_cast<int>(aokey) << std::endl;
			tensor = nullptr;
		}
	};
	
	switch (metr) {
		case ints::metric::coulomb:
		{
			get(eri3c2e, ints::key::coul_xbb);
			get(fitting, ints::key::dfit_coul_xbb);
			get(v_xx, ints::key::coul_xx_inv);
			break;
		}
		case ints::metric::erfc_coulomb:
		{
			get(eri3c2e, ints::key::erfc_xbb);
			get(fitting, ints::key::dfit_erfc_xbb);
			get(v_xx, ints::key::erfc_xx_inv);
			break;
		}
		case ints::metric::pari:
		{
			get(eri3c2e, ints::key::pari_xbb);
			if (!eri3c2e) std::cout << "NULL" << std::endl;
			get(fitting, ints::key::dfit_pari_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
		case ints::metric::qr_fit:
		{
			get(eri3c2e, ints::key::qr_xbb);
			get(fitting, ints::key::dfit_qr_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
	}
	
	/*if (m_local) {
		
		LOG.os<>("Using a cutoff of ", m_cutoff, '\n');
		
		auto atom_idx = get_significant_blocks(m_wfn->adc_wfn->davidson_eigenvectors()[0],m_cutoff);
		
		int natoms = 0;
		
		for (auto b : atom_idx) {
			if (b) natoms++;
		}
		
		LOG.os<>("Taking ", natoms, " atoms of ", m_wfn->mol->atoms().size(), '\n');
		
		auto [vlocal, clocal] = test_fitting(atom_idx);
		
		//fitting = eri3c2e;
		eri3c2e = clocal; //clocal;
		v_xx = vlocal;
		
		v_xx->filter(dbcsr::global::filter_eps);
		
		ints::dfitting fit(m_world, m_wfn->mol, 1);
		
		fitting = fit.compute(eri3c2e, v_xx, dbcsr::btype::core);
		
		eri3c2e->print_info();
		fitting->print_info();
		
	}*/
		
	auto ptr = MVP_AORIADC1::create()
		.set_world(m_world)
		.set_molecule(m_wfn->mol)
		.print(LOG.global_plev())
		.c_bo(m_wfn->hf_wfn->c_bo_A())
		.c_bv(m_wfn->hf_wfn->c_bv_A())
		.eps_occ(*m_wfn->hf_wfn->eps_occ_A())
		.eps_vir(*m_wfn->hf_wfn->eps_vir_A())
		.eri3c2e_batched(eri3c2e)
		.fitting_batched(fitting)
		.metric_inv(v_xx)
		.jmethod(jmeth)
		.kmethod(kmeth)
		.build();
		
	ptr->init();
	
	return ptr;
	
}

std::shared_ptr<MVP> adcmod::create_adc2(std::optional<canon_lmo> clmo) {
	
	desc::shared_molecule mol;
	std::shared_ptr<std::vector<double>> eps_o, eps_v;
	dbcsr::shared_matrix<double> v_xx, s_bb, c_bo, c_bv;
	dbcsr::sbtensor<3,double> eri3c2e, fitting;
	
	auto jmeth = fock::str_to_jmethod(m_build_J);
	auto kmeth = fock::str_to_kmethod(m_build_K);
	auto zmeth = mp::str_to_zmethod(m_build_Z);
	
	auto itype = dbcsr::get_btype(m_imeds);
	
	auto metr = ints::str_to_metric(m_df_metric);
	auto aoreg = m_aoloader->get_registry();
	
	auto get = [&aoreg](auto& tensor, ints::key aokey) {
		if (aoreg.present(aokey)) {
			tensor = aoreg.get<typename std::remove_reference<decltype(tensor)>::type>(aokey);
		} else {
			tensor = nullptr;
		}
	};
	
	switch (metr) {
		case ints::metric::coulomb:
		{
			get(eri3c2e, ints::key::coul_xbb);
			get(fitting, ints::key::dfit_coul_xbb);
			get(v_xx, ints::key::coul_xx_inv);
			break;
		}
		case ints::metric::erfc_coulomb:
		{
			get(eri3c2e, ints::key::erfc_xbb);
			get(fitting, ints::key::dfit_erfc_xbb);
			get(v_xx, ints::key::erfc_xx_inv);
			break;
		}
		case ints::metric::pari:
		{
			get(eri3c2e, ints::key::pari_xbb);
			get(fitting, ints::key::dfit_pari_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
		case ints::metric::qr_fit:
		{
			get(eri3c2e, ints::key::qr_xbb);
			get(fitting, ints::key::dfit_qr_xbb);
			get(v_xx, ints::key::coul_xx);
			break;
		}
	}
	
	s_bb = aoreg.get<dbcsr::shared_matrix<double>>(ints::key::ovlp_bb);
	
	/*if (clmo) {
		
		int natoms = m_hfwfn->mol()->atoms().size();
	
		std::vector<int> iat(natoms, 0);
		std::iota(iat.begin(), iat.end(), 0);
	
		int noa = clmo->c_br->nfullcols_total();
		int nva = clmo->c_bs->nfullcols_total();
	
		mol = m_hfwfn->mol()->fragment(noa, noa, nva, nva, iat);
		
		c_bo = clmo->c_br;
		c_bv = clmo->c_bs;
		
		eps_o = std::make_shared<std::vector<double>>(clmo->eps_r);
		eps_v = std::make_shared<std::vector<double>>(clmo->eps_s);
		
	} else {*/
		
	mol = m_wfn->mol;
	c_bo = m_wfn->hf_wfn->c_bo_A();
	c_bv = m_wfn->hf_wfn->c_bv_A();
	eps_o = m_wfn->hf_wfn->eps_occ_A();
	eps_v = m_wfn->hf_wfn->eps_vir_A();

	auto ptr = MVP_AORISOSADC2::create()
		.set_world(m_world)
		.set_molecule(mol)
		.print(LOG.global_plev())
		.c_bo(c_bo)
		.c_bv(c_bv)
		.s_bb(s_bb)
		.eps_occ(*eps_o)
		.eps_vir(*eps_v)
		.eri3c2e_batched(eri3c2e)
		.fitting_batched(fitting)
		.metric_inv(v_xx)
		.jmethod(jmeth)
		.kmethod(kmeth)
		.zmethod(zmeth)
		.btype(itype)
		.nlap(m_nlap)
		.c_os(m_c_os)
		.c_os_coupling(m_c_os_coupling)
		.build();
				
	ptr->init();
	
	return ptr;
	
}

} // end namespace

} // end namespace mega
