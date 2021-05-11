#include "locorb/locorb.hpp"
#include "math/linalg/LLT.hpp"
#include "math/linalg/SVD.hpp"
#include "math/solvers/hermitian_eigen_solver.hpp"
#include "utils/matrix_plot.hpp"

namespace megalochem {

namespace locorb {

std::tuple<smat_d,smat_d> mo_localizer::compute_pao(smat_d c_bm, smat_d s_bb) {
	
	auto l_bb = dbcsr::matrix<>::create_template(*s_bb)
		.name("u_bb")
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto u_bm = dbcsr::matrix<>::create_template(*c_bm)
		.name("u_bm")
		.build();
		
	dbcsr::multiply('N', 'N', 1.0, *s_bb, *c_bm, 0.0, *u_bm).perform();
	dbcsr::multiply('N', 'T', 1.0, *c_bm, *u_bm, 0.0, *l_bb).perform();
	
	return std::make_tuple(l_bb, u_bm);
	
} 

std::tuple<smat_d, smat_d, std::vector<double>> 
	mo_localizer::compute_truncated_pao(
	smat_d c_bm, smat_d s_bb, std::vector<double> eps_m,
	std::vector<int> blkidx,
	smat_d u_km)
{
	
	/*
	 * b : atomic orbitals
	 * p : truncated atomic orbitals
	 * m : occupied or virtual (o,v)
	 * r : truncated non-canonical o/v
	 * s : truncated canonical o/v
	 * 
	 */
	 
	LOG.os<>("PAO 1\n");
	
	// === Compute overlap matrix inverse of truncated AOs
	
	auto b = c_bm->row_blk_sizes();
	auto m = c_bm->col_blk_sizes();
	//auto k = u_km->row_blk_sizes();
	auto boff = c_bm->row_blk_offsets();
	
	std::vector<int> p, poff; // projected space
	
	for (auto iblk : blkidx) {
		p.push_back(b[iblk]);
	}
	
	poff = p;
	int off = 0;
	for (auto& i : poff) {
		int size = i;
		i = off;
		off += size;
	}
	
	auto s_pp = dbcsr::matrix<>::create()
		.name("s_pp")
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(p)
		.col_blk_sizes(p)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	auto s_pb = dbcsr::matrix<>::create()
		.name("s_pb")
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(p)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	s_pp->reserve_all();
	s_pb->reserve_all();
	
	auto s_nosym = s_bb->desymmetrize();
	s_nosym->replicate_all();
	
	int np = std::accumulate(p.begin(), p.end(), 0);
	int nb = std::accumulate(b.begin(), b.end(), 0);
	
	LOG.os<>("PAO 2\n");
	
	dbcsr::iterator iterpp(*s_pp);
	iterpp.start();
	
	while(iterpp.blocks_left()) {
		
		iterpp.next_block();
		int iblk = blkidx[iterpp.row()];
		int jblk = blkidx[iterpp.col()];
		
		bool found = false;
		auto blk = s_nosym->get_block_p(iblk,jblk,found);
		if (!found) continue;
		
		std::copy(blk.data(), blk.data() + blk.ntot(), &iterpp(0,0));
		
	}
	
	iterpp.stop();
	
	LOG.os<>("PAO 3\n");
	
	dbcsr::iterator iterpb(*s_pb);
	iterpb.start(); 
	
	while(iterpb.blocks_left()) {
		
		iterpb.next_block();
		int iblk = blkidx[iterpb.row()];
		int jblk = iterpb.col();
		
		bool found = false;
		auto blk = s_nosym->get_block_p(iblk,jblk,found);
		if (!found) continue;
		
		std::copy(blk.data(), blk.data() + blk.ntot(), &iterpb(0,0));
		
	}
	
	iterpb.stop();
			
	math::LLT llt(m_world, s_pp, LOG.global_plev());
	llt.compute();
	
	math::LLT lltfull(m_world, s_bb, LOG.global_plev());
	lltfull.compute();
	
	auto s_sqrt_pp = llt.L(p);
	auto s_invsqrt_pp = llt.L_inv(p);
	auto s_inv_pp = llt.inverse(p);
	
	auto s_sqrt_bb = lltfull.L(b);
	
	//dbcsr::print(*s_inv_pp);
	
	LOG.os<>("PAO 4\n");
	
	auto cortho_bm = dbcsr::matrix<>::create_template(*c_bm)
		.name("c_ortho")
		.build();
	
	auto q_pm = dbcsr::matrix<>::create()
		.name("q_pm")
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(p)
		.col_blk_sizes(m)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto temp_pb = dbcsr::matrix<>::create()
		.name("temp_pb")
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(p)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	LOG.os<>("PAO 5\n");
	
	dbcsr::multiply('N', 'N', 1.0, *s_inv_pp, *s_pb, 0.0, *temp_pb)
		.perform();
	
	dbcsr::multiply('N', 'N', 1.0, *s_sqrt_bb, *c_bm, 0.0, *cortho_bm)
		.perform();
	
	dbcsr::multiply('N', 'N', 1.0, *temp_pb, *cortho_bm, 0.0, *q_pm)
		.perform();
	
	//dbcsr::print(*q_pm);
	
	math::SVD svd(m_world, q_pm, 'V', 'V', LOG.global_plev());
	svd.compute();
	
	auto s = svd.s();
	int rank = s.size();
	
	LOG.os<>("SING VALS:\n");
	for (auto val : s) {
		LOG.os<>(val, " ");
	} LOG.os<>('\n');
	
	LOG.os<>("RANK: ", rank, '\n');
	
	auto r = dbcsr::split_range(rank, m_mol->mo_split());
	
	auto u_pr = svd.U(p,r);
	
	auto vt_rm = svd.Vt(r,m);
	
	//vt_rm->filter(1e-6);
	//dbcsr::print(*vt_rm);
	
	// canonicalize
	
	auto f_mm = dbcsr::matrix<>::create()
		.name("Fock")
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(m)
		.col_blk_sizes(m)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto f_ht_rm = dbcsr::matrix<>::create()
		.name("Fock_HT")
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(r)
		.col_blk_sizes(m)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto f_rr = dbcsr::matrix<>::create()
		.name("Fock_HT")
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(r)
		.col_blk_sizes(r)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	f_mm->reserve_diag_blocks();
	f_mm->set_diag(eps_m);
	
	dbcsr::multiply('N', 'N', 1.0, *vt_rm, *f_mm, 0.0, *f_ht_rm)
		.perform();
		
	dbcsr::multiply('N', 'T', 1.0, *f_ht_rm, *vt_rm, 0.0, *f_rr)
		.perform();
		
	math::hermitian_eigen_solver hermsolver(m_world, f_rr, 'V', LOG.global_plev());
	hermsolver.compute();
	
	// new molecular energies
	auto eps_s = hermsolver.eigvals();
	
	if (m_world.rank() == 0) {
		std::cout << "Old molecular energies: " << std::endl;
		for (auto e : eps_m) {
			std::cout << e << " ";
		} std::cout << std::endl;
		std::cout << "New molecular energies: " << std::endl;
		for (auto e : eps_s) {
			std::cout << e << " ";
		} std::cout << std::endl;
	}
	
	// transformation matrix noncanon -> canon
	auto T_rs = hermsolver.eigvecs();
	
	// compute MO -> trunc. canonical MO transformation matrix
	// T_sm = T_rs^t Σ_r Vt_rm
	
	auto T_ms = dbcsr::matrix<>::create()
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(m)
		.col_blk_sizes(r)
		.name("Transformation matrix canon. MOs -> canon. truncated MOs")
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	dbcsr::multiply('T', 'N', 1.0, *vt_rm, *T_rs, 0.0, *T_ms)
		.perform();
		
	// new MO coefficient matrix:
	
	auto c_bs = dbcsr::matrix<>::create()
		.name(c_bm->name() + "_truncated")
		.set_cart(m_world.dbcsr_grid())
		.row_blk_sizes(b)
		.col_blk_sizes(r)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	dbcsr::multiply('N', 'N', 1.0, *c_bm, *T_ms, 0.0, *c_bs).perform();
	
	// compute truncated canonical MO coefficient matrix
	// Ctrunc_ps = X_pp * U_pr * T_rs
	
	/*
	auto ctrunc_ortho_ps = dbcsr::create_template<double>(*u_pr)
		.name("Truncated orthogonal coefficient matrix")
		.build();
		
	auto ctrunc_ps = dbcsr::create_template<double>(*u_pr)
		.name("Truncated coefficient matrix")
		.build();
	
	u_pr->scale(s, "right");
	
	dbcsr::multiply('N', 'N', *u_pr, *T_rs, *ctrunc_ortho_ps)
		.perform();
		
	dbcsr::multiply('N', 'N', *s_invsqrt_pp, *ctrunc_ortho_ps, *ctrunc_ps)
		.perform();*/
		
	/*auto p_bb = dbcsr::create<double>()
		.name("p")
		.set_cart(m_cart)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto p_pp = dbcsr::create<double>()
		.name("p")
		.set_cart(m_cart)
		.row_blk_sizes(p)
		.col_blk_sizes(p)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
	
	dbcsr::multiply('N', 'T', *c_bm, *c_bm, *p_bb)
		.perform();
		
	dbcsr::multiply('N', 'T', *ctrunc_ps, *ctrunc_ps, *p_bb)
		.alpha(-1.0)
		.beta(1.0)
		.perform();
		
	p_bb->filter(1e-6);
	dbcsr::print(*p_bb);
	
	//LOG.os<>("POP: ", pop_mulliken(*ctrunc_ps, *s_pp, *spp), '\n');*/
		
	return std::make_tuple(c_bs, T_ms, eps_s);
	
	
}
/*
double mo_localizer::pop_mulliken(dbcsr::matrix<double>& c_bo, 
	dbcsr::matrix<double>& s_bb, dbcsr::matrix<double>& s_sqrt_bb) 
{
	auto w = c_bo.get_cart();
	auto b = c_bo.row_blk_sizes();
	
	auto p_bb = dbcsr::create<double>()
		.name("p_bb")
		.set_cart(w)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	auto pop_bb = dbcsr::create<double>()
		.name("pop_bb")
		.set_cart(w)
		.row_blk_sizes(b)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.build();
		
	dbcsr::multiply('N', 'T', c_bo, c_bo, *p_bb)
		.perform();
		
	dbcsr::multiply('N', 'N', *p_bb, s_bb, *pop_bb)
		.perform();
		
	long long int datasize;
	double* ptr = pop_bb->data(datasize);
	
	double nele_loc = std::accumulate(ptr, ptr + datasize, 0.0);
	
	MPI_Allreduce(MPI_IN_PLACE, &nele_loc, 1, MPI_DOUBLE, MPI_SUM, w.comm());
	
	return nele_loc;
	
}*/

} // end namespace

} // end namespace megalochem
