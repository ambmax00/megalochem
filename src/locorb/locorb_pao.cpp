#include "locorb/locorb.h"
#include "math/linalg/LLT.h"
#include "math/linalg/SVD.h"
#include "utils/matrix_plot.h"

namespace locorb {

std::pair<smat_d,smat_d> mo_localizer::compute_pao(smat_d c_bm, smat_d s_bb) {
	
	auto l_bb = dbcsr::create_template<double>(*s_bb)
		.name("u_bb")
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto u_bm = dbcsr::create_template<double>(*c_bm)
		.name("u_bm")
		.get();
		
	dbcsr::multiply('N', 'N', *s_bb, *c_bm, *u_bm).perform();
	dbcsr::multiply('N', 'T', *c_bm, *u_bm, *l_bb).perform();
	
	return std::make_pair<smat_d,smat_d>(
		std::move(l_bb), std::move(u_bm)
	);
	
} 

std::tuple<smat_d, smat_d, std::vector<double>> 
	mo_localizer::compute_truncated_pao(
	smat_d c_bm, smat_d s_bb, std::vector<int> blkidx)
{
	
	// === Compute overlap matrix inverse of truncated AOs
	
	auto b = c_bm->row_blk_sizes();
	auto m = c_bm->col_blk_sizes();
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
	
	auto s_pp = dbcsr::create<double>()
		.name("s_pp")
		.set_world(m_world)
		.row_blk_sizes(p)
		.col_blk_sizes(p)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	auto s_pb = dbcsr::create<double>()
		.name("s_pb")
		.set_world(m_world)
		.row_blk_sizes(p)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	s_pp->reserve_all();
	s_pb->reserve_all();
	
	auto s_nosym = s_bb->desymmetrize();
	s_nosym->replicate_all();
	
	int np = std::accumulate(p.begin(), p.end(), 0);
	int nb = std::accumulate(b.begin(), b.end(), 0);
	
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
	
	
	dbcsr::print(*s_pb);
		
	math::LLT llt(s_pp, LOG.global_plev());
	llt.compute();
	
	math::LLT lltfull(s_bb, LOG.global_plev());
	lltfull.compute();
	
	auto s_inv_pp = llt.inverse(p);
	auto s_sqrt_bb = lltfull.L(b);
	
	
	dbcsr::print(*s_inv_pp);
	
	auto cortho_bm = dbcsr::create_template<double>(*c_bm)
		.name("c_ortho")
		.get();
	
	auto q_pm = dbcsr::create<double>()
		.name("q_pm")
		.set_world(m_world)
		.row_blk_sizes(p)
		.col_blk_sizes(m)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto temp_pb = dbcsr::create<double>()
		.name("temp_pb")
		.set_world(m_world)
		.row_blk_sizes(p)
		.col_blk_sizes(b)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
		
	auto temp_pp = dbcsr::create<double>()
		.name("temp_pb")
		.set_world(m_world)
		.row_blk_sizes(p)
		.col_blk_sizes(p)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	dbcsr::multiply('N', 'N', *s_pp, *s_inv_pp, *temp_pp)
		.perform();
	
	std::cout << "HERE" << std::endl;
	dbcsr::print(*temp_pp);
	
	dbcsr::multiply('N', 'N', *s_inv_pp, *s_pb, *temp_pb)
		.perform();
	
	dbcsr::multiply('N', 'N', *s_sqrt_bb, *c_bm, *cortho_bm)
		.perform();
	
	dbcsr::multiply('N', 'N', *temp_pb, *cortho_bm, *q_pm)
		.perform();
	
	dbcsr::print(*q_pm);
	
	math::SVD svd(q_pm, 'V', 'V', LOG.global_plev());
	svd.compute();
	
	auto s = svd.s();
	int rank = s.size();
	
	LOG.os<>("RANK: ", rank, '\n');
	
	auto u = dbcsr::split_range(rank, m_mol->mo_split());
	
	auto c_bu = dbcsr::create<double>()
		.set_world(m_world)
		.name("c_bu")
		.row_blk_sizes(b)
		.col_blk_sizes(u)
		.matrix_type(dbcsr::type::no_symmetry)
		.get();
	
	auto vt_um = svd.Vt(u,m);
	
	dbcsr::multiply('N', 'T', *c_bm, *vt_um, *c_bu)
		.perform();
	
	if (m_world.rank() == 0) {
		for (auto i : s) {
			std::cout << i << " ";
		} std::cout << std::endl;
	}
	
	dbcsr::print(*c_bu);
	
	//util::plot(c_bm, 1e-4);
	//util::plot(c_bu, 1e-4);
	
	MPI_Barrier(m_world.comm());
	
	exit(0);
	
	/*
	dbcsr::print(*s_pp);
	
	
	//s_pp->replicate_all();
	
	dbcsr::iterator iter(*s_bb);
	iter.start();
	
	/*
	while(iter.blocks_left()) {
		
		iter.next_block();
		int irblk = iter.row();
		int icblk = iter.col();
		int rsize = iter.row_size();
		int csize = iter.col_size();
		
		bool found = true;
		auto blkpp = s_pp->get_block_p(irblk,icblk,found);
		
		std::copy(&iter(0,0), &iter(0,0) + rsize * csize, &blkp(0,0));
		
	}
	
	s_pp->sum_replicated();
	s_pp->distribute();
		
	std::vector<int> resrow, rescol;
		
	for (auto iblk : blkidx) {
		for (auto jblk : blkidx) {
			if (strunc_bb->proc(iblk,jblk) == m_world.rank()) {
				resrow.push_back(iblk);
				rescol.push_back(jblk);
			}
		}
	}
	
	strunc_bb->reserve_blocks(resrow, rescol);
	strunc_bb->copy_in(*s_bb, true);
	
	// invert
	math::LLT llt(strunc_bb, LOG.global_plev());
	llt.compute();
	
	auto b = m_mol->dims().b();
	auto strunc_inv_bb = llt.inverse(b);
	
	strunc_bb->clear();
	resrow.clear();
	rescol.clear();
	
	// === compute AO-truncAO overlap ===
	// reuse strunc_bb
	
	for (auto iblk : blkidx) {
		for (int jblk = 0; jblk != b.size(); ++jblk) {
			if (strunc_bb->proc(iblk,jblk) == m_world.rank()) {
				resrow.push_back(iblk);
				rescol.push_back(jblk);
			}
		}
	}
	
	strunc_bb->reserve_blocks(resrow, rescol);
	strunc_bb->copy_in(*s_bb, true);
	
	// === compute truncated MO coefficients ===
	
	auto temp = dbcsr::create_template<double>(*c_bm)
		.name("temp")
		.get();
		
	auto ctrunc_bm = dbcsr::create_template<double>(*c_bm)
		.name("ctrunc")
		.get();
		
	dbcsr::multiply('N', 'N', *strunc_bb, *c_bm, *temp)
		.perform();
	dbcsr::multiply('N', 'N', *strunc_inv_bb, *temp, *ctrunc_bm)
		.perform();
	
	auto utrunc = compute_conversion(c_bm, strunc_bb, ctrunc_bm);
	
	std::vector<double> out = {};
	
	return std::make_tuple(ctrunc_bm, utrunc, out);*/
	
	
}

} // end namespace
