#include "ints/fitting.h"
#include "utils/constants.h"

namespace ints {

dbcsr::sbtensor<3,double> dfitting::compute(dbcsr::sbtensor<3,double> eri_batched, 
	dbcsr::shared_matrix<double> inv, dbcsr::btype mytype) {
		
	auto spgrid2 = dbcsr::create_pgrid<2>(m_world.comm()).get();
	auto x = m_mol->dims().x();
	
	arrvec<int,2> xx = {x,x};

	auto s_xx_inv = dbcsr::tensor_create<2>()
		.name("s_xx_inv")
		.pgrid(spgrid2)
		.blk_sizes(xx)
		.map1({0}).map2({1})
		.get();
		
	dbcsr::copy_matrix_to_tensor(*inv, *s_xx_inv);
	inv->clear();
	
	auto cfit = this->compute(eri_batched, s_xx_inv, mytype);
	dbcsr::copy_tensor_to_matrix(*s_xx_inv, *inv);
	
	return cfit;
	
}
	
dbcsr::sbtensor<3,double> dfitting::compute(dbcsr::sbtensor<3,double> eri_batched, 
	dbcsr::shared_tensor<2,double> inv, dbcsr::btype mytype) {
	
	auto spgrid3_xbb = eri_batched->spgrid();
		
	auto b = m_mol->dims().b();
	auto x = m_mol->dims().x();
	
	arrvec<int,3> xbb = {x,b,b};
	
	// ======== Compute inv_xx * i_xxb ==============
	
	auto print = [](auto v) {
		for (auto e : v) {
			std::cout << e << " ";
		} std::cout << std::endl;
	};
	
	auto c_xbb_0_12 = dbcsr::tensor_create<3>()
		.name("c_xbb_0_12")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
	
	auto c_xbb_1_02 = dbcsr::tensor_create_template<3>(c_xbb_0_12)
		.name("c_xbb_1_02")
		.map1({1}).map2({0,2})
		.get();
			
	int nbatches_x = eri_batched->nbatches(0);
	int nbatches_b = eri_batched->nbatches(2);

	std::cout << "NBATCHES: " << nbatches_x << " " << nbatches_b << std::endl;

	std::array<int,3> bdims = {nbatches_x,nbatches_b,nbatches_b};

	auto blkmap_b = m_mol->c_basis()->block_to_atom(m_mol->atoms());
	auto blkmap_x = m_mol->c_dfbasis()->block_to_atom(m_mol->atoms());
		
	arrvec<int,3> blkmaps = {blkmap_x, blkmap_b, blkmap_b};
	
	auto c_xbb_batched = dbcsr::btensor_create<3>()
		.name(m_mol->name() + "_c_xbb_batched")
		.pgrid(spgrid3_xbb)
		.blk_sizes(xbb)
		.batch_dims(bdims)
		.blk_map(blkmaps)
		.btensor_type(mytype)
		.print(LOG.global_plev())
		.get();
	
	auto& con = TIME.sub("Contraction");
	auto& reo = TIME.sub("Reordering");
	auto& write = TIME.sub("Writing");
	auto& fetch = TIME.sub("Fetching ints");
	
	LOG.os<1>("Computing C_xbb.\n");
	
	TIME.start();
	
	eri_batched->decompress_init({2},vec<int>{0},vec<int>{1,2});
	c_xbb_batched->compress_init({2,0}, vec<int>{1}, vec<int>{0,2});
	//c_xbb_0_12->batched_contract_init();
	//c_xbb_1_02->batched_contract_init();
	
	//auto e = eri_batched->get_work_tensor();
	//std::cout << "ERI OCC: " << e->occupation() * 100 << std::endl;
	
	for (int inu = 0; inu != c_xbb_batched->nbatches(2); ++inu) {
		
		fetch.start();
		eri_batched->decompress({inu});
		auto eri_0_12 = eri_batched->get_work_tensor();
		fetch.finish();
		
		//inv->batched_contract_init();
			
		for (int ix = 0; ix != c_xbb_batched->nbatches(0); ++ix) {
			
				vec<vec<int>> b2 = {
					c_xbb_batched->bounds(0,ix)
				};
				
				vec<vec<int>> b3 = {
					c_xbb_batched->full_bounds(1),
					c_xbb_batched->bounds(2,inu)
				};

				con.start();
				dbcsr::contract(*inv, *eri_0_12, *c_xbb_0_12)
					.bounds2(b2).bounds3(b3)
					.filter(dbcsr::global::filter_eps)
					.perform("XY, YMN -> XMN");
				con.finish();
				
				reo.start();
				dbcsr::copy(*c_xbb_0_12, *c_xbb_1_02)
					.move_data(true)
					.perform();
				reo.finish();
					
				c_xbb_batched->compress({inu,ix}, c_xbb_1_02);
				
		}
		
		//inv->batched_contract_finalize();
		
	}
	
	c_xbb_batched->compress_finalize();
	eri_batched->decompress_finalize();
	//c_xbb_0_12->batched_contract_finalize();
	//c_xbb_1_02->batched_contract_finalize();
	
	//auto cw = c_xbb_batched->get_work_tensor();
	//dbcsr::print(*cw);
	
	double cfit_occupation = c_xbb_batched->occupation() * 100;
	LOG.os<1>("Occupancy of c_xbb: ", cfit_occupation, "%\n");
	
	if (cfit_occupation > 100) throw std::runtime_error(
		"Fitting coefficients occupation more than 100%");

	TIME.finish();
	
	LOG.os<1>("Done.\n");
	
	// ========== END ==========	
	
	return c_xbb_batched;
	
}
	
} // end namespace
