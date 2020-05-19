#include <mpi.h>
#include <random>
#include <stdexcept>
#include <string>
#include <dbcsr_matrix.hpp>
#include <dbcsr_conversions.hpp>
#include <dbcsr_matrix_ops.hpp>
#include "input/reader.h"
#include "hf/hfmod.h"
//#include "adc/adcmod.h"
#include "utils/mpi_time.h"

#ifdef USE_SCALAPACK
#include "extern/scalapack.h"
#endif

//#include "ints/aofactory.h"
//#include "math/solvers/hermitian_eigen_solver.h"

template <int N>
void fill_random(dbcsr::tensor<N,double>& t_in, arrvec<int,N>& nz) {
	
	int myrank, mpi_size;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	if (myrank == 0) std::cout << "Filling Tensor..." << std::endl;
	if (myrank == 0) std::cout << "Dimension: " << N << std::endl;
	
	int nblocks = nz[0].size();
	arrvec<int,N> mynz;
	dbcsr::index<N> idx;
	
	for (int i = 0; i != nblocks; ++i) {
		
		// make index out of nzblocks
		for (int j = 0; j != N; ++j) idx[j] = nz[j][i];
		
		int proc = t_in.proc(idx);
		
		if (proc == myrank) {
			for (int j = 0; j != N; ++j) 
				mynz[j].push_back(idx[j]);
		}		
		
	} 
	
	for (int i = 0; i != mpi_size; ++i) {
		
		if (myrank == i) {
			
			std::cout << "process: " << i << std::endl;
		
			for (auto x : mynz ) {
				for (auto y : x) {
					std::cout << y << " ";
				} std::cout << std::endl;
			}
			
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
	}
	
	t_in.reserve(mynz);
	
	#pragma omp parallel 
	{
		dbcsr::iterator_t<N,double> iter(t_in);
		
		while (iter.blocks_left()) {
			  
			iter.next();
			
			dbcsr::block<N,double> blk(iter.size());
			
			blk.fill_rand(-1.0,1.0);
			
			auto idx = iter.idx();
			t_in.put_block(idx, blk);
				
		}
	}
	
	t_in.finalize();

}


int main(int argc, char** argv) {

	MPI_Init(&argc, &argv);
	
	if (argc < 2) {
		throw std::runtime_error("Wrong number of command line inputs.");
	}
	
	util::mpi_time time(MPI_COMM_WORLD, "Megalochem");
	
	util::mpi_log LOG(MPI_COMM_WORLD,0);
	
	std::string s = R"(|  \/  ||  ___|  __ \ / _ \ | |   |  _  /  __ \| | | ||  ___|  \/  |)""\n"
					R"(| .  . || |__ | |  \// /_\ \| |   | | | | /  \/| |_| || |__ | .  . |)""\n"
					R"(| |\/| ||  __|| | __ |  _  || |   | | | | |    |  _  ||  __|| |\/| |)""\n"
					R"(| |  | || |___| |_\ \| | | || |___\ \_/ / \__/\| | | || |___| |  | |)""\n"
					R"(\_|  |_/\____/ \____/\_| |_/\_____/\___/ \____/\_| |_/\____/\_|  |_/)";
	
	
	LOG.banner(s,90,'*');
	
	std::string filename(argv[1]);
	
	time.start();
	
	dbcsr::init();
	
	dbcsr::world wrd(MPI_COMM_WORLD);
	
#ifdef USE_SCALAPACK
	int sysctxt = -1;
	c_blacs_get(0, 0, &sysctxt);
	int gridctxt = sysctxt;
	c_blacs_gridinit(&gridctxt, 'R', wrd.nprow(), wrd.npcol());
	std::cout << "HEREEE: " << wrd.nprow() << " " << wrd.npcol() << std::endl;
	scalapack::global_grid.set(gridctxt);
#endif
	
	reader filereader(MPI_COMM_WORLD, filename);
	
	auto mol = std::make_shared<desc::molecule>(filereader.get_mol());
	auto opt = filereader.get_opt();
	
	auto hfopt = opt.subtext("hf");
	
	desc::shf_wfn myhfwfn = std::make_shared<desc::hf_wfn>();
	hf::hfmod myhf(mol,hfopt,wrd);
	
	bool skip_hf = hfopt.get<bool>("skip", false);
	
	if (!skip_hf) {
	
		myhf.compute();
		myhfwfn = myhf.wfn();
		myhfwfn->write_to_file();
		
	} else {
		
		LOG.os<>("Reading HF info from files...\n");
		myhfwfn->read_from_file(mol,wrd);
		LOG.os<>("Done.\n");
		
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	/*
	
	auto adcopt = opt.subtext("adc");
	
	std::cout << adcopt.get<int>("nroots") << std::endl;
	
	adc::adcmod myadc(myhfwfn,adcopt,MPI_COMM_WORLD);
	
	myadc.compute();
	
	time.finish();
	
	time.print_info();
	*/
	/*
	//ints::aofactory ao(mol, MPI_COMM_WORLD);
	
	//auto s = ao.compute<2>({.op = "overlap", .bas = "bb", .name = "S", .map1 = {0}, .map2 = {1}});
	
	//dbcsr::print(s);
	
	
	dbcsr::pgrid<3> pgrid3d(MPI_COMM_WORLD);
	dbcsr::pgrid<4> pgrid4d(MPI_COMM_WORLD);
							 
	vec<int> blk1 = {3, 9, 12, 1};
    vec<int> blk2 = {4, 2, 3, 1, 9, 2, 32, 10, 5, 8, 7};
    vec<int> blk3 = {7, 3, 8, 7, 9, 5, 10, 23, 2};
    vec<int> blk4 = {8, 1, 4, 13, 6};
    vec<int> blk5 = {4, 2, 22};
    
	vec<int> map11 = {0, 2};
	vec<int> map12 = {1};
	vec<int> map21 = {3, 2};
	vec<int> map22 = {1, 0};
	vec<int> map31 = {0};
	vec<int> map32 = {1, 2};
	
	arrvec<int,3> sizes1 = {blk1,blk2,blk3}; 
	arrvec<int,4> sizes2 = {blk1,blk2,blk4,blk5};
	arrvec<int,3> sizes3 = {blk3,blk4,blk5};
	
	auto tensor1 = dbcsr::make_stensor<2>(dbcsr::tensor<3>::create().name("(13|2)")
		.ngrid(pgrid3d).map1(map11).map2(map12).blk_sizes(sizes1).get_stensor());
		
	dbcsr::tensor<4> tensor2 = dbcsr::tensor<4>::create().name("(54|21)")
		.ngrid(pgrid4d).map1(map21).map2(map22).blk_sizes(sizes2);
		
	dbcsr::tensor<3> tensor3 = dbcsr::tensor<3>::create().name("(3|45)")
		.ngrid(pgrid3d).map1(map31).map2(map32).blk_sizes(sizes3);
	
	/*
	//dbcsr::tensor<3,double> tensortest({.name="test", .pgridN = pgrid3d, .map1 = map11,
	//	.map2 = map12, .blk_sizes = sizes1});
						   
	std::cout << "Tensors created." << std::endl;
	
	vec<int> nz11 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		    0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 
		    2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 
		    3, 3};
	vec<int> nz12 = {2, 4, 4, 4, 5, 5, 6, 7, 9,10,10, 
		    0, 0, 3, 6, 6, 8, 9 ,1, 1, 4, 5, 
		    7, 7, 8,10,10, 1 ,3, 4, 4, 7};
    vec<int> nz13 = {6, 2, 4, 8, 5, 7, 1, 7, 2, 1, 2, 
			0, 3, 5, 1, 6, 4, 7, 2, 6, 0, 3, 
			2, 6, 7, 4, 7, 8, 5, 0, 1, 6};
	
	vec<int> nz21 = { 0, 0, 0, 0, 0, 1, 1, 1,  1,  1, 
             1, 1, 1, 1, 1, 1, 1, 1,  1,  1, 
             2, 2, 2, 2, 2, 2, 2, 2,  2,  2, 
             3, 3, 3, 3, 3, 3 };
    vec<int> nz22 = { 0, 2, 3, 5, 9,  1, 1, 3,  4,  4, 
             5, 5, 5, 6,  6,  8, 8, 8, 9, 10, 
             0, 2, 2, 3,  4,  5, 7, 8, 10, 10, 
             0, 2, 3, 5, 9, 10 };
	vec<int> nz24 = { 2, 4, 1, 2,  1,  2, 4, 0,  0,  3, 
             1, 2, 3, 0,  3,  2, 3, 3,  1,  0, 
             2, 0, 0, 2,  3,  2, 3, 1,  1,  2, 
             0, 0, 2, 1,  4,  4 };
    vec<int> nz25 = { 0, 2, 1, 0,  0,  1, 2,  0,  2, 0, 
             1, 2, 1, 0,  2,  1, 2,  1,  0, 1, 
             2, 0, 1, 2,  1,  1, 1,  2,  0, 1, 
             0, 2, 1, 0,  2,  1 };		
             
    vec<int> nz33 = { 1, 3, 4, 4, 4, 5, 5, 7 };
    vec<int> nz34 = { 2, 1, 0, 0, 2, 1, 3, 4 };
    vec<int> nz35 = { 2, 1, 0, 1, 2, 1, 0, 0 };
   
    arrvec<int,3> nz1 = {nz11,nz12,nz13};
    arrvec<int,4> nz2 = {nz21,nz22,nz24,nz25};
    arrvec<int,3> nz3 = {nz33,nz34,nz35};
    
    fill_random<3>(tensor1,nz1);
    fill_random<4>(tensor2,nz2);
	fill_random<3>(tensor3,nz3);
	
	int unitnr = 0;
	
	
	if (rank == 0) unitnr = 6;
	
	std::cout << "NAME: " << std::endl;
	std::cout << tensor2.name() << std::endl;
	
	//dbcsr::contract<3,4,3>().alpha(1.0).t1(tensor1).t2(tensor2).t3(tensor3).beta(1.0)
	//	.con1({0,1}).ncon1({2}).con2({0,1}).ncon2({2,3}).map1({0}).map2({1,2}).print(true).log(true).perform();
	
	//dbcsr::contract(tensor1,tensor2,tensor3).print(true).perform("ijk, ijlm -> klm");
	
	//dbcsr::copy(tensor1,tensor3).perform();
	
	arrvec<int,2> sizes0 = {blk1,blk2};
	
	dbcsr::pgrid<2> grid2(MPI_COMM_WORLD);
	dbcsr::tensor<2> t2 = dbcsr::tensor<2>::create().name("TEST").ngrid(grid2).map1({0}).map2({1}).blk_sizes(sizes0);
	
	arrvec<int,2> nz00 = {nz11,nz12};
	
	fill_random<2>(t2,nz00);
	
	dbcsr::print(t2);
	
	tensor1.destroy();
	tensor2.destroy();	
	tensor3.destroy();
	
	pgrid3d.destroy();
	pgrid4d.destroy();

	*/
	
#ifdef USE_SCALAPACK
	scalapack::global_grid.free();
	c_blacs_exit(0);
#endif

	dbcsr::finalize();

	MPI_Finalize();

	return 0;

}
