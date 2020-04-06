#include <mpi.h>
#include <random>
#include <stdexcept>
#include <string>
#include "input/reader.h"
#include "hf/hfmod.h"
#include "adc/adcmod.h"
#include "utils/mpi_time.h"

/*
template <int N>
void fill_random(dbcsr::tensor<N,double>& t_in, vec<vec<int>>& nz) {
	
	
	int myrank, mpi_size;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank); 
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	if (myrank == 0) std::cout << "Filling Tensor..." << std::endl;
	if (myrank == 0) std::cout << "Dimension: " << N << std::endl;
	
	int nblocks = nz[0].size();
	std::vector<std::vector<int>> mynz(N);
	dbcsr::index<N> idx;
	
	for (int i = 0; i != nblocks; ++i) {
		
		// make index out of nzblocks
		for (int j = 0; j != N; ++j) idx[j] = nz[j][i];
		
		int proc = -1;
		
		t_in.get_stored_coordinates({.idx = idx, .proc = proc});
		
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
    
    dbcsr::iterator<N,double> iter(t_in);
    
    while (iter.blocks_left()) {
		  
		iter.next();
		
		dbcsr::block<N,double> blk(iter.sizes());
		
		blk.fill_rand();
		
		auto idx = iter.idx();
		t_in.put_block({.idx = idx, .blk = blk});
			
	}
	
}
*/

int main(int argc, char** argv) {

	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	util::mpi_time time(MPI_COMM_WORLD, "Megalochem");
	
	util::mpi_log LOG(MPI_COMM_WORLD,0);
	
	std::string s = R"(|  \/  ||  ___|  __ \ / _ \ | |   |  _  /  __ \| | | ||  ___|  \/  |)""\n"
					R"(| .  . || |__ | |  \// /_\ \| |   | | | | /  \/| |_| || |__ | .  . |)""\n"
					R"(| |\/| ||  __|| | __ |  _  || |   | | | | |    |  _  ||  __|| |\/| |)""\n"
					R"(| |  | || |___| |_\ \| | | || |___\ \_/ / \__/\| | | || |___| |  | |)""\n"
					R"(\_|  |_/\____/ \____/\_| |_/\_____/\___/ \____/\_| |_/\____/\_|  |_/)";
	
	
	LOG.banner(s,90,'*');
	
	if (argc < 2) {
		throw std::runtime_error("Wrong number of command line inputs.");
	}
	
	std::string filename(argv[1]);
	
	time.start();
	
	reader filereader(MPI_COMM_WORLD, filename);
	
	dbcsr::init();
	
	auto mol = std::make_shared<desc::molecule>(filereader.get_mol());
	auto opt = filereader.get_opt(); 
	
	auto cbas = mol->c_basis();
	
	for (int i = 0; i != cbas.size(); ++i) {
		for (auto s : cbas[i]) {
			std::cout << s << std::endl;
		}
	}
	
	auto hfopt = opt.subtext("hf");
	
	desc::shf_wfn myhfwfn = std::make_shared<desc::hf_wfn>();
	hf::hfmod myhf(mol,hfopt,MPI_COMM_WORLD);
	
	bool skip_hf = hfopt.get<bool>("skip", false);
	
	if (!skip_hf) {
	
		myhf.compute();
		myhfwfn = myhf.wfn();
		myhfwfn->write_to_file();
		
	} else {
		
		LOG.os<>("Reading HF info from files...\n");
		myhfwfn->read_from_file(mol,MPI_COMM_WORLD);
		
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	auto adcopt = opt.subtext("adc");
	
	std::cout << adcopt.get<int>("nroots") << std::endl;
	
	adc::adcmod myadc(myhfwfn,adcopt,MPI_COMM_WORLD);
	
	myadc.compute();
	
	time.finish();
	
	time.print_info();
	//ints::aofactory ao(mol, MPI_COMM_WORLD);
	
	//auto s = ao.compute<2>({.op = "overlap", .bas = "bb", .name = "S", .map1 = {0}, .map2 = {1}});
	
	//dbcsr::print(s);
	
	/*
	dbcsr::pgrid<3> pgrid3d({.comm = MPI_COMM_WORLD});
	dbcsr::pgrid<4> pgrid4d({.comm = MPI_COMM_WORLD});
							 
	std::cout << "Created!" << std::endl;
	
	
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
	
	vec<vec<int>> sizes1 = {blk1,blk2,blk3}; 
	vec<vec<int>> sizes2 = {blk1,blk2,blk4,blk5};
	vec<vec<int>> sizes3 = {blk3,blk4,blk5};
	
	dbcsr::tensor<3,double> tensor1({.name = "(13|2)", .pgridN = pgrid3d, .map1 = map11,
		.map2 = map12, .blk_sizes = sizes1});
		
	//dbcsr::tensor<3,double> tensortest({.name="test", .pgridN = pgrid3d, .map1 = map11,
	//	.map2 = map12, .blk_sizes = sizes1});
						   
    dbcsr::tensor<4,double> tensor2({.name = "(54|21)", .pgridN = pgrid4d, .map1 = map21,
		.map2 = map22, .blk_sizes = sizes2});
    
    dbcsr::tensor<3,double> tensor3({.name = "(3|45)", .pgridN = pgrid3d, .map1 = map31,
		.map2 = map32, .blk_sizes = sizes3});
	
	std::cout << "Tensors created." << std::endl;
	
	pgrid3d.destroy();
	
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
   
    vec<vec<int>> nz1 = {nz11,nz12,nz13};
    vec<vec<int>> nz2 = {nz21,nz22,nz24,nz25};
    vec<vec<int>> nz3 = {nz33,nz34,nz35};
    
    fill_random(tensor1,nz1);
    fill_random(tensor2,nz2);
	fill_random(tensor3,nz3);
	
	int unitnr = 0;
	
	if (rank == 0) unitnr = 6;
	
	std::cout << "NAME: " << std::endl;
	std::cout << tensor2.name() << std::endl;
	
	
	/*
	//dbcsr::contract<3,4,3>({.alpha = 1.0d, .t1 = tensor1, .t2 = tensor2, .beta = 3.0d, 
	//		.t3 = tensor3, .con1 = {0,1}, .ncon1 = {2}, .con2 = {0,1}, .ncon2 = {2,3},
	//		.map1 = {0}, .map2 = {1,2}, .unit_nr = unitnr, .log = true });
	
	//vec<int> c1, c2, nc1, nc2, m1, m2;
	dbcsr::einsum<3,4,3>({.x = "ijk, ijlm -> klm", .t1 = tensor1, .t2 = tensor2, 
			.t3 = tensor3, .unit_nr = unitnr, .log = true});
	
			
	tensor1.destroy();
	tensor2.destroy();	
	tensor3.destroy();
	//tensortest.destroy();
	
	
	pgrid4d.destroy();
	
	
	 /*
	
	dbcsr::pgrid<2> pgrid2d({.comm = MPI_COMM_WORLD});
							 
	std::cout << "Created!" << std::endl;
	
	
	vec<int> blk1 = {2, 2};
    vec<int> blk2 = {2, 3};
	
	vec<vec<int>> sizes1 = {blk1,blk2}; 
	
	auto d1 = dbcsr::random_dist(2, pgrid2d.dims()[0]);
	auto d2 = dbcsr::random_dist(2, pgrid2d.dims()[1]);

	dbcsr::dist<2> dist2d({.pgridN = pgrid2d, .map1 = {0}, .map2 = {1}, .nd_dists = {d1,d2}});
	
	dbcsr::tensor<2,double> tensor1({.name = "(1|2)", .distN = dist2d, .map1 = {0},
		.map2 = {1}, .blk_sizes = sizes1});
	
	std::cout << "Tensors created." << std::endl;
	
	vec<int> nz1 = {0, 0, 1};
	vec<int> nz2 = {0, 1, 1};
   
    vec<vec<int>> nz = {nz1,nz2};
    
    
    fill_random(tensor1,nz);
	
	
	
	auto blks = tensor1.blks_local();
	
	auto blks2 = tensor1.blks_local();
	
	for (auto blk : blks) {
		for (auto x : blk) {
			std::cout << x << std::endl;
		}
	}
	
	auto mat = tensor_to_eigen(tensor1,1);
	
	auto ten2 = eigen_to_tensor(mat, "other", pgrid2d, vec<int>{0}, vec<int>{1}, vec<vec<int>>{blk1,blk2});
	
	dbcsr::print(tensor1);
	dbcsr::print(ten2);
	
	tensor1.destroy();
	ten2.destroy();
	dist2d.destroy();
	pgrid2d.destroy();

	*/

	dbcsr::finalize();
	

	MPI_Finalize();

	return 0;

}
