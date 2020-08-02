#include <mpi.h>
#include <random>
#include <stdexcept>
#include <string>
#include <dbcsr_matrix.hpp>
#include <dbcsr_conversions.hpp>
#include <dbcsr_matrix_ops.hpp>
#include "io/reader.h"
#include "hf/hfmod.h"
#include "mp/mpmod.h"
//#include "adc/adcmod.h"
#include "utils/mpi_time.h"

#include "extern/scalapack.h"

#include "math/linalg/LLT.h"
#include <filesystem>

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
	
	util::mpi_time time(MPI_COMM_WORLD, "Megalochem");
	
	util::mpi_log LOG(MPI_COMM_WORLD,0);
	
	if (argc != 3) {
		LOG.os<>("Usage: ./chem [filename] [working_directory]\n");
		exit(0);
	}
	
	std::string s = R"(|  \/  ||  ___|  __ \ / _ \ | |   |  _  /  __ \| | | ||  ___|  \/  |)""\n"
					R"(| .  . || |__ | |  \// /_\ \| |   | | | | /  \/| |_| || |__ | .  . |)""\n"
					R"(| |\/| ||  __|| | __ |  _  || |   | | | | |    |  _  ||  __|| |\/| |)""\n"
					R"(| |  | || |___| |_\ \| | | || |___\ \_/ / \__/\| | | || |___| |  | |)""\n"
					R"(\_|  |_/\____/ \____/\_| |_/\_____/\___/ \____/\_| |_/\____/\_|  |_/)";
	
	
	LOG.banner(s,90,'*');
	
	std::string filename(argv[1]);
	std::string workdir(argv[2]);
	
	if (!std::filesystem::exists(workdir))
		throw std::runtime_error("Working directory does not exist.");
	
	//set working directory
	std::filesystem::current_path(workdir);
	
	//create data and batching directory
	std::filesystem::create_directory("data");
	std::filesystem::create_directory("batching");
	
	dbcsr::init();
	
	dbcsr::world wrd(MPI_COMM_WORLD);
	
	time.start();
	
	
	{ // BEGIN MAIN CONTEXT
	
//#ifdef USE_SCALAPACK
	int sysctxt = -1;
	c_blacs_get(0, 0, &sysctxt);
	int gridctxt = sysctxt;
	c_blacs_gridinit(&gridctxt, 'R', wrd.nprow(), wrd.npcol());
	scalapack::global_grid.set(gridctxt);
//#endif
	
	filio::reader filereader(MPI_COMM_WORLD, filename);
	
	auto mol = std::make_shared<desc::molecule>(filereader.get_mol());
	auto opt = filereader.get_opt();
	
	auto hfopt = opt.subtext("hf");
	
	desc::shf_wfn myhfwfn = std::make_shared<desc::hf_wfn>();
	hf::hfmod myhf(mol,hfopt,wrd);

	bool skip_hf = hfopt.get<bool>("skip", false);
	bool do_hf = opt.get<bool>("do_hf", true);
	
	if (do_hf && !skip_hf) {
	
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
	auto mpopt = opt.subtext("mp");
	
	bool do_mp = opt.get<bool>("do_mp");
	
	if (do_mp) {
	
		mp::mpmod mymp(myhfwfn,mpopt,wrd);
		mymp.compute_batch();
		
	}*/
	
//#ifdef USE_SCALAPACK
	scalapack::global_grid.free();
	//c_blacs_exit(0);
//#endif

   } // END MAIN CONTEX*/
    
    LOG.os<>("THE END\n");
    
    time.finish();
	
	time.print_info();
	
	dbcsr::print_statistics(true);

	dbcsr::finalize();

	MPI_Finalize();

	return 0;

}
