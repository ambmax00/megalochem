#include <mpi.h>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <dbcsr_matrix.hpp>
#include <dbcsr_conversions.hpp>
#include <dbcsr_matrix_ops.hpp>
#include "io/reader.h"
#include "io/io.h"
#include "hf/hfmod.h"
#include "mp/mpmod.h"
#include "adc/adcmod.h"
#include "utils/mpi_time.h"

#include "extern/scalapack.h"

#include <filesystem>
#include <chrono>
#include <thread>

int main(int argc, char** argv) {

	MPI_Init(&argc, &argv);
	
	util::mpi_time time(MPI_COMM_WORLD, "Megalochem");
	
	util::mpi_log LOG(MPI_COMM_WORLD,0);
	
	//std::cout << std::scientific;

#ifndef DO_TESTING
	if (argc != 3) {
		LOG.os<>("Usage: ./chem [filename] [working_directory]\n");
		exit(0);
	}
#else
	LOG.os<>("MEGALOCHEM TEST RUN.\n");
#endif
	
	std::string s = R"(|  \/  ||  ___|  __ \ / _ \ | |   |  _  /  __ \| | | ||  ___|  \/  |)""\n"
					R"(| .  . || |__ | |  \// /_\ \| |   | | | | /  \/| |_| || |__ | .  . |)""\n"
					R"(| |\/| ||  __|| | __ |  _  || |   | | | | |    |  _  ||  __|| |\/| |)""\n"
					R"(| |  | || |___| |_\ \| | | || |___\ \_/ / \__/\| | | || |___| |  | |)""\n"
					R"(\_|  |_/\____/ \____/\_| |_/\_____/\___/ \____/\_| |_/\____/\_|  |_/)";
	
	LOG.banner(s,90,'*');
	LOG.os<>('\n');
	
#ifndef NDEBUG
	LOG.os<>("Build type: DEBUG\n\n");
#else 
	LOG.os<>("Build type: RELEASE\n\n");
#endif

	LOG.os<>("Authors: \n \t M. A. Ambroise\n\n");
	
	std::string filename(argv[1]);
	std::string workdir(argv[2]);
	
	if (!std::filesystem::exists(workdir))
		throw std::runtime_error("Working directory does not exist.");
	
	//set working directory
	std::filesystem::current_path(workdir);
	
	//create data and batching directory
	std::filesystem::create_directory(filename + "_data");
	std::filesystem::create_directory("batching");
	
	dbcsr::init();
	
	dbcsr::world wrd(MPI_COMM_WORLD);
	
	LOG.os<>("Running ", wrd.size(), " MPI processes with ", omp_get_max_threads(), " threads each.\n\n");
	
	time.start();
	
	try { // BEGIN MAIN CONTEXT
	
	int sysctxt = -1;
	c_blacs_get(0, 0, &sysctxt);
	int gridctxt = sysctxt;
	c_blacs_gridinit(&gridctxt, 'R', wrd.nprow(), wrd.npcol());
	scalapack::global_grid.set(gridctxt);
	
	filio::reader filereader(MPI_COMM_WORLD, filename);
	
	auto mol = filereader.get_mol();
	auto opt = filereader.get_opt();
	
	if (wrd.rank() == 0) {
		opt.print();
	}
	
	/*
	auto atoms = mol->atoms();
	auto xbas = std::make_shared<desc::cluster_basis>("cc-pvdz-ri", atoms, "atomic", 1);
	mol->set_cluster_dfbasis(xbas);
	
	ints::aofactory aofac(mol,wrd);
	auto s = aofac.ao_overlap();
	auto t = aofac.ao_kinetic();
	auto v = aofac.ao_nuclear();
	
	dbcsr::print(*s);
	dbcsr::print(*t);
	dbcsr::print(*v);
	
	auto grid3 = dbcsr::create_pgrid<3>(wrd.comm()).get();
	
	arrvec<int,3> xbb = {mol->dims().x(), mol->dims().b(), mol->dims().b()};
	
	auto ten = dbcsr::tensor_create<3>()
		.name("3c")
		.pgrid(grid3)
		.blk_sizes(xbb)
		.map1({0}).map2({1,2})
		.get();
		
	vec<vec<int>> bds = {
		vec<int>{0,2},
		vec<int>{0,2},
		vec<int>{0,2}
	};

	aofac.ao_3c2e_setup("coulomb");
	aofac.ao_3c2e_fill(ten, bds, nullptr);
	
	//dbcsr::print(*ten);
	
	auto ov = aofac.ao_2c2e("coulomb");
	
	auto scr = aofac.ao_schwarz();
	auto xscr = aofac.ao_3cschwarz();
	
	dbcsr::print(*ov);
	dbcsr::print(*scr);
	dbcsr::print(*xscr);
	
	exit(0);
*/	

	auto hfopt = opt.subtext("hf");
	
	hf::shared_hf_wfn myhfwfn = std::make_shared<hf::hf_wfn>();
	hf::hfmod myhf(wrd,mol,hfopt);

	bool skip_hf = hfopt.get<bool>("skip", false);
	bool do_hf = opt.get<bool>("do_hf", true);
	
	if (do_hf && !skip_hf) {
	
		myhf.compute();
		myhfwfn = myhf.wfn();
		myhfwfn->write_to_file(filename);
		myhfwfn->write_results(filename + "_data/out.json");
		
	} else {
		
		LOG.os<>("Reading HF info from files...\n");
		myhfwfn->read_from_file(filename,mol,wrd);
		myhfwfn->read_results(filename + "_data/out.json");
		LOG.os<>("Done.\n");
		
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	auto mpopt = opt.subtext("mp");
	
	bool do_mp = opt.get<bool>("do_mp");
	
	if (do_mp) {
	
		mp::mpmod mymp(wrd,myhfwfn,mpopt);
		mymp.compute();
		
	}

	auto adcopt = opt.subtext("adc");
	bool do_adc = opt.get<bool>("do_adc");
	
	if (do_adc) {
		
		adc::adcmod myadc(wrd,myhfwfn,adcopt);
		myadc.compute();
	}
	
	scalapack::global_grid.free();

   } catch (std::exception& e) {

		if (wrd.rank() == 0) {
			std::cout << "ERROR: ";
			std::cout << e.what() << std::endl;
			std::string skull = R"(
			
			
                  _( )                 ( )_
                 (_, |      __ __      | ,_)
                    \'\    /  ^  \    /'/
                     '\'\,/\      \,/'/'
                       '\| []   [] |/'
                         (_  /^\  _)
                           \  ~  /
                           /HHHHH\
                         /'/{^^^}\'\
                     _,/'/'  ^^^  '\'\,_
                    (_, |           | ,_)
                      (_)           (_)

            =====================================
            =      MEGALOCHEM WAS ABORTED       =
            =====================================
		)";
			std::cout << skull << std::endl;
			MPI_Abort(wrd.comm(), MPI_ERR_OTHER);
		} else {
		
			std::this_thread::sleep_for(std::chrono::milliseconds(10000));
			MPI_Abort(wrd.comm(), MPI_ERR_OTHER);
			
		}
	   
   }
    
    LOG.os<>("Wrapping up...\n");
    
    time.finish();
	
	time.print_info();
	
	dbcsr::print_statistics(true);

	//dbcsr::finalize();

	LOG.os<>("========== FINISHED WITHOUT CRASHING ! =========\n");

#ifdef DO_TESTING
	LOG.os<>("Now comparing reference to output\n");
	std::string output_ref(argv[3]);
	
	if (wrd.rank() == 0) {
		bool is_same = filio::compare_outputs(
			filename + "_data/out.json", output_ref);
		if (!is_same) {
			LOG.os<>("Not the same\n");
			MPI_Abort(wrd.comm(), MPI_ERR_OTHER);
		}
	}
#endif 

	MPI_Finalize();

	return 0;

}
