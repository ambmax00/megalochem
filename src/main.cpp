#include <mpi.h>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <dbcsr_matrix.hpp>
#include <dbcsr_conversions.hpp>
#include <dbcsr_matrix_ops.hpp>
#include "io/parser.hpp"
#include "io/data_handler.hpp"
#include "hf/hfmod.hpp"
#include "mp/mpmod.hpp"
#include "adc/adcmod.hpp"
#include "utils/mpi_time.hpp"
#include "utils/unique.hpp"
#include "extern/scalapack.hpp"
#include "git.hpp"

#include <filesystem>
#include <chrono>
#include <thread>

#include "math/solvers/davidson.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Core>

#include "megalochem.hpp"
#include "megalochem_driver.hpp"

/*
namespace megalochem {
	
dbcsr::cart init(MPI_Comm comm) {
	
	// init dbcsr
	dbcsr::init();
	
	dbcsr::cart wrd(comm);
	
	// init scalapack
	int sysctxt = -1;
	c_blacs_get(0, 0, &sysctxt);
	
	int gridctxt = sysctxt;
	c_blacs_gridinit(&gridctxt, 'R', wrd.nprow(), wrd.npcol());
	
	scalapack::global_grid.set(gridctxt);
	
	return wrd;

}

class world {
private:

	

public:

	world(MPI_Comm comm)
	
	
};


	
}

int main(int argc, char** argv) {
	
	MPI_Init(&argc,&argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	
	// init SCALPACK, DBCSR, etc...
	auto megaworld = megalochem::init(comm);
	
	megalochem::print_devinfo(comm);
	
	/*
	// init file output? checks if hdf5 present, returns fistream, fostream, logstream
	mg::fhandle fh = megalochem::init_io(comm, hdf5name, logname);
	
	// stack/job controller
	// saves file data, checks for errors.
	
	
	stack = megalochem::parse_file(megaworld, file);
	
	stack.run(fh);
	
	megalochem::finalize();
	
}*/

using namespace megalochem;

void print_devinfo(MPI_Comm comm) {

	util::mpi_log LOG(comm, 0);
	
	MPI_Barrier(comm);
	
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
	LOG.os<>("Commit: ", GitMetadata::CommitSHA1(), '\n');
	
	std::string dirty = (GitMetadata::AnyUncommittedChanges()) ? "true" : "false";
	
	LOG.os<>("Uncommitted changes: ", dirty, '\n');
	
	MPI_Barrier(comm);

}
	
int main(int argc, char** argv) {

	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	
	util::mpi_time time(comm, "Megalochem");
	util::mpi_log LOG(comm ,0);
	
	if (argc != 3) {
		LOG.os<>("Usage: ./chem [filename] [working_directory]\n");
		exit(0);
	}
	
	std::string filename(argv[1]);
	std::string workdir(argv[2]);
	
	megalochem::init(comm, workdir);
	
	megalochem::world mega_world(comm);
	
	
	
	print_devinfo(comm);
	
	LOG.os<>("Running ", mega_world.size(), " MPI processes with ", 
		omp_get_max_threads(), " threads each.\n\n");
	
	// set up files
	std::string hdf5file = filename + ".hdf5";
	std::string hdf5backup = filename + ".back.hdf5";
	
	std::shared_ptr<filio::data_handler> dh, dh_back;
	
	if (std::filesystem::exists(hdf5file)) {
		if (mega_world.rank() == 0 && std::filesystem::exists(hdf5backup))
			std::filesystem::remove(hdf5backup);
		
		if (mega_world.rank() == 0) std::filesystem::copy(hdf5file, hdf5backup);
		
		dh_back = std::make_shared<filio::data_handler>(hdf5backup, 
			filio::create_mode::append, comm);
	}
	
	dh = std::make_shared<filio::data_handler>(hdf5file, 
		filio::create_mode::truncate, comm);
		
	filio::data_io fh = {dh_back,dh};
		
	megalochem::driver d(mega_world, fh);
	
	d.parse_file(filename + ".json");
	
	d.run();
	
	exit(0);
	
	time.start();
	
	char* emergency_buffer = new char[1000];

	try { // BEGIN MAIN CONTEXT
	
	std::string input_file = filename + ".json";
	if (!std::filesystem::exists(input_file)) {
		throw std::runtime_error("Could not find input file!");
	}
	
	nlohmann::json data;
	std::ifstream ifstr(input_file);
	ifstr >> data;
	
	auto mol = filio::parse_molecule(data,comm,0);
	auto opt = filio::parse_options(data,comm,0);
	
	auto cbas = mol->c_basis();
	auto atm = mol->atoms();
	
	//LOG.os<>("REMOVING LINDEP\n");
	//auto newcbas = ints::remove_lindep(wrd, cbas, atm);
	
	/*auto newmol = desc::molecule::create()
		.comm(wrd.comm())
		.name(mol->name())
		.atoms(atm)
		.cluster_basis(newcbas)
		.charge(mol->charge())
		.mult(mol->mult())
		.mo_split(mol->mo_split())
		.build();
		
	auto cbas2 = mol->c_basis2();
	if (cbas2) newmol->set_cluster_basis2(cbas2);*/
		
	mol->print_info(1);
	LOG.os<>('\n');
		
	//newmol->print_info(1);
	LOG.os<>('\n'); 
	
	//mol = newmol;
	
	//exit(0);
	
	desc::write_molecule("molecule", *mol, *dh);
	
	if (mega_world.rank() == 0) {
		opt.print();
	}
	
	auto hfopt = opt.subtext("hf");
	
	hf::shared_hf_wfn myhfwfn;
	hf::hfmod myhf(mega_world,mol,hfopt);

	bool skip_hf = hfopt.get<bool>("skip", false);
	bool do_hf = opt.get<bool>("do_hf", true);
	
	if (do_hf && !skip_hf) {
	
		myhf.compute();
		myhfwfn = myhf.wfn();
		hf::write_hfwfn("hf_wfn", *myhfwfn, *dh);
		
	} else {
		
		LOG.os<>("Reading HF info from files...\n");
		myhfwfn = hf::read_hfwfn("hf_wfn", mol, mega_world, *dh_back);
		hf::write_hfwfn("hf_wfn", *myhfwfn, *dh);
		LOG.os<>("Done.\n");
		
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	auto mpopt = opt.subtext("mp");
	
	bool do_mp = opt.get<bool>("do_mp");
	
	if (do_mp) {
	
		//mp::mpmod mymp(mega_world,myhfwfn,mpopt);
		//mymp.compute();
		
	}

	auto adcopt = opt.subtext("adc");
	bool do_adc = opt.get<bool>("do_adc");
	
	if (do_adc) {
		
		//adc::adcmod myadc(mega_world,myhfwfn,adcopt);
		//myadc.compute();
	}
	
	mega_world.free();

   } catch (std::exception& e) {
	
	   	delete [] emergency_buffer;
		if (mega_world.rank() == 0) {
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
			MPI_Abort(mega_world.comm(), MPI_ERR_OTHER);
		} else {
		
			std::this_thread::sleep_for(std::chrono::milliseconds(10000));
			MPI_Abort(mega_world.comm(), MPI_ERR_OTHER);
			
		}
	   
   }
    
    LOG.os<>("Wrapping up...\n");
    
    time.finish();
	
	time.print_info();
	
	dbcsr::print_statistics(true);
	
	// close hdf5 files
	
	dh.reset();
	dh_back.reset();

	//dbcsr::finalize();

	LOG.os<>("========== FINISHED WITHOUT CRASHING ! =========\n");

#ifdef DO_TESTING
	LOG.os<>("Now comparing reference to output\n");
	std::string output_ref(argv[3]);
	
	if (mega_world.rank() == 0) {
		bool is_same = filio::compare_outputs(
			filename + "_data/out.json", output_ref);
		if (!is_same) {
			LOG.os<>("Not the same\n");
			MPI_Abort(mega_world.comm(), MPI_ERR_OTHER);
		}
	}
#endif 

	MPI_Finalize();

	return 0;

}
