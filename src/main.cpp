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
	
	std::shared_ptr<filio::data_handler> dh_out, dh_in;
	
	if (std::filesystem::exists(hdf5file)) {
		
		if (mega_world.rank() == 0 && std::filesystem::exists(hdf5backup))
			std::filesystem::remove(hdf5backup);
		
		if (mega_world.rank() == 0) std::filesystem::copy(hdf5file, hdf5backup);
		
		dh_in = std::make_shared<filio::data_handler>(hdf5backup, 
			filio::create_mode::append, comm);
	}
	
	dh_out = std::make_shared<filio::data_handler>(hdf5file, 
		filio::create_mode::truncate, comm);
		
	filio::data_io fh = {dh_in,dh_out};
	
	time.start();
	
	char* emergency_buffer = new char[1000];
	
	try {
		
		megalochem::driver mega_driver(mega_world, fh);
	
		mega_driver.parse_file(filename + ".json");
	
		mega_driver.run();
		
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
	
	dh_in.reset();
	dh_out.reset();

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
