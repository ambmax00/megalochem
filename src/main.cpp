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

#include "math/solvers/davidson.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Core>

#include "io/data_handler.h"

class testmvp {
public:
	Eigen::MatrixXd m_mat;

	testmvp(Eigen::MatrixXd& A) : m_mat(A) {}
	
	dbcsr::shared_matrix<double> compute(dbcsr::shared_matrix<double> u_ia,
		double omega = 0.0) {
			
		auto u_eigen = dbcsr::matrix_to_eigen(u_ia);
		int r = u_eigen.rows();
		int c = u_eigen.cols();
				
		Eigen::VectorXd u_vec(r*c);
		std::copy(u_eigen.data(), u_eigen.data() + r*c, u_vec.data());
		
		Eigen::VectorXd sig_vec = m_mat * u_vec;
		
		Eigen::MatrixXd sig(r,c);
		std::copy(sig_vec.data(), sig_vec.data() + r*c, sig.data());
		
		auto world = u_ia->get_world();
		auto rblk = u_ia->row_blk_sizes();
		auto cblk = u_ia->col_blk_sizes();
		
		auto out = dbcsr::eigen_to_matrix(sig, world, "test", rblk, cblk, 
			dbcsr::type::no_symmetry);
						
		return out;
		
	}
	
};

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
	
	if (std::filesystem::exists(filename + ".h5")) {
		std::filesystem::rename(filename + ".h5", filename + "_back.h5");
	}
	
	dbcsr::init();
	
	dbcsr::world wrd(MPI_COMM_WORLD);
	
	filio::data_handler dhandler_write(wrd, filename + ".h5", filio::create_mode::create);
	filio::data_handler dhandler_read(wrd, filename + "_back.h5", filio::create_mode::no_create);
		
	LOG.os<>("Running ", wrd.size(), " MPI processes with ", omp_get_max_threads(), " threads each.\n\n");
	
	time.start();
	
	char* emergency_buffer = new char[1000];

	try { // BEGIN MAIN CONTEXT
	
	int sysctxt = -1;
	c_blacs_get(0, 0, &sysctxt);
	int gridctxt = sysctxt;
	c_blacs_gridinit(&gridctxt, 'R', wrd.nprow(), wrd.npcol());
	scalapack::global_grid.set(gridctxt);
	
	filio::reader filereader(MPI_COMM_WORLD, filename);
	
	auto mol = filereader.get_mol();
	auto opt = filereader.get_opt();
	
	dhandler_write.open(filio::access_mode::rdwr);
	dhandler_write.write_molecule(mol);
	dhandler_write.close();
	
	if (wrd.rank() == 0) {
		opt.print();
	}
	
	/*
	int no = 6;
	int nv = 12;
	int nroots = 8;
	double sparsity = 0.01;
	
	std::vector<int> rblk = {no};
	std::vector<int> cblk = {nv};
	
	std::vector<dbcsr::shared_matrix<double>> guesses(nroots);
	for (int i = 0; i != nroots; ++i) {
		Eigen::MatrixXd guess_eigen = Eigen::MatrixXd::Zero(no,nv);
		guess_eigen.data()[i] = 1.0;
		guesses[i] = dbcsr::eigen_to_matrix(guess_eigen, wrd, "test", rblk, cblk, 
			dbcsr::type::no_symmetry);
	}
	
	auto diagm = dbcsr::copy(guesses[0]).get();
	diagm->set(2.0);	

	Eigen::MatrixXd M = Eigen::MatrixXd::Random(no*nv, no*nv);
	
	Eigen::MatrixXd Mt = M.transpose();
	M = 0.5 * M + 0.5 * Mt;
	
	M = sparsity * M;
	for (int i = 0; i != M.rows(); ++i) {
		M(i,i) = i+1.2;
	}
	
	std::cout << M << std::endl;
	
	std::shared_ptr<testmvp> test = std::make_shared<testmvp>(M);
	
	math::davidson<testmvp> dav(MPI_COMM_WORLD, 3);
	dav.set_factory(test);
	dav.set_diag(diagm);
	dav.compute(guesses, nroots);
	
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
	es.compute(M);
			
	auto evals = es.eigenvalues();
	auto evecs = es.eigenvectors();
	
	for (int i = 0; i != nroots; ++i) {
		std::cout << evals(i) << " ";
	} std::cout << std::endl;
	
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
		
		dhandler_write.open(filio::access_mode::rdwr);
		dhandler_write.write_hf_wfn(myhfwfn);
		dhandler_write.close();
		
	} else {
		
		LOG.os<>("Reading HF info from files...\n");
		
		dhandler_read.open(filio::access_mode::read_only);
		myhfwfn = dhandler_read.read_hf_wfn(mol);
		dhandler_read.close();
		
		dhandler_write.open(filio::access_mode::rdwr);
		dhandler_write.write_hf_wfn(myhfwfn);
		dhandler_write.close();
		
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
	
	   	delete [] emergency_buffer;
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
