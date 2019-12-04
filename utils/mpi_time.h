#ifndef UTIL_MPI_TIME_H
#define UTIL_MPI_TIME_H

#include "utils/mpi_log.h"
#include <map>
#include <exception>

namespace util {

class mpi_time {

private:

	std::string proc_name;
	double time_ini, time_fin, tot;
	bool has_started;
	bool has_finished;
	mpi_log LOG;
	
	static int lev;
	
	std::map<std::string, mpi_time> subprocs;
	
	mpi_time operator=(mpi_time t) {
		return t;
	}

public: 

	mpi_time() : proc_name(""), has_started(false), has_finished(true), LOG(0) {}
	mpi_time(std::string n) : proc_name(n), has_started(false), has_finished(true), LOG(0) {}
	
	~mpi_time() {}
	
	void start() {
		
		if (has_started) {
			throw std::runtime_error("Timer has already been started!");
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		time_ini = MPI_Wtime();
		has_started = true;
		has_finished = false;
	}
		
	void end() {	
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		if (!has_started) {
			throw std::runtime_error("Timer has not been started yet!");
		}
		
		time_fin = MPI_Wtime();
		has_finished = true;
		has_started = false;
		
		tot += time_fin - time_ini;
		
	}
	
	mpi_time& sub(std::string subname) {
		
		if (subprocs.find(subname) == subprocs.end()) {
		 subprocs.insert({subname, mpi_time(subname)});			
		}
		
		return subprocs[subname];
		
	}
	
	void print_info() {
		
		++lev;
		
		std::string pad(4*lev + 2, '-');
		LOG.os<0>(0, pad, " ");
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		LOG.os<0>(0, proc_name, " took ", tot, " s\n");
		
		for (auto it : subprocs) {
			it.second.print_info();
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		--lev;
		
	}
		
		
};

} // end namespace util

#endif		
		
