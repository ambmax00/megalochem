#ifndef UTIL_MPI_TIME_H
#define UTIL_MPI_TIME_H

#include "utils/mpi_log.h"
#include <map>
#include <exception>

namespace util {

class mpi_time {

private:

	std::string proc_name;
	MPI_Comm m_comm;
	double time_ini, time_fin, tot;
	bool has_started;
	bool has_finished;
	mpi_log LOG;
	
	int lev;
	int nproc; //<- how many times it was started
	
	std::map<std::string, mpi_time> subprocs;
	
	mpi_time operator=(mpi_time t) {
		return t;
	}

public: 

	mpi_time() : proc_name(""), has_started(false), has_finished(true), LOG(m_comm,0), lev(0), nproc(0) {}
	mpi_time(MPI_Comm comm, std::string n, int plev = 0) : proc_name(n), has_started(false), 
		has_finished(true), LOG(comm,plev), lev(0), nproc(0), tot(0), m_comm(comm) {}
	
	~mpi_time() {}
	
	void start() {
		
		if (has_started) {
			throw std::runtime_error("Timer has already been started!");
		}
		
		MPI_Barrier(m_comm);
		
		time_ini = MPI_Wtime();
		has_started = true;
		has_finished = false;
		
		++nproc;
		
	}
		
	void finish() {	
		
		MPI_Barrier(m_comm);
		
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
		 subprocs.insert({subname, mpi_time(m_comm, subname, LOG.global_plev())});			
		}
		
		auto& s = subprocs[subname];
		s.lev = lev + 1;
		
		return s;
		
	}
	
	void print_info() {
		
		//++lev;
		
		std::string pad(4*lev + 2, '-');
		LOG.os<0>(pad, " ");
		
		MPI_Barrier(m_comm);
		
		LOG.os<0>(proc_name, " \t took \t ", tot, " s");
		
		if (nproc > 1)
			LOG.os<0>(" \t (Average: \t", tot/nproc, " s)");
			
		LOG.os<0>('\n');
		
		for (auto it : subprocs) {
			it.second.print_info();
		}
		
		MPI_Barrier(m_comm);
		
		//--lev;
		
	}
		
		
};

} // end namespace util

#endif		
		
