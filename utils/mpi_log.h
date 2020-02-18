#ifndef UTIL_MPI_LOG_H
#define UTIL_MPI_LOG_H

#include <iostream> 
#include <mpi.h>
#include <omp.h>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <stdexcept>
	
#define ALL -1

namespace util {

class mpi_log {

private:

	int m_proc = 0;
	int global_plev_;
	MPI_Comm m_comm;
	
public:
 
    mpi_log(MPI_Comm comm, int glb) 
		: global_plev_(glb), m_comm(comm) {};
    
    ~mpi_log() {}
    
    int global_plev() { return global_plev_; }
    
    void flush() {std::cout << std::flush; }
    
	void print_() { std::cout << std::flush; }
    
    template<typename T, typename... Args>
	void print_(T in, Args ... args) {
		std::cout << in;
		print_(args...);
	}
    
    template <int nprint = 0, typename T, typename... Args>
    void os(T in, Args ... args) {
		
		if (nprint <= global_plev_) {
			int rank_;
			MPI_Comm_rank(m_comm,&rank_);
			
			if (m_proc == -1) {
				
				if (omp_in_parallel()) 
					throw std::runtime_error("ERROR: You should not use mpi_log for all processes (-1) in parallel regions.");
				
				int size;
				MPI_Comm_size(m_comm,&size);
				for (int i = 0; i != size; ++i) {
					if (i == rank_) print_(in, args...);
					MPI_Barrier(m_comm);
				}
					
			} else {
				
				if (m_proc == rank_) print_(in, args...);
					
			} //end ifelse	
			
		}//endif nprint
		
	} //end os
	
	mpi_log operator()(int n) {
		mpi_log sub(m_comm, global_plev_);
		sub.m_proc = n;
		return sub;
	}
    
    template <typename T>
    void dbg(T& in) {
		std::cout << std::string("[DEBUG]: ");
		this->operator()(999).os<-1>(in);
	}
	
	template <typename T>
    void warning(T& in) { 
		std::cout << "[WARNING!]: ";
		this->operator()(-1).os<-1>(in);
	}
	
	template <typename T>
    void error(T& in) {
		std::cout << "[ERROR!]: ";
		this->operator()(-1).os<-1>(in);
		throw std::runtime_error("LOG ERROR.");
		
	}
	
	template <int nproc = 0, typename T>
	void banner(T m, int totl, char del) {
	
	  int rank_;
	  MPI_Comm_rank(m_comm,&rank_);
	  
	  if (rank_ == nproc) {
	
	  std::ostringstream stream;
	  stream << m;
	  
	  std::string str =  stream.str();
	
	  std::stringstream ss(str);
	  std::string to;
	  std::vector<std::string> message;

	  if (&m != NULL)
	  {
		while(std::getline(ss,to,'\n')){
		  message.push_back(to);
		}
	  }
	  
	  for (int i = 0; i != message.size(); ++i) {
		  if ((int)message[i].size() > totl-2) {
			  totl = message[i].size() + 2;
		  }
	  }
	  
	  std::string top(totl, del);
	  std::cout << "\n" << top << std::endl;
	  
	  for (int i = 0; i != message.size(); ++i) {
		  
		int nstr = message[i].size();
		int padl, padr;
			
		if ((totl-2-nstr)%2 != 0) {
			padl = (totl-2-nstr)/2;
			padr = padl + 1;
		} else {
			padl = (totl-2-nstr)/2;
			padr = padl;
		}
			
		std::string padleft(padl, ' ');
		std::string padright(padr, ' ');
		
		std::cout << del << padleft << message[i] << padright << del << std::endl;
		
	  }
			
		std::cout << top << "\n";
	
	}	
	
	std::cout << std::endl;
	
	}
	
	
	
}; // end class

} // end namespace


#endif  /* LOG_H */


	
