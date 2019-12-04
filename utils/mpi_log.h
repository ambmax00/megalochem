#ifndef UTIL_MPI_LOG_H
#define UTIL_MPI_LOG_H

#include <iostream> 
#include <mpi.h>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <stdexcept>
	
#define ALL -1

namespace util {

class mpi_log {

private:

	int global_plev_;
	
public:
 
    mpi_log(int glb) 
		: global_plev_(glb) {};
    
    ~mpi_log() {}
    
	void print_() { std::cout << std::flush; }
    
    template<typename T, typename... Args>
	void print_(T in, Args ... args) {
		std::cout << in;
		print_(args...);
	}
    
    template <int nprint = 0, int nproc = 0, typename T, typename... Args>
    void os(T in, Args ... args) {
		
		if (nprint <= global_plev_) {
			int rank_;
			MPI_Comm_rank(MPI_COMM_WORLD,&rank_); 
			
			if ((nproc == rank_) || (nproc == -1)) print_(in, args...);
			
		}
		
	}
    
    template <int nproc = 0, typename T>
    void dbg(T& in) {
		std::cout << std::string("[DEBUG]: ");
		os<999, nproc>(in);
	}
	
	template <int nproc = 0, typename T>
    void warning(T& in) { 
		std::cout << "[WARNING!]: ";
		os<-1,nproc>(in);
	}
	
	template <int nproc = 0, typename T>
    void error(T& in) {
		std::cout << "[ERROR!]: ";
		os<-1, nproc>(in);
		throw std::runtime_error("LOG ERROR.");
		
	}
	
	template <int nproc = 0, typename T>
	void banner(T m, int totl, char del) {
	
	  int rank_;
	  MPI_Comm_rank(MPI_COMM_WORLD,&rank_);
	  
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


	
