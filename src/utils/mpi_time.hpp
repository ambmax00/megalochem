#ifndef UTIL_MPI_TIME_H
#define UTIL_MPI_TIME_H

#include <exception>
#include <map>
#include "utils/mpi_log.hpp"

namespace util {

class mpi_time;

class mpi_time {
 private:
  std::string proc_name;
  MPI_Comm m_comm;
  double time_ini, time_fin, tot;
  bool has_started;
  bool has_finished;
  mpi_log LOG;

  int lev;
  int nproc;  //<- how many times it was started

  std::map<std::string, mpi_time> subprocs;

 public:
  mpi_time() :
      proc_name(""), has_started(false), has_finished(true), LOG(m_comm, 0),
      lev(0), nproc(0)
  {
  }

  mpi_time(MPI_Comm comm, std::string n, int plev = 0) :
      proc_name(n), m_comm(comm), tot(0), has_started(false),
      has_finished(true), LOG(comm, plev), lev(0), nproc(0)
  {
  }

  ~mpi_time()
  {
  }

  mpi_time(const mpi_time& in) = default;

  mpi_time& operator=(const mpi_time& in) = default;

  void start()
  {
    if (has_started) {
      throw std::runtime_error(
          "Timer " + proc_name + " has already been started!");
    }

    // MPI_Barrier(m_comm);

    time_ini = MPI_Wtime();
    has_started = true;
    has_finished = false;

    ++nproc;
  }

  void finish()
  {
    // MPI_Barrier(m_comm);

    if (!has_started) {
      throw std::runtime_error(
          "Timer " + proc_name + " has not been started yet!");
    }

    time_fin = MPI_Wtime();
    has_finished = true;
    has_started = false;

    tot += time_fin - time_ini;
  }

  mpi_time& sub(std::string subname)
  {
    if (subprocs.find(subname) == subprocs.end()) {
      subprocs.insert({subname, mpi_time(m_comm, subname, LOG.global_plev())});
    }

    auto& s = subprocs[subname];
    s.lev = lev + 1;

    return s;
  }

  /* inserts another time object into this time
   * Overwrites if inserted time obj. has the same name as another
   * subprocess of the time object it is inserted into.
   */
  void insert(mpi_time& time_in)
  {
    auto name = time_in.proc_name;

    this->subprocs[name] = time_in;
  }

  void print_info(int n = 0)
  {
    //++lev;

    std::string pad(4 * lev + 2, '-');
    LOG(n).os<0>(pad, " ");

    // MPI_Barrier(m_comm);

    LOG(n).os<0>(proc_name, " \t took \t ", tot, " s");

    if (nproc > 1)
      LOG(n).os<0>(" \t (Average: \t", tot / nproc, " s)");

    LOG(n).os<0>('\n');

    for (auto it : subprocs) { it.second.print_info(n); }

    // MPI_Barrier(m_comm);

    //--lev;
  }
};

}  // end namespace util

#endif
