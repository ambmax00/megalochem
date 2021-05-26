#ifndef UTIL_SCHEDULER_H
#define UTIL_SCHEDULER_H

#include <mpi.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

//#define _DLOG

namespace util {

class mutex {
 private:
  MPI_Comm _comm;
  MPI_Win _lock_win;
  int _comm_size, _comm_rank;

  std::atomic<bool>* _lock_data;
  std::atomic<bool>* _lock_ptr;

  const bool _IS_LOCKED = true;
  const bool _IS_UNLOCKED = false;

 public:
  mutex(MPI_Comm comm) : _comm(comm)
  {
    assert(ATOMIC_BOOL_LOCK_FREE == 2);

    MPI_Comm_size(_comm, &_comm_size);
    MPI_Comm_rank(_comm, &_comm_rank);

    int atomic_size = sizeof(std::atomic<bool>);
    int nele = (_comm_rank == 0) ? 1 : 0;

    MPI_Win_allocate_shared(
        nele * atomic_size, atomic_size, MPI_INFO_NULL, _comm, &_lock_data,
        &_lock_win);

    if (_comm_rank == 0) {
      new (_lock_data) std::atomic<bool>(_IS_UNLOCKED);
    }

    int size;
    MPI_Aint disp;

    MPI_Win_shared_query(_lock_win, 0, &disp, &size, &_lock_ptr);

    MPI_Barrier(_comm);
  }

  bool try_lock()
  {
    bool expected = _IS_UNLOCKED;
    bool desired = _IS_LOCKED;

    bool success = _lock_ptr->compare_exchange_strong(
        expected, desired, std::memory_order_relaxed,
        std::memory_order_relaxed);

    return success;
  }

  void lock()
  {
    bool expected = _IS_UNLOCKED;
    bool desired = _IS_LOCKED;

    while (_lock_ptr->compare_exchange_strong(
        expected, desired, std::memory_order_relaxed,
        std::memory_order_relaxed)) {
      expected = false;
    }
  }

  void unlock()
  {
    bool expected = _IS_LOCKED;
    bool desired = _IS_UNLOCKED;

    _lock_ptr->compare_exchange_strong(
        expected, desired, std::memory_order_relaxed,
        std::memory_order_relaxed);
  }

  ~mutex()
  {
    MPI_Win_free(&_lock_win);
  }

  mutex(const mutex& in) = delete;
  mutex& operator=(const mutex& in) = delete;
};

class basic_scheduler {
 private:
  const int64_t NO_TASK = -1;
  const int64_t TERMINATE = -2;

  MPI_Comm _global_comm;
  int _global_rank;
  int _global_size;

  MPI_Comm _local_comm;
  int _local_rank;
  int _local_size;

  int64_t _global_counter;

  // local mutex
  std::shared_ptr<util::mutex> _local_mtx;

  // local atomics;
  MPI_Win _local_atomic_win;
  std::atomic<int64_t>* _atomic_data;
  std::atomic<int64_t>* _local_counter;
  std::atomic<int64_t>* _local_ubound;

  std::function<void(int64_t)>& _executer;
  int64_t _ntasks;

  std::thread* _poll_thread;

  int64_t _ntasks_completed;

#define _DPRINT(str) \
  std::cout << "RANK: " << _global_rank << "/" << _local_rank << " : " << str \
            << std::endl;

  bool local_queue_empty()
  {
    return (*_local_counter >= *_local_ubound);
  }

  bool terminate()
  {
    return (*_local_counter == TERMINATE);
  }

  void communicate_0()
  {
    MPI_Status status;
    int flag = 0;

    MPI_Iprobe(MPI_ANY_SOURCE, 0, _global_comm, &flag, &status);

    if (flag) {
      _DPRINT("Got request from " + std::to_string(status.MPI_SOURCE));

      int64_t ntasks_requested;
      MPI_Recv(
          &ntasks_requested, 1, MPI_INT64_T, status.MPI_SOURCE, 0, _global_comm,
          MPI_STATUS_IGNORE);

      int64_t return_counter =
          (_global_counter < _ntasks) ? _global_counter : TERMINATE;

      int64_t last_task = (return_counter + ntasks_requested <= _ntasks) ?
          return_counter + ntasks_requested :
          _ntasks;

      _global_counter += ntasks_requested;

      int64_t buffer[2] = {return_counter, last_task};

      MPI_Send(buffer, 2, MPI_INT64_T, status.MPI_SOURCE, 1, _global_comm);
    }
  }

  void fetch_tasks()
  {
    _DPRINT("Fetching tasks");
    if (_local_mtx->try_lock()) {
      if (!local_queue_empty()) {
        _local_mtx->unlock();
        return;
      }

      _DPRINT("Filling queue")

      int64_t nrequests = _local_size;

      _DPRINT("Requesting " + std::to_string(nrequests))

      MPI_Send(&_local_size, 1, MPI_INT64_T, 0, 0, _global_comm);

      int64_t buffer[2];

      MPI_Recv(buffer, 2, MPI_INT64_T, 0, 1, _global_comm, MPI_STATUS_IGNORE);

      int64_t counter = buffer[0];
      int64_t last_task = buffer[1];

      _DPRINT(
          "NCOUNTER/UBOUND: " + std::to_string(counter) + "/" +
          std::to_string(last_task));

      std::atomic_store(_local_counter, counter);
      std::atomic_store(_local_ubound, last_task);

      _local_mtx->unlock();
    }
    else {
      while (local_queue_empty() && !terminate()) {}
    }
  }

  int64_t pop_task()
  {
    int64_t itask = std::atomic_fetch_add(_local_counter, (int64_t)1);
    return (itask >= *_local_ubound) ? NO_TASK : itask;
  }

 public:
  basic_scheduler(
      MPI_Comm comm, int64_t ntasks, std::function<void(int64_t)>& executer) :
      _global_comm(comm),
      _global_rank(-1), _global_size(0), _local_rank(-1), _local_size(0),
      _global_counter(0), _executer(executer), _ntasks(ntasks)
  {
    // create communicators
    MPI_Comm_dup(comm, &_global_comm);

    MPI_Comm_size(_global_comm, &_global_size);
    MPI_Comm_rank(_global_comm, &_global_rank);

    if (_global_size < 2) {
      throw std::runtime_error("Scheduler needs 2 or more processes.");
    }

    // int color = _global_rank % 2;
    // int split = _global_rank / 2;

    // MPI_Comm_split(_global_comm, color, split, &_local_comm);

    MPI_Comm_split_type(
        _global_comm, MPI_COMM_TYPE_SHARED, _global_rank, MPI_INFO_NULL,
        &_local_comm);

    MPI_Comm_size(_local_comm, &_local_size);
    MPI_Comm_rank(_local_comm, &_local_rank);

    _local_mtx.reset(new mutex(_local_comm));

    // allocate shared windows
    // ... for atomics

    int atom_size = sizeof(std::atomic<int64_t>);
    int nele_atom = (_local_rank == 0) ? 2 : 0;

    MPI_Win_allocate_shared(
        nele_atom * atom_size, atom_size, MPI_INFO_NULL, _local_comm,
        &_atomic_data, &_local_atomic_win);

    // initialize
    if (_local_rank == 0) {
      new (_atomic_data) std::atomic<int64_t>(0);
      new (_atomic_data + 1) std::atomic<int64_t>(0);
    }

    int size0;
    MPI_Aint disp0;

    MPI_Win_shared_query(_local_atomic_win, 0, &disp0, &size0, &_local_counter);
    _local_ubound = _local_counter + 1;

    _DPRINT(
        "INIT: " + std::to_string(*_local_counter) + "/" +
        std::to_string(*_local_ubound))
  }

  void run()
  {
    MPI_Barrier(_global_comm);

    _DPRINT("STARTING RUN WITH NTASKS = " + std::to_string(_ntasks))

    _ntasks_completed = 0;

    std::vector<int> ntvec(_ntasks, 0);

    if (_global_rank != 0) {
      while (true) {
        if (local_queue_empty()) {
          fetch_tasks();
        }

        if (terminate())
          break;

        // get task
        int64_t task = pop_task();

        // perform if task valid
        if (task != NO_TASK) {
          _DPRINT("EXECUTING TASK " + std::to_string(task))
          _executer(task);
          ++_ntasks_completed;
          ntvec[task] += 1;
        }
      }
    }
    else {
      while (true) {
        communicate_0();
        if (terminate())
          break;
      }
    }

    _DPRINT("DONE")

    MPI_Request barrier_request;
    int flag = 0;

    MPI_Ibarrier(_global_comm, &barrier_request);

    while (!flag) {
      MPI_Test(&barrier_request, &flag, MPI_STATUS_IGNORE);
      if (_global_rank == 0) {
        communicate_0();
      }
    }

    _DPRINT("EXITED LOOP");

    MPI_Barrier(_global_comm);

    int64_t ntasks_completed_global = 0;

    // check for consistency

    MPI_Allreduce(
        &_ntasks_completed, &ntasks_completed_global, 1, MPI_INT64_T, MPI_SUM,
        _global_comm);

    MPI_Barrier(_global_comm);

    _DPRINT("Done here")

    if (_global_rank == 0 && ntasks_completed_global != _ntasks) {
      // not throwing here because it somehow leads to a deadlock, dunno why
      std::cout << "Scheduler: Not all tasks were executed!!" << std::endl;
      MPI_Abort(_global_comm, MPI_ERR_OTHER);
    }

    MPI_Barrier(_global_comm);

    _DPRINT("DONE WITH RUN")
  }

  ~basic_scheduler()
  {
    MPI_Comm_free(&_global_comm);
    MPI_Win_free(&_local_atomic_win);
    MPI_Comm_free(&_local_comm);
  }
};

#undef _DPRINT

}  // namespace util

#endif
