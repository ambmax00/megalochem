#ifndef UTIL_SCHEDULER_H
#define UTIL_SCHEDULER_H

#include <deque>
#include <vector>
#include <mpi.h>
#include <functional>
#include <algorithm>
#include <random>
#include <unistd.h>
#include <mutex>
#include <atomic>
#include <iostream>
#include <thread>
#include <chrono>

//#define _DLOG

namespace util {

// BROKEN ON SOME CLUSTERS. ONE-SIDED MPI KINDA BLOWS :(
class scheduler {
private:

	// for transfer
	const int64_t TRANSFER_NO_RESP = -1;
	const int64_t TRANSFER_NO_TASK = -2;
	
	// for request
	const int REQUEST_NONE = -1;
	
	// for status
	const int STATUS_NO_WORK = 0;
	const int STATUS_HAS_WORK = 1;

	// for token
	const int TOKEN_NONE = -1;
	const int TOKEN_TERMINATE = -2;
	
	// mpi assertions
	int MPI_MODE = 0; // MPI_MODE_NOCHECK;

	MPI_Comm m_comm;
	int m_mpisize;
	int m_mpirank;
	int64_t m_ntasks;
	
	bool m_terminate_flag;
	
	int64_t* m_transfer;
	int* m_request;
	int* m_status;
	int* m_token;
	
	MPI_Win m_transfer_win = MPI_WIN_NULL;
	MPI_Win m_request_win = MPI_WIN_NULL;
	MPI_Win m_status_win = MPI_WIN_NULL;
	MPI_Win m_token_win = MPI_WIN_NULL;
	
	std::deque<int64_t> m_local_tasks; 
	std::function<void(int64_t)>& m_executer;
	int64_t m_exectasks;
	
	void init_deque() {
		for (int64_t i = 0; i != m_ntasks; ++i) {
			int proc = i % m_mpisize;
			if (proc == m_mpirank) m_local_tasks.push_back(i);
		}
	}
	
	void update_status() {
		MPI_Win_flush(m_mpirank, m_status_win);
		m_status[m_mpirank] = (m_local_tasks.empty()) ? 
			STATUS_NO_WORK : STATUS_HAS_WORK;
		MPI_Win_sync(m_status_win);
	}
	
	void add_task(int64_t t) {
		m_local_tasks.push_front(t);
		update_status();
	}
	
	bool terminate() {
		return (m_mpisize == 1) ? m_local_tasks.empty() : m_terminate_flag;
	}
	
	void acquire() {
		
		std::random_device random_device;
		std::mt19937 random_engine(random_device());
		std::uniform_int_distribution<int> distr(0, m_mpisize-1);
		
		auto get_rand = [&distr,&random_engine,this]() 
		{
			int r = m_mpirank;
			if (m_mpisize == 1) return 0;
			
			while (r == m_mpirank) { 
				r = distr(random_engine);
			}
			
			return r;
		};

		//std::cout << m_mpirank << " : " << "ACQUIRE" << std::endl;
		while (!terminate()) {
			
			MPI_Win_flush(m_mpirank, m_transfer_win);
			m_transfer[m_mpirank] = TRANSFER_NO_RESP;
			MPI_Win_sync(m_transfer_win);
			
			int victim = get_rand();
			int v_work = STATUS_NO_WORK;
			
			// get status of victim
			MPI_Get(&v_work, 1, MPI_INT, victim, victim, 1, MPI_INT, m_status_win);
			MPI_Win_flush(victim, m_status_win);
							
			// continue if victim has no work
			if (v_work == STATUS_NO_WORK) {
				communicate();
				continue;
			}
			
			// make victim aware that task can be stolen
			int result = REQUEST_NONE;
			
			MPI_Compare_and_swap(&m_mpirank, &REQUEST_NONE, &result, MPI_INT,
				victim, victim, m_request_win);
			MPI_Win_flush(victim, m_request_win);
						
			// continue if not succeeded 
			if (result != REQUEST_NONE) {
				communicate();
				continue;
			}
			
			#ifdef _DLOG
			std::cout << m_mpirank << " : " << "CAS SUCCEEDED, NOW WAITING" << std::endl;
			#endif
			
			// loop and check if request accepted
			int64_t newtask = TRANSFER_NO_RESP;
			
			while (newtask == TRANSFER_NO_RESP && !terminate()) {
				newtask = m_transfer[m_mpirank];
				MPI_Win_flush(m_mpirank, m_transfer_win);
				communicate();
			}
			
			#ifdef _DLOG
			std::cout << m_mpirank << " : " << "GOT RESPONSE FROM VICTIM: " 
				<< newtask << std::endl;
			#endif
			
			if (newtask != TRANSFER_NO_TASK) {
				
				add_task(newtask);
				
				return;
			}
			
			communicate();
			
		}
	}
	
	void transfer_token() {
		
		MPI_Win_flush(m_mpirank, m_token_win);
		int mytoken = m_token[m_mpirank];
		
		if (mytoken < -2 || mytoken > m_mpisize) {
			throw std::runtime_error("Scheduler: something went wrong.");
		}
		
		// no token -> nothing to do
		if (mytoken == TOKEN_NONE) return;
		
		// update my token
		if (mytoken != TOKEN_TERMINATE) {
			
			if (m_mpirank == 0 && mytoken == 0) {
				mytoken = TOKEN_TERMINATE;
			} else if (m_mpirank == 0 && mytoken > 0) {
				mytoken = (m_local_tasks.empty()) ? 0 : 1;
			} else {
				mytoken += (m_local_tasks.empty()) ? 0 : 1;
			}
						
		}
		
		// give neighbour token
		if (!terminate()) {
			
			int neigh = (m_mpirank + 1) % m_mpisize;
			
			#ifdef _DLOG
			std::cout << m_mpirank << " : " << "GIVING TOKEN " << mytoken 
				<< " to " << neigh << std::endl;
			#endif
			
			MPI_Put(&mytoken, 1, MPI_INT, neigh, neigh, 1, MPI_INT, m_token_win);
			MPI_Win_flush(neigh, m_token_win);
			
			m_token[m_mpirank] = TOKEN_NONE;
			MPI_Win_sync(m_token_win);
						
		}	
		
		if (mytoken == TOKEN_TERMINATE) {
			m_terminate_flag = true;
		}
		
	}
	
	void communicate() {
		
		// token
		transfer_token();
		
		// check if there is a processor request
		// we lock because we are swapping values, so we don't miss any requests
		int thief = -1;
		
		MPI_Win_flush(m_mpirank, m_request_win);
		thief = m_request[m_mpirank];
                   				
		if (thief == REQUEST_NONE) return;
		
		int64_t t = TRANSFER_NO_TASK;
		
		if (!m_local_tasks.empty()) {
			t = m_local_tasks.back();
			m_local_tasks.pop_back();
			update_status();
		}
		
		#ifdef _DLOG
		std::cout << m_mpirank << " : " << "PUTTING task " << t << " on THIEF " 
			<< thief << std::endl;
		#endif
		
		MPI_Put(&t, 1, MPI_LONG_LONG, thief, thief, 1, MPI_LONG_LONG,
			m_transfer_win);
		MPI_Win_flush(thief, m_transfer_win);
		
		m_request[m_mpirank] = REQUEST_NONE;
		MPI_Win_sync(m_request_win);
						
	}
	
	void execute(int64_t task) {
		#ifdef _DLOG
		std::cout << m_mpirank << " : " << "EXECUTE" << std::endl;
		#endif
		m_executer(task);
		++m_exectasks;
	}
			
	
public:

	scheduler(MPI_Comm comm, int64_t ntasks, std::function<void(int64_t)>& executer) 
		: m_comm(comm), m_executer(executer), m_ntasks(ntasks)
	{
				
		MPI_Comm_size(comm, &m_mpisize);
		MPI_Comm_rank(comm, &m_mpirank);
		init_deque();
		
	}
	
	void run() {
		
		m_exectasks = 0;
		
		MPI_Alloc_mem(m_mpisize * sizeof(int64_t), MPI_INFO_NULL, &m_transfer);
		MPI_Alloc_mem(m_mpisize * sizeof(int), MPI_INFO_NULL, &m_status);
		MPI_Alloc_mem(m_mpisize * sizeof(int), MPI_INFO_NULL, &m_request);
		MPI_Alloc_mem(m_mpisize * sizeof(int), MPI_INFO_NULL, &m_token);
		
		std::fill(m_transfer, m_transfer + m_mpisize, TRANSFER_NO_RESP);
		std::fill(m_request, m_request + m_mpisize, REQUEST_NONE);
		std::fill(m_status, m_status + m_mpisize, STATUS_HAS_WORK); 
		    // ^ maybe change this in case more procs than tasks
		std::fill(m_token, m_token + m_mpisize, TOKEN_NONE);
		
		m_terminate_flag = false;
		if (m_mpirank == 0) m_token[0] = 1;
		
		MPI_Win_create(m_transfer, m_mpisize * sizeof(int64_t), sizeof(int64_t), 
			MPI_INFO_NULL, m_comm, &m_transfer_win);
			
		MPI_Win_create(m_request, m_mpisize * sizeof(int), sizeof(int), 
			MPI_INFO_NULL, m_comm, &m_request_win);
			
		MPI_Win_create(m_status, m_mpisize * sizeof(int), sizeof(int), 
			MPI_INFO_NULL, m_comm, &m_status_win);
			
		MPI_Win_create(m_token, m_mpisize * sizeof(int), sizeof(int), 
			MPI_INFO_NULL, m_comm, &m_token_win);
		
		MPI_Win_lock_all(MPI_MODE_NOCHECK, m_transfer_win);
		MPI_Win_lock_all(MPI_MODE_NOCHECK, m_status_win);
		MPI_Win_lock_all(MPI_MODE_NOCHECK, m_request_win);
		MPI_Win_lock_all(MPI_MODE_NOCHECK, m_token_win);
		
		MPI_Win_sync(m_transfer_win);
		MPI_Win_sync(m_status_win);
		MPI_Win_sync(m_request_win);
		MPI_Win_sync(m_token_win);
		
		MPI_Barrier(m_comm);
				
		while (!terminate()) {
			
			if (m_local_tasks.empty()) {
				acquire();
			} else {
				int64_t task = m_local_tasks.front();
				m_local_tasks.pop_front();
				update_status();
				communicate();
				execute(task);
			}
			
		}
		
		#ifdef _DLOG
		std::cout << m_mpirank << " : EXIT" << std::endl;
		#endif
		
		MPI_Request barrier_request;
		int flag = 0;
		
		MPI_Ibarrier(m_comm,&barrier_request);
		
		while (!flag) {
			MPI_Test(&barrier_request, &flag, MPI_STATUS_IGNORE);
			communicate();
		}
		
		MPI_Win_unlock_all(m_token_win);
		MPI_Win_unlock_all(m_request_win);
		MPI_Win_unlock_all(m_status_win);
		MPI_Win_unlock_all(m_transfer_win);
		
		MPI_Win_free(&m_transfer_win);
		MPI_Win_free(&m_request_win);
		MPI_Win_free(&m_status_win);
		MPI_Win_free(&m_token_win);
		
		MPI_Free_mem(m_transfer);
		MPI_Free_mem(m_status);
		MPI_Free_mem(m_token);
		MPI_Free_mem(m_request);
		
		MPI_Allreduce(MPI_IN_PLACE, &m_exectasks, 1, MPI_LONG_LONG, 
			MPI_SUM, m_comm);
			
		if (m_exectasks != m_ntasks) {
			throw std::runtime_error("Scheduler: not all tasks were executed.");
		} 
		#ifdef _DLOG
		else {
			std::cout << "All tasks executed successfully." << std::endl;
		}
		#endif
				
	}
	
	~scheduler() {}
	
};

/* new dynamic scheduler
 * uses shared memory for processes on same node
 * each node has a master, which distributes tasks to slaves
 * masters communicate between nodes for work sharing
 * 
 * Hopefully more transferrable than the other scheduler...
 * After testing: Nope, this one also has problems
 */

class dynamic_scheduler {
private:

	const int64_t NO_TASK = -1;
	const int64_t TERMINATE = -2;

	MPI_Comm _global_comm;
	int _global_rank;
	int _global_size;
	
	MPI_Comm _local_comm;
	int _local_rank;
	int _local_size;
	
	MPI_Comm _king_comm;
	int _king_rank;
	int _king_size;
	bool _is_king;
	
	MPI_Win _task_window;
	int64_t* _task_data;

	MPI_Win _king_window;
	int64_t* _king_data;
	
	std::function<void(int64_t)>& _executer;
	int64_t _ntasks;
	
	int64_t _ntasks_completed;

	std::deque<int64_t> _task_queue;
	
	void knight_run() {
		
		int64_t* task0 = &_task_data[-_local_size];
		int64_t* taskp = &_task_data[-_local_size + _local_rank];
		
		while (true) {
			
			//std::cout << "RANK: " << _global_rank << " TASK: " <<
			//	*taskp << std::endl;
			
			MPI_Win_sync(_task_window);
			if (*taskp >= 0) {
				std::cout << _global_rank << " : " << " EXECUTING: " << *taskp << std::endl;
				_executer(*taskp);
				++_ntasks_completed;
				*taskp = NO_TASK;
			}

			//sleep(1);
			
			if (*taskp == NO_TASK && *task0 == TERMINATE) break;
			
		}
	}
	
	// gets tasks from global queue
	void king_get_tasks() {
		
		//std::cout << _global_rank << ": " << "AQCUIRING TASKS" << std::endl;
		if (_task_queue.empty()) {
			
			int64_t nrequests = _local_size - 1;
			int64_t ncounter = -1;
			
			std::cout << _global_rank << ": " << "REQUESTING: " 
				<< nrequests << std::endl;
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, _king_window);
			MPI_Fetch_and_op(&nrequests, &ncounter, MPI_LONG_LONG, 0, 0,
				MPI_SUM, _king_window);
			MPI_Win_unlock(0, _king_window);
			
			std::cout << _global_rank << ": " << "NCOUNTER " << ncounter << std::endl;
			int64_t last_task = (ncounter + nrequests <= _ntasks) ? 
				ncounter + nrequests : _ntasks;
			
			// add to queue
			for (int64_t itask = ncounter; itask < last_task; ++itask) 
			{
				_task_queue.push_back(itask);
			}
			
			std::cout << "GOT TASKS: ";
			for (auto q : _task_queue) { std::cout << q << " " << std::endl; }
			
			if (_task_queue.empty()) {
				_task_data[0] = TERMINATE;
			}
			
		}
			
	}
	
	void king_run() {
		
		while (_task_data[0] != TERMINATE) {
				
			//std::cout << "RANK: " << _global_rank << " SIZE: " <<
			//	_task_queue.size() << std::endl;
			
			MPI_Win_sync(_task_window);
			for (int iknight = 1; iknight != _local_size; ++iknight) {
				
				if (_task_queue.empty()) continue;
				
				//std::cout << "RANK: " << _global_rank << " KNIGHT: " << iknight << " " << _task_data[iknight] << std::endl;
				if (_task_data[iknight] < 0) {
					_task_data[iknight] = _task_queue.front();
					_task_queue.pop_front();
				}
			}
			
			king_get_tasks();
			//sleep(1);
			
		}
		
	}

public:

	dynamic_scheduler(MPI_Comm comm, int64_t ntasks, std::function<void(int64_t)>& executer) 
		: _global_comm(comm), _executer(executer), _ntasks(ntasks),
		_global_size(0), _global_rank(-1),
		_local_size(0), _local_rank(-1),
		_king_size(0), _king_rank(-1)
	{
		
		// create communicators
		
		MPI_Comm_size(_global_comm, &_global_size);
		MPI_Comm_rank(_global_comm, &_global_rank);
				
		int color = _global_rank % 2;
		int split = _global_rank / 2;
		
		//MPI_Comm_split(_global_comm, color, split, &_local_comm);
		MPI_Comm_split_type(_global_comm, MPI_COMM_TYPE_SHARED, _global_rank, 
			MPI_INFO_NULL, &_local_comm);
		
		MPI_Comm_size(_local_comm, &_local_size);
		MPI_Comm_rank(_local_comm, &_local_rank);
		
		//std::cout << "RANK: " << _local_rank << std::endl;
		
		if (_local_size < 2) 
			throw std::runtime_error("More than 2 processes per node required.");
		
		int king_color = (_local_rank == 0) ? 0 : MPI_UNDEFINED;
		MPI_Comm_split(_global_comm, king_color, _global_rank, &_king_comm);
		_is_king = false;
		
		if (_king_comm != MPI_COMM_NULL) {
			//std::cout << "We are the kings!!" << std::endl;
			_is_king = true;
			MPI_Comm_size(_king_comm, &_king_size);
			MPI_Comm_rank(_king_comm, &_king_rank);
		}
		
	}
	
	void run() {
		
		_ntasks_completed = 0;
		
		// create windows
		if (_is_king) {
			MPI_Win_allocate(sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL,
				_king_comm, &_king_data, &_king_window);
			
			if (_king_rank == 0) {
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, _king_window);
				_king_data[0] = 0;
				MPI_Win_unlock(0, _king_window);
			}
			
			MPI_Barrier(_king_comm);
			king_get_tasks();
		}
		
		int nele = (_is_king) ? _local_size : 0;
		MPI_Win_allocate_shared(nele * sizeof(int64_t), sizeof(int64_t),
			MPI_INFO_NULL, _local_comm, &_task_data, &_task_window);
		
		MPI_Win_lock_all(MPI_MODE_NOCHECK, _task_window);
		
		if (_is_king) {
			std::fill(_task_data, _task_data + _local_size, NO_TASK);
		}
		
		MPI_Win_sync(_task_window);
		
		if (_is_king) {
			king_run();
		} else {
			knight_run();
		}
		
		std::cout << "RANK: " << _global_rank << " : DONE" << std::endl;

		// check for consistency
		MPI_Allreduce(MPI_IN_PLACE, &_ntasks_completed, 1, MPI_LONG_LONG,
			MPI_SUM, _global_comm);
			
		MPI_Win_unlock_all(_task_window);
		MPI_Win_free(&_task_window);
		if (_is_king) MPI_Win_free(&_king_window);
			
		if (_global_rank == 0 && _ntasks_completed != _ntasks) {
			throw std::runtime_error("Scheduler: Not all tasks were executed!!");
		}

	}
		
	
	~dynamic_scheduler() {
		MPI_Comm_free(&_local_comm);
		if (_is_king) MPI_Comm_free(&_king_comm);
	}
	
};

class shared_mutex {
private:

	MPI_Comm _comm;
	int _rank, _size;
	
	MPI_Win _mutex_window;
	
	int* _mutex_data;
	int* _mutex_var;
	
public:

	shared_mutex(MPI_Comm comm) : _comm(comm) {
		
		MPI_Comm_rank(_comm, &_rank);
		MPI_Comm_size(_comm, &_size);
		
		int nele = (_rank == 0) ? 1 : 0;
		MPI_Win_allocate_shared(nele * sizeof(int), sizeof(int),
			MPI_INFO_NULL, _comm, &_mutex_data, &_mutex_window);
		
		if (_rank == 0) {
			*_mutex_data = 0;
			_mutex_var = _mutex_data;
		} else {
			_mutex_var = _mutex_data - 1;
		}
		
		MPI_Win_lock_all(MPI_MODE_NOCHECK, _mutex_window);
		
	}
	
	bool try_lock() {		
		
		MPI_Win_sync(_mutex_window);
		
		int expected = 0;
		int desired = 1;
				
		bool success = __atomic_compare_exchange (_mutex_var, &expected, &desired, false, 
			__ATOMIC_RELAXED, __ATOMIC_RELAXED);
		
		return success;
		
	}
	
	void lock() {		
		
		MPI_Win_sync(_mutex_window);
		
		int expected = 0;
		int desired = 1;
		
		//std::cout << "Rank: " << _rank << " is requesting a lock " << std::endl;
		
		while (!__atomic_compare_exchange (_mutex_var, &expected, &desired, false, 
			__ATOMIC_RELAXED, __ATOMIC_RELAXED)) 
		{
			expected = 0;
		};
		
		//std::cout << "Rank: " << _rank << " request granted " << std::endl;
		
	}
	
	void unlock() {
		
		MPI_Win_sync(_mutex_window);
		
		int expected = 1;
		int desired = 0;
	
		__atomic_compare_exchange (_mutex_var, &expected, &desired, false, 
			__ATOMIC_RELAXED, __ATOMIC_RELAXED);
	}
	
	~shared_mutex() {
		MPI_Win_unlock_all(_mutex_window);
		MPI_Win_free(&_mutex_window);
	}
		
};

class shared_queue {
private: 

	using iterator = int64_t*;

	MPI_Comm _comm;
	int _mpirank, _mpisize;
	
	MPI_Win _queue_window;
	int64_t* _queue_data;
	int64_t _queue_maxsize;
	
	int64_t* _queue_data_begin;
	int64_t* _queue_data_end;
	int64_t* _queue_pos_begin;
	int64_t* _queue_pos_end;
	
	
public:

	shared_queue(MPI_Comm comm, const int64_t size) 
		: _comm(comm), _queue_maxsize(size)
	{
		
		MPI_Comm_rank(_comm, &_mpirank);
		MPI_Comm_size(_comm, &_mpisize);
		
		int64_t extent = _queue_maxsize + 2;
		int64_t nele = (_mpirank == 0) ? extent : 0;
		
		MPI_Win_allocate_shared(nele * sizeof(int64_t), sizeof(int64_t),
			MPI_INFO_NULL, _comm, &_queue_data, &_queue_window);
			
		_queue_data_begin = &_queue_data[(_mpirank == 0) ? 0 : -extent];
		_queue_data_end = _queue_data_begin + _queue_maxsize;
		_queue_pos_begin = _queue_data_end + 0;
		_queue_pos_end = _queue_data_end + 1;
			
		if (_mpirank == 0) {
			std::fill(_queue_data, _queue_data + _queue_maxsize, 0);
			*_queue_pos_begin = 0;
			*_queue_pos_end = 0;
		}
		
		MPI_Win_lock_all(MPI_MODE_NOCHECK, _queue_window);
		
	}
	
	iterator begin() {
		MPI_Win_sync(_queue_window);
		return _queue_data_begin + *_queue_pos_begin;
	}
	
	iterator end() {
		MPI_Win_sync(_queue_window);
		return _queue_data_begin + *_queue_pos_end;
	}
	
	int64_t size() {
		return *_queue_pos_end - *_queue_pos_begin;
	}
	
	int64_t& operator[](size_t i) {
		return *(_queue_data_begin + *_queue_pos_begin + i);
	}
	
	void push(int64_t t) {
		MPI_Win_sync(_queue_window);
		*(this->end()) = t;
		(*_queue_pos_end)++;
	}
	
	int64_t pop() {
		MPI_Win_sync(_queue_window);
		(*_queue_pos_end)--;
		return *(this->end());
	}
	
	bool empty() {
		MPI_Win_sync(_queue_window);
		return (*_queue_pos_end - *_queue_pos_begin == 0);
	}
	
	~shared_queue() {
		MPI_Win_unlock_all(_queue_window);
		MPI_Win_free(&_queue_window);
	}
	
};

// This one has problems, too T_T

class dynamic_scheduler2 {
private:

	const int64_t NO_TASK = -1;
	const int64_t TERMINATE = -2;

	MPI_Comm _global_comm;
	int _global_rank;
	int _global_size;
	
	MPI_Comm _local_comm;
	int _local_rank;
	int _local_size;
	
	// window for global queue
	MPI_Win _globQ_window;
	int64_t* _globQ_data;
	int64_t* _global_counter;
	
	std::function<void(int64_t)>& _executer;
	int64_t _ntasks;
	
	int64_t _ntasks_completed;
	
	std::unique_ptr<shared_queue> _locQ;
	std::unique_ptr<shared_mutex> _mtx;

	void fetch_tasks() {
		
		if (_locQ->empty()) {
			if (_mtx->try_lock()) {
			
				int64_t nrequests = _local_size;
				int64_t ncounter = -1;
				
				//std::cout << _global_rank << ": " << "REQUESTING: " 
				//	<< nrequests << std::endl;
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, _globQ_window);
				MPI_Fetch_and_op(&nrequests, &ncounter, MPI_LONG_LONG, 0, 0,
					MPI_SUM, _globQ_window);
				MPI_Win_unlock(0, _globQ_window);
				
				//std::cout << _global_rank << ": " << "NCOUNTER " << ncounter << std::endl;
				int64_t last_task = (ncounter + nrequests <= _ntasks) ? 
					ncounter + nrequests : _ntasks;
				
				// add to queue
				for (int64_t itask = ncounter; itask < last_task; ++itask) 
				{
					_locQ->push(itask);
				}
				
				//std::cout << "GOT TASKS: ";
				for (auto q : *_locQ) { std::cout << q << " " << std::endl; }
				
				if (_locQ->empty()) {
					for (int i = 0; i != _local_size; ++i) {
						_locQ->push(TERMINATE);
					}
				}
				
				_mtx->unlock();
			}
		}
		
	}
			

public:

	dynamic_scheduler2(MPI_Comm comm, int64_t ntasks, std::function<void(int64_t)>& executer) 
		: _global_comm(comm), _executer(executer), _ntasks(ntasks),
		_global_size(0), _global_rank(-1),
		_local_size(0), _local_rank(-1)
	{
		
		// create communicators
		
		MPI_Comm_size(_global_comm, &_global_size);
		MPI_Comm_rank(_global_comm, &_global_rank);
		
		//MPI_Comm_split(_global_comm, color, split, &_local_comm);
		MPI_Comm_split_type(_global_comm, MPI_COMM_TYPE_SHARED, _global_rank, 
			MPI_INFO_NULL, &_local_comm);
		
		MPI_Comm_size(_local_comm, &_local_size);
		MPI_Comm_rank(_local_comm, &_local_rank);
		
	}
	
	void run() {
		
		//std::cout << "STARTING RUN : " << _ntasks << std::endl;
		
		_ntasks_completed = 0;
		int64_t qsize = std::max((int64_t)_local_size, _ntasks);
		
		MPI_Win_allocate(sizeof(int64_t), sizeof(int64_t),
			MPI_INFO_NULL, _global_comm, &_globQ_data, &_globQ_window);

		if (_global_rank == 0) {
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, MPI_MODE_NOCHECK, _globQ_window);
			int64_t zero = 0;
			MPI_Put(&zero, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, _globQ_window);
			MPI_Win_unlock(0, _globQ_window);
		}
		
		_locQ = std::make_unique<shared_queue>(_local_comm, qsize);
		_mtx = std::make_unique<shared_mutex>(_local_comm);
		
		while (true) {
						
			fetch_tasks();
			
			_mtx->lock();
			
			if (_locQ->empty()) {
				_mtx->unlock();
				continue;
			}
				
			int64_t task = _locQ->pop();
			
			_mtx->unlock();
			
			if (task == TERMINATE) break;
						
			//std::cout << _global_rank << " : EXECUTING " << task << std::endl;
			_executer(task);
			++_ntasks_completed;
			
			
		}
		
		//std::cout << _global_rank << " : DONE " << std::endl;
		
		// check for consistency
		MPI_Allreduce(MPI_IN_PLACE, &_ntasks_completed, 1, MPI_LONG_LONG,
			MPI_SUM, _global_comm);
			
		MPI_Win_free(&_globQ_window);
			
		if (_global_rank == 0 && _ntasks_completed != _ntasks) {
			throw std::runtime_error("Scheduler: Not all tasks were executed!!");
		}
		
		//std::cout << "DONE WITH RUN" << std::endl;

	}
		
	
	~dynamic_scheduler2() {
		MPI_Comm_free(&_local_comm);
	}
	
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
	MPI_Win _local_mtx_win;
	std::mutex* _mtx_data;
	std::mutex* _local_mtx;
	
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
	std::cout << "RANK: " << _global_rank << "/" << _local_rank \
		<< " : " << str << std::endl;
	
	bool local_queue_empty() {
		return (*_local_counter >= *_local_ubound);
	}
	
	bool terminate() {
		return (*_local_counter == TERMINATE);
	}
	
	void communicate_0() {
		
		MPI_Status status;
		int flag = 0;
		
		MPI_Iprobe(MPI_ANY_SOURCE, 0, _global_comm, &flag, &status);
			
		if (flag) {
			
			_DPRINT("Got request from " + std::to_string(status.MPI_SOURCE));
			
			int64_t ntasks_requested;
			MPI_Recv(&ntasks_requested, 1, MPI_LONG_LONG, status.MPI_SOURCE,
				0, _global_comm, MPI_STATUS_IGNORE);
			
			int64_t return_counter = (_global_counter < _ntasks) ?
				_global_counter : TERMINATE;
				
			int64_t last_task = (return_counter + ntasks_requested <= _ntasks) ? 
				return_counter + ntasks_requested : _ntasks;
				
			_global_counter += ntasks_requested;
			
			int64_t buffer[2] = {return_counter, last_task};
			
			MPI_Send(buffer, 2, MPI_LONG_LONG, status.MPI_SOURCE, 1, _global_comm);
			
		}
		
	}
			
	void fetch_tasks() {
				
		_DPRINT("Fetching tasks");
		if (_local_mtx->try_lock()) {
			
			_DPRINT("Filling queue")
			
			int64_t nrequests = _local_size;
			int64_t ncounter = -1;
				
			_DPRINT("Requesting " + std::to_string(nrequests))
			
			MPI_Send(&_local_size, 1, MPI_LONG_LONG, 0, 0, _global_comm);
			
			int64_t buffer[2]; 
			
			MPI_Recv(buffer, 2, MPI_LONG_LONG, 0, 1, _global_comm, MPI_STATUS_IGNORE);
			
			int64_t counter = buffer[0];
			int64_t last_task = buffer[1];
											
			_DPRINT("NCOUNTER/UBOUND: " + std::to_string(counter) +  "/" 
				+ std::to_string(last_task));
				
			std::atomic_store(_local_counter, counter);
			std::atomic_store(_local_ubound, last_task); 
			
			_local_mtx->unlock();
			
		} else {
			
			while (local_queue_empty() && !terminate()) {}
			
		}
							
	}
	
	int64_t pop_task() {
		
		int64_t itask = std::atomic_fetch_add(_local_counter,(int64_t)1);
		return (itask >= *_local_ubound) ? NO_TASK : itask;
		
	}
			

public:

	basic_scheduler(MPI_Comm comm, int64_t ntasks, std::function<void(int64_t)>& executer) 
		: _global_comm(comm), _executer(executer), _ntasks(ntasks),
		_global_size(0), _global_rank(-1),
		_local_size(0), _local_rank(-1),
		_global_counter(0)
	{
		
		// create communicators
		MPI_Comm_dup(comm, &_global_comm);

		MPI_Comm_size(_global_comm, &_global_size);
		MPI_Comm_rank(_global_comm, &_global_rank);
		
		if (_global_size < 2) {
			throw std::runtime_error("Scheduler needs 2 or more processes.");
		}
		
		int color = _global_rank % 2;
		int split = _global_rank / 2;
		
		//MPI_Comm_split(_global_comm, color, split, &_local_comm);
		
		MPI_Comm_split_type(_global_comm, MPI_COMM_TYPE_SHARED, _global_rank, 
			MPI_INFO_NULL, &_local_comm);
		
		MPI_Comm_size(_local_comm, &_local_size);
		MPI_Comm_rank(_local_comm, &_local_rank);
		
		// allocate shared windows
		// ... for mutex
		
		int mtx_size = sizeof(std::mutex);
		int nele = (_local_rank == 0) ? 1 : 0;
		
		MPI_Aint size0;
		int disp0;
		
		MPI_Win_allocate_shared(nele * mtx_size, mtx_size, MPI_INFO_NULL, 
			_local_comm, &_mtx_data, &_local_mtx_win);
		
		// initialize
		if (_local_rank == 0) {
			auto init_mtx = new (_mtx_data) std::mutex();
		}
			
		MPI_Win_shared_query(_local_mtx_win, 0, &size0, &disp0,&_local_mtx);
		
		// ... for atomics
		
		int atom_size = sizeof(std::atomic<int64_t>);
		int nele_atom = (_local_rank == 0) ? 2 : 0;
		
		MPI_Win_allocate_shared(nele_atom * atom_size, atom_size,
			MPI_INFO_NULL, _local_comm, &_atomic_data, &_local_atomic_win);
			
		// initialize
		if (_local_rank == 0) {
			auto init = new (_atomic_data) std::atomic<int64_t>(0);
			auto init2 = new (_atomic_data + 1) std::atomic<int64_t>(0);
		}
		
		MPI_Win_shared_query(_local_atomic_win, 0, &size0, &disp0, &_local_counter);
		_local_ubound = _local_counter + 1;
		
		_DPRINT("INIT: " + std::to_string(*_local_counter) + "/" 
			+ std::to_string(*_local_ubound))
		
	}
	
	void run() {
		
		_DPRINT("STARTING RUN WITH NTASKS = " + std::to_string(_ntasks))
		
		_ntasks_completed = 0;
		
		if (_global_rank != 0) {
		
			while (true) {
				
				if (local_queue_empty()) {
					
					fetch_tasks();
					
				}
				
				if (terminate()) break;
				
				// get task
				int64_t task = pop_task();
				
				// perform if task valid
				if (task != NO_TASK) {
					_DPRINT("EXECUTING TASK " + std::to_string(task))
					_executer(task);
					++_ntasks_completed;
				}
				
			}
			
		} else {
			
			while (true) {
				
				communicate_0();
				if (terminate()) break;
				
			}
			
		}
				
		_DPRINT("DONE")
		
		MPI_Request barrier_request;
		int flag = 0;
		
		MPI_Ibarrier(_global_comm,&barrier_request);
		
		while (!flag) {
			MPI_Test(&barrier_request, &flag, MPI_STATUS_IGNORE);
			if (_global_rank == 0) {
				communicate_0();
			}
		}	
		
		// check for consistency
		MPI_Allreduce(MPI_IN_PLACE, &_ntasks_completed, 1, MPI_LONG_LONG,
			MPI_SUM, _global_comm);
						
		if (_global_rank == 0 && _ntasks_completed != _ntasks) {
			throw std::runtime_error("Scheduler: Not all tasks were executed!!");
		}
		
		_DPRINT("DONE WITH RUN")

	}
		
	
	~basic_scheduler() {
		MPI_Comm_free(&_global_comm);
		MPI_Win_free(&_local_mtx_win);
		MPI_Win_free(&_local_atomic_win);
		MPI_Comm_free(&_local_comm);
	}
	
};

#undef _DPRINT

} // end namespace

#endif
