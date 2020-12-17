#ifndef UTIL_SCHEDULER_H
#define UTIL_SCHEDULER_H

#include <deque>
#include <vector>
#include <mpi.h>
#include <functional>
#include <algorithm>
#include <random>
#include <unistd.h>

#define _DLOG

namespace util {

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
	
	void init_deque() {
		for (int64_t i = 0; i != m_ntasks; ++i) {
			int proc = i % m_mpisize;
			if (proc == m_mpirank) m_local_tasks.push_back(i);
		}
	}
	
	void update_status() {
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, m_mpirank, MPI_MODE, m_status_win);
		m_status[m_mpirank] = (m_local_tasks.empty()) ? 
			STATUS_NO_WORK : STATUS_HAS_WORK;
		MPI_Win_unlock(m_mpirank, m_status_win);
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
			
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, m_mpirank, MPI_MODE, m_transfer_win);
			m_transfer[m_mpirank] = TRANSFER_NO_RESP;
			MPI_Win_unlock(m_mpirank, m_transfer_win);
			
			int victim = get_rand();
			int v_work = STATUS_NO_WORK;
			
			// get status of victim
			MPI_Win_lock(MPI_LOCK_SHARED, victim, MPI_MODE, m_status_win);  
			MPI_Get(&v_work, 1, MPI_INT, victim, victim, 1, MPI_INT, m_status_win);
			MPI_Win_unlock(victim, m_status_win);
							
			// continue if victim has no work
			if (v_work == STATUS_NO_WORK) {
				communicate();
				continue;
			}
			
			// make victim aware that task can be stolen
			int result = REQUEST_NONE;
			
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, victim, MPI_MODE, m_request_win);
			MPI_Compare_and_swap(&m_mpirank, &REQUEST_NONE, &result, MPI_INT,
				victim, victim, m_request_win);
			MPI_Win_unlock(victim, m_request_win);
						
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
				
				MPI_Win_lock(MPI_LOCK_SHARED, m_mpirank, MPI_MODE, m_transfer_win);
				newtask = m_transfer[m_mpirank];
				MPI_Win_unlock(m_mpirank, m_transfer_win);
				
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
						
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, m_mpirank, MPI_MODE, m_token_win);
		int mytoken = m_token[m_mpirank];
		MPI_Win_unlock(m_mpirank, m_token_win);
		
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
			
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, neigh, MPI_MODE, m_token_win);
			MPI_Put(&mytoken, 1, MPI_INT, neigh, neigh, 1, MPI_INT, m_token_win);
			m_token[m_mpirank] = TOKEN_NONE;
			MPI_Win_unlock(neigh, m_token_win);
			
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
		
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, m_mpirank, MPI_MODE, m_request_win);
		MPI_Fetch_and_op(&REQUEST_NONE, &thief, MPI_INT, m_mpirank, m_mpirank,
			MPI_REPLACE, m_request_win);
		MPI_Win_unlock(m_mpirank, m_request_win);
                   				
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
		
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, thief, MPI_MODE, m_transfer_win);
		MPI_Put(&t, 1, MPI_LONG_LONG, thief, thief, 1, MPI_LONG_LONG,
			m_transfer_win);
		MPI_Win_unlock(thief, m_transfer_win);
		
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, m_mpirank, MPI_MODE, m_request_win);
		m_request[m_mpirank] = REQUEST_NONE;
		MPI_Win_unlock(m_mpirank, m_request_win);
						
	}
	
	void execute(int64_t task) {
		#ifdef _DLOG
		std::cout << m_mpirank << " : " << "EXECUTE" << std::endl;
		#endif
		m_executer(task);
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
		
		MPI_Barrier(m_comm);

		MPI_Win_create(m_transfer, m_mpisize * sizeof(int64_t), sizeof(int64_t), 
			MPI_INFO_NULL, m_comm, &m_transfer_win);
			
		MPI_Win_create(m_request, m_mpisize * sizeof(int), sizeof(int), 
			MPI_INFO_NULL, m_comm, &m_request_win);
			
		MPI_Win_create(m_status, m_mpisize * sizeof(int), sizeof(int), 
			MPI_INFO_NULL, m_comm, &m_status_win);
			
		MPI_Win_create(m_token, m_mpisize * sizeof(int), sizeof(int), 
			MPI_INFO_NULL, m_comm, &m_token_win);
				
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
		
		MPI_Win_free(&m_transfer_win);
		MPI_Win_free(&m_request_win);
		MPI_Win_free(&m_status_win);
		MPI_Win_free(&m_token_win);
		
		MPI_Free_mem(m_transfer);
		MPI_Free_mem(m_status);
		MPI_Free_mem(m_token);
		MPI_Free_mem(m_request);
				
	}
	
	~scheduler() {}
	
};

} // end namespace

#endif
