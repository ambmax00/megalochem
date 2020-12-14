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
	const int64_t NO_RESP = -1;
	const int64_t NO_TASK = -2;
	
	// for request
	const int NO_REQU = -1;
	
	// for status
	const int NO_WORK = 0;
	const int HAS_WORK = 1;

	// for token
	const int NO_TOKEN = -1;
	const int TERMINATE = -2;

	MPI_Comm m_comm;
	int m_mpisize;
	int m_mpirank;
	int64_t m_ntasks;
	
	bool m_terminate_flag;
	
	std::vector<int64_t> m_transfer;
	std::vector<int> m_request;
	std::vector<int> m_status;
	std::vector<int> m_token;
	
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
		m_status[m_mpirank] = (m_local_tasks.empty()) ? NO_WORK : HAS_WORK;
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
			
			m_transfer[m_mpirank] = NO_RESP;
			
			#ifdef _DLOG
			std::cout << m_mpirank << " : " << "STATUS " << m_status[m_mpirank] << std::endl;
			std::cout << m_mpirank << " : " << "QSIZE " << m_local_tasks.size() << std::endl;
			#endif
			
			int victim = get_rand();
			int v_work = NO_WORK;
			
			#ifdef _DLOG
			std::cout << m_mpirank << " : " << "STEALING FROM " << victim << std::endl;
			#endif
			
			// get status of victim
			MPI_Get(&v_work, 1, MPI_INT, victim, victim, 1, MPI_INT, m_status_win);
			MPI_Win_flush_local(victim, m_status_win);
							
			// continue if victim has no work
			if (v_work == NO_WORK) {
				communicate();
				continue;
			}
			
			// make victim aware that task can be stolen
			int result = NO_REQU;
			
			MPI_Compare_and_swap(&m_mpirank, &NO_REQU, &result, MPI_INT,
				victim, victim, m_request_win);
			MPI_Win_flush(victim, m_request_win);
						
			// continue if not succeeded 
			if (result != NO_REQU) {
				communicate();
				continue;
			}
			
			#ifdef _DLOG
			std::cout << m_mpirank << " : " << "CAS SUCCEEDED, NOW WAITING" << std::endl;
			#endif
			
			// loop and check if request accepted
			while (m_transfer[m_mpirank] == NO_RESP && !terminate()) {
				communicate();
			}
			
			#ifdef _DLOG
			std::cout << m_mpirank << " : " << "GOT RESPONSE FROM VICTIM: " 
				<< m_transfer[m_mpirank] << std::endl;
			#endif
			
			if (m_transfer[m_mpirank] != NO_TASK) {
				m_request[m_mpirank] = NO_REQU;
				add_task(m_transfer[m_mpirank]);
				return;
			}
			
			communicate();
			
		}
	}
	
	void transfer_token() {
				
		//int mytoken = NO_TOKEN;
		
		// fetch my token (atomically)
		int mytoken = m_token[m_mpirank];
		
		//MPI_Fetch_and_op(&NO_TOKEN, &mytoken, MPI_INT, m_mpirank, m_mpirank,
		//	MPI_REPLACE, m_token_win);
		//MPI_Win_flush(m_mpirank, m_token_win);
		
		if (mytoken < -2 || mytoken > m_mpisize) {
			throw std::runtime_error("Scheduler: something went wrong.");
		}
		
		#ifdef _DLOG
		std::cout << m_mpirank << " : " << "TOKEN " << mytoken << std::endl;
		#endif
		
		// no token -> nothing to do
		if (mytoken == NO_TOKEN) return;
		
		// update my token
		if (mytoken != TERMINATE) {
			
			if (m_mpirank == 0 && mytoken == 0) {
				mytoken = TERMINATE;
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
			
			m_token[m_mpirank] = NO_TOKEN;
			MPI_Put(&mytoken, 1, MPI_INT, neigh, neigh, 1, MPI_INT, m_token_win);
			MPI_Win_flush_local(neigh, m_token_win);
			
		}	
		
		if (mytoken == TERMINATE) {
			m_terminate_flag = true;
		}
		
	}
	
	void communicate() {
		
		#ifdef _DLOG
		std::cout << m_mpirank << " : " << "COMM" << std::endl;
		#endif
		
		// token
		transfer_token();
		
		// check if there is a processor request
		int thief = m_request[m_mpirank]; // NO_REQU;
		
		//MPI_Fetch_and_op(&NO_REQU, &thief, MPI_INT, m_mpirank, m_mpirank,
		//	MPI_REPLACE, m_request_win);
		//MPI_Win_flush(m_mpirank, m_request_win);
				
		if (thief == NO_REQU) return;
		
		int64_t t = NO_TASK;
		
		if (!m_local_tasks.empty()) {
			t = m_local_tasks[0];
			m_local_tasks.pop_front();
			update_status();
		}
		
		#ifdef _DLOG
		std::cout << m_mpirank << " : " << "PUTTING task " << t << " on THIEF " 
			<< thief << std::endl;
		#endif
		
		MPI_Put(&t, 1, MPI_LONG_LONG, thief, thief, 1, MPI_LONG_LONG,
			m_transfer_win);
		MPI_Win_flush(thief, m_transfer_win);
		
		m_request[m_mpirank] == NO_REQU;
						
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
		
		m_transfer.resize(m_mpisize, NO_RESP);
		m_request.resize(m_mpisize, NO_REQU);
		m_status.resize(m_mpisize, NO_WORK);
		m_token.resize(m_mpisize, NO_TOKEN);
		
	}
	
	void run() {
				
		MPI_Win_create(m_transfer.data(), m_mpisize * sizeof(int64_t),
			sizeof(int64_t), MPI_INFO_NULL, m_comm, &m_transfer_win);
			
		MPI_Win_create(m_request.data(), m_mpisize * sizeof(int),
			sizeof(int), MPI_INFO_NULL, m_comm, &m_request_win);
			
		MPI_Win_create(m_status.data(), m_mpisize * sizeof(int),
			sizeof(int), MPI_INFO_NULL, m_comm, &m_status_win);
			
		MPI_Win_create(m_token.data(), m_mpisize * sizeof(int),
			sizeof(int), MPI_INFO_NULL, m_comm, &m_token_win);	
		
		std::fill(m_transfer.begin(), m_transfer.end(), NO_RESP);
		std::fill(m_request.begin(), m_request.end(), NO_REQU);
		std::fill(m_status.begin(), m_status.end(), NO_WORK);
		std::fill(m_token.begin(), m_token.end(), NO_TOKEN);
		
		m_terminate_flag = false;
		if (m_mpirank == 0) m_token[0] = 1;
		
		MPI_Win_lock_all(0, m_transfer_win);
		MPI_Win_lock_all(0, m_request_win);
		MPI_Win_lock_all(0, m_status_win);
		MPI_Win_lock_all(0, m_token_win);
		
		while (!terminate()) {
			
			if (m_local_tasks.empty()) {
				acquire();
			} else {
				int64_t task = m_local_tasks.back();
				m_local_tasks.pop_back();
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
		
		MPI_Win_unlock_all(m_transfer_win);
		MPI_Win_unlock_all(m_request_win);
		MPI_Win_unlock_all(m_status_win);
		MPI_Win_unlock_all(m_token_win);
		
		MPI_Win_free(&m_transfer_win);
		MPI_Win_free(&m_request_win);
		MPI_Win_free(&m_status_win);
		MPI_Win_free(&m_token_win);
				
	}
	
	~scheduler() {}
	
};

} // end namespace

#endif
