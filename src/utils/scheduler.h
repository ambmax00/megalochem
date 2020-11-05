#ifndef UTIL_SCHEDULER_H
#define UTIL_SCHEDULER_H

#include <deque>
#include <vector>
#include <mpi.h>
#include <functional>

namespace util {

class scheduler {
private:

	const int64_t NO_RESP = -1;
	const int64_t NO_TASK = -2;
	const int NO_REQU = -1;
	const int NO_WORK = 0;
	const int HAS_WORK = 1;

	MPI_Comm m_comm;
	int m_mpisize;
	int m_mpirank;
	int64_t m_ntasks;
	
	std::vector<int64_t> m_transfer;
	std::vector<int> m_request;
	std::vector<int> m_status;
	
	MPI_Win m_transfer_win = MPI_WIN_NULL;
	MPI_Win m_request_win = MPI_WIN_NULL;
	MPI_Win m_status_win = MPI_WIN_NULL;
	
	std::deque<int64_t> m_local_tasks; 
	std::function<void(int64_t)>& m_executer;
	
	void init_deque() {
		for (int64_t i = 0; i != m_ntasks; ++i) {
			int proc = i % m_mpisize;
			if (proc == m_mpirank) m_local_tasks.push_back(i);
		}
	}
	
	void update_status() {
		//std::cout << m_mpirank << " : " << "UPDATE STATUS" << std::endl;
		m_status[m_mpirank] = (m_local_tasks.empty()) ? 
			NO_WORK : HAS_WORK;
	}
	
	void add_task(int64_t t) {
		//std::cout << m_mpirank << " : " << "ADD TASK" << std::endl;
		m_local_tasks.push_front(t);
		update_status();
	}
	
	void acquire() {
		std::cout << m_mpirank << " : " << "ACQUIRE" << std::endl;
		while (true) {
			
			m_transfer[m_mpirank] = NO_RESP;
			
			int victim = rand() % m_mpisize;
			int v_work = NO_WORK;
			
			//std::cout << m_mpirank << " : " << "STEALING FROM " << victim << std::endl;
			
			// get status of victim
			MPI_Get(&v_work, 1, MPI_INT, victim, victim, 1, 
				MPI_INT, m_status_win);
				
			//std::cout << ((v_work == NO_WORK) ? "NO_WORK" : "HAS_WORK") << std::endl;
			
			// continue if victim has no work
			if (v_work == NO_WORK) {
				communicate();
				//usleep(1000000);
				return;
			}
			
			// make victim aware that task can be stolen
			//std::cout << "PROC: " << m_mpirank << " WRITING" << std::endl;
			int result = NO_REQU;
			int r = MPI_Compare_and_swap(&m_mpirank, &NO_REQU, &result, MPI_INT,
				victim, victim, m_request_win);
			
			// continue if not succeeded 
			if (result != NO_REQU) {
				//std::cout << m_mpirank << " : " << "CAS FAILED" << std::endl;
				communicate();
				//usleep(1000000);
				return;
			}
			
			//std::cout << m_mpirank << " : " << "CAS SUCCEEDED, NOW WAITING" << std::endl;
			
			// loop and check if request accepted
			while (m_transfer[m_mpirank] == NO_RESP) {
				communicate();
				//usleep(1000000);

			}
			
			std::cout << m_mpirank << " : " << "GOT RESPONSE FROM VICTIM: " 
				<< m_transfer[m_mpirank] << std::endl;
			
			if (m_transfer[m_mpirank] != NO_TASK) {
				//std::cout << m_mpirank << " : " << "ADDING TASK " << 
				//	m_transfer[m_mpirank] << " FROM VICTIM" << std::endl;
				add_task(m_transfer[m_mpirank]);
				m_request[m_mpirank] = NO_REQU;
				//usleep(1000000);
				return;
			}
			
			communicate();
			
		}
	}
	
	void communicate() {
		
		// check if there is a processor request
		int thief = m_request[m_mpirank];
		
		if (thief != NO_REQU) {
			//std::cout << m_mpirank << " : " 
			//	<< " GOT REQUEST FROM THIEF " << thief << std::endl;
		} else {
			//std::cout << m_mpirank << " : " << "NO REQUEST" << std::endl;
		}
		
		//usleep(500000);
		
		if (thief == NO_REQU) return;
		
		int64_t t = NO_TASK;
		
		if (!m_local_tasks.empty()) {
			t = m_local_tasks[0];
			m_local_tasks.pop_front();
		}
		
		//std::cout << m_mpirank << " : " << "PUTTING task " << t << " on THIEF " 
		//	<< thief << std::endl;
		MPI_Put(&t, 1, MPI_LONG_LONG, thief, thief, 1, MPI_LONG_LONG,
			m_transfer_win);
		
		//usleep(500000);
		
		m_request[m_mpirank] = NO_REQU;
		
	}
	
	void execute(int64_t task) {
		//std::cout << m_mpirank << " : " << "EXECUTE" << std::endl;
		m_executer(task);
	}
			
	
public:

	scheduler(MPI_Comm comm, int64_t ntasks, std::function<void(int64_t)>& executer) 
		: m_comm(comm), m_executer(executer), m_ntasks(ntasks)
	{
		
		std::cout << "NTASKS: " << m_ntasks << std::endl;
		
		MPI_Comm_size(comm, &m_mpisize);
		MPI_Comm_rank(comm, &m_mpirank);
		init_deque();
		
		m_transfer.resize(m_mpisize, NO_RESP);
		m_request.resize(m_mpisize, NO_REQU);
		m_status.resize(m_mpisize, NO_WORK);
		
		std::cout << "MY QUEUE: " << std::endl;
		for (auto q : m_local_tasks) {
			std::cout << q << " ";
		} std::cout << '\n';
		
		MPI_Win_create(m_transfer.data(), m_mpisize * sizeof(int64_t),
			sizeof(int64_t), MPI_INFO_NULL, m_comm, &m_transfer_win);
			
		MPI_Win_create(m_request.data(), m_mpisize * sizeof(int),
			sizeof(int), MPI_INFO_NULL, m_comm, &m_request_win);
			
		MPI_Win_create(m_status.data(), m_mpisize * sizeof(int),
			sizeof(int), MPI_INFO_NULL, m_comm, &m_status_win);		
			
		MPI_Win_fence(0, m_transfer_win); 
		MPI_Win_fence(0, m_request_win); 
		MPI_Win_fence(0, m_status_win); 
		
	}
	
	void run() {
		
		const int nmax_run = 100;
		int nrun = 0;
		
		while (true) {
			if (m_local_tasks.empty()) {
				acquire();
			} else {
				//std::cout << "WORKER ON PROC " << m_mpirank << std::endl;
				int64_t task = m_local_tasks.back();
				m_local_tasks.pop_back();
				update_status();
				communicate();
				execute(task);
			}
			
			
			// check every namx_run iterations wether all procs are empty
			if (nrun++ > nmax_run) {
				
				MPI_Request barrier_request;
				int flag = 0;
				
				//std::cout << "FLAGGING" << std::endl;
				MPI_Ibarrier(m_comm,&barrier_request);
				
				while (!flag) {
					MPI_Test(&barrier_request, &flag, MPI_STATUS_IGNORE);
					communicate();
				}
				
				//std::cout << m_mpirank << " : " << "REDUCE" << std::endl;
				
				int empty = m_local_tasks.empty() ? 1 : 0;
				int all_empty = 0;
				//if (empty) std::cout << m_mpirank << " : " << "IS EMPTY" << std::endl;
				MPI_Allreduce(&empty, &all_empty, 1, MPI_INT, MPI_LAND, m_comm);
				
				//std::cout << m_mpirank << " : " << "CONTINUEING" << std::endl;
				
				// ==== EXIT ====
				if (all_empty) return;
				nrun = 0;
			}
		}
		
	}
	
	~scheduler() {
		
		MPI_Win_fence(0, m_transfer_win); 
		MPI_Win_fence(0, m_request_win); 
		MPI_Win_fence(0, m_status_win); 
		
		MPI_Win_free(&m_transfer_win);
		MPI_Win_free(&m_request_win);
		MPI_Win_free(&m_status_win);
	}
	
};

} // end namespace

#endif
