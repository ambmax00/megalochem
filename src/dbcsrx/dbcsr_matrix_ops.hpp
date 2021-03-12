#ifndef DBCSR_OPS_HPP
#define DBCSR_OPS_HPP

#ifndef TEST_MACRO
#include <dbcsr_matrix.hpp>
#endif

#include "utils/ppdirs.hpp"

namespace dbcsr {

template <typename T>
class multiply_base {
	
	typedef multiply_base _create_base;

#define MULT_BASE_LIST (\
	((util::optional<int>), first_row),\
	((util::optional<int>), first_col),\
	((util::optional<int>), last_row),\
	((util::optional<int>), last_col),\
	((util::optional<int>), first_k),\
	((util::optional<int>), last_k),\
	((util::optional<bool>), retain_sparsity),\
	((util::optional<double>), filter_eps),\
	((util::optional<long long int&>), flop))
    
    MAKE_BUILDER_MEMBERS(multiply, MULT_BASE_LIST)
        
private:

    char m_transa, m_transb;
    T m_alpha, m_beta;
    matrix<T>& m_A; 
    matrix<T>& m_B; 
    matrix<T>& m_C;
    
public:

	MAKE_BUILDER_SETS(multiply, MULT_BASE_LIST)

    multiply_base(char transa, char transb, T alpha, matrix<T>& A, matrix<T>& B, 
		T beta, matrix<T>& C) :
        m_transa(transa), m_transb(transb), m_alpha(alpha), m_A(A), 
        m_B(B), m_beta(beta), m_C(C) {}
    
    ~multiply_base() {}

    void perform() {
            
        c_dbcsr_multiply(m_transa, m_transb, 
                        m_alpha, m_A.m_matrix_ptr, m_B.m_matrix_ptr,
                        m_beta, m_C.m_matrix_ptr,
                        (c_first_row) ? &*c_first_row : nullptr,
                        (c_last_row) ? &*c_last_row : nullptr,
                        (c_first_col) ? &*c_first_col : nullptr,
                        (c_last_col) ? &*c_last_col : nullptr,
                        (c_first_k) ? &*c_first_k : nullptr,
                        (c_last_k) ? &*c_last_k : nullptr,
                        (c_retain_sparsity) ? &*c_retain_sparsity : nullptr,
                        (c_filter_eps) ? &*c_filter_eps : nullptr, 
                        (c_flop) ? &(*c_flop) : nullptr);
    }

};

template <typename T>
multiply_base(char transa, char transb, T alpha, matrix<T>& A, matrix<T>& B, 
	T beta, matrix<T>& C) -> multiply_base<typename matrix<T>::value_type>;

template <typename T>
inline auto multiply(char transa, char transb, T alpha, matrix<T>& A, 
	matrix<T>& B, T beta, matrix<T>& C) -> multiply_base<T> {
	return multiply_base(transa,transb,alpha,A,B,beta,C);
}

} // end nmespace

#endif
