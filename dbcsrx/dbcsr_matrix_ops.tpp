#ifndef DBCSR_OPS_HPP
#define DBCSR_OPS_HPP

#:include "dbcsr.fypp"
#include <dbcsr_matrix.hpp>

namespace dbcsr {

template <typename T>
class multiply_base {

#:set list = [ &
    ['alpha', 'T', 'optional', 'val'],&
    ['beta', 'T', 'optional', 'val'],&
    ['first_row', 'int', 'optional', 'val'],&
    ['last_row', 'int', 'optional', 'val'],&
    ['first_col', 'int', 'optional', 'val'],&
    ['last_col', 'int', 'optional', 'val'],&
    ['first_k', 'int', 'optional', 'val'],&
    ['last_k', 'int', 'optional', 'val'],&
    ['retain_sparsity', 'bool', 'optional', 'val'],&
    ['filter_eps', 'double', 'optional', 'val'],&
    ['flop', 'long long int', 'optional', 'ref']] 
    
    ${make_param('multiply_base',list)}$
    
private:

    char m_transa, m_transb;
    matrix<T>& m_A; 
    matrix<T>& m_B; 
    matrix<T>& m_C;
    
public:

    multiply_base(char transa, char transb, matrix<T>& A, matrix<T>& B, matrix<T>& C) :
        m_transa(transa), m_transb(transb), m_A(A), m_B(B), m_C(C) {}
    
    ~multiply_base() {}

    void perform() {
        
        c_dbcsr_multiply(m_transa, m_transb, 
                        (c_alpha) ? *c_alpha : (T)1.0, 
                        m_A.m_matrix_ptr, m_B.m_matrix_ptr,
                        (c_beta) ? *c_beta : T(),
                        m_C.m_matrix_ptr,
                        (c_first_row) ? &*c_first_row : nullptr,
                        (c_last_row) ? &*c_last_row : nullptr,
                        (c_first_col) ? &*c_first_col : nullptr,
                        (c_last_col) ? &*c_last_col : nullptr,
                        (c_first_k) ? &*c_first_k : nullptr,
                        (c_last_k) ? &*c_last_k : nullptr,
                        (c_retain_sparsity) ? &*c_retain_sparsity : nullptr,
                        (c_filter_eps) ? &*c_filter_eps : &global::filter_eps, 
                        (c_flop) ? &*c_flop : nullptr);
    }

};

template <typename T>
multiply_base(char transa, char transb, matrix<T>& A, matrix<T>& B, matrix<T>& C) -> multiply_base<typename matrix<T>::value_type>;

template <typename T>
inline auto multiply(char transa, char transb, matrix<T>& A, matrix<T>& B, matrix<T>& C) -> multiply_base<T> {
	return multiply_base(transa,transb,A,B,C);
}

} // end nmespace

#endif
