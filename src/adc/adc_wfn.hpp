#ifndef ADC_WFN_HPP
#define ADC_WFN_HPP

#include <dbcsr_matrix.hpp>
#include <memory>

namespace megalochem {
	
namespace adc {
	
class adc_wfn {
private:

	std::vector<double> m_dav_eigvals;
	std::vector<dbcsr::shared_matrix<double>> m_dav_eigvecs;
	
	bool m_blocked; // indicates if all roots are converged
	
public:

	adc_wfn(
		bool blocked,
		std::vector<double> dav_eigvals,
		std::vector<dbcsr::shared_matrix<double>> dav_eigvecs) :
		m_blocked(blocked),
		m_dav_eigvals(dav_eigvals),
		m_dav_eigvecs(dav_eigvecs)
	{}
	
	bool blocked() { return m_blocked; }
	
	std::vector<double> davidson_eigenvalues() {
		return m_dav_eigvals;
	}
	
	std::vector<double> davidson_eigenvectors() {
		return m_dav_eigvecs;
	}
		
	~adc_wfn() {}
	
	
};

using shared_adc_wfn = std::shared_ptr<adc_wfn>;

} // namespace adc

} // namespace mega

#endif
