#ifndef MP_MP_WFN_H
#define MP_MP_WFN_H

#include "hf/hf_wfn.hpp"

namespace megalochem {

namespace mp {
	
class mpmod;

class mp_wfn : hf::hf_wfn {
protected:

	double m_mp_os_energy;
	double m_mp_ss_energy;
	double m_mp_energy; 
	
public:

	mp_wfn(const hf::hf_wfn& in) : hf_wfn(in) {}
	
	friend class mpmod;
	
};

using smp_wfn = std::shared_ptr<mp_wfn>;

} // namespace mp

} // namespace megalochem
	
#endif
