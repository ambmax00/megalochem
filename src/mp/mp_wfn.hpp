#ifndef MP_MP_WFN_H
#define MP_MP_WFN_H

namespace megalochem {

namespace mp {
	
class mp_wfn : hf::hf_wfn {
protected:

	double m_mp_os_energy;
	double m_mp_ss_energy;
	double m_mp_energy; 
	
public:

	mp_wfn(
		double mp_os_energy,
		double mp_ss_energy,
		double mp_energy) :
		m_mp_os_energy(mp_os_energy),
		m_mp_ss_energy(mp_ss_energy),
		m_mp_energy(mp_energy)
	{}
	
	double mp_os_energy() { return m_mp_os_energy; }
	double mp_ss_energy() { return m_mp_ss_energy; }
	double mp_energy() { return m_mp_energy; }
	
	~mp_wfn() {}
	
};

using shared_mp_wfn = std::shared_ptr<mp_wfn>;

} // namespace mp

} // namespace megalochem
	
#endif
