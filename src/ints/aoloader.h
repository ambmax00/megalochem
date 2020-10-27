#ifndef AOLOADER_H
#define AOLOADER_H

#include <dbcsr_common.hpp>
#include "ints/aofactory.h"
#include "ints/screening.h"
#include "utils/mpi_time.h"
#include "utils/registry.h"
#include "desc/options.h"
#include "desc/molecule.h"

namespace ints {

enum class key {
	pgrid2 = 0,
	pgrid3,
	pgrid4, // processor grids
	ovlp_bb, // overlap matrix
	ovlp_xx, // overlap matrix for aux basis
	pot_bb, // nuc. potential ints
	kin_bb, // kin. ints
	ovlp_bb_inv, 
	ovlp_xx_inv,
	coul_xbb, // 3c2e integrals
	erfc_xbb, // 3c2e integrals in coulomb metric
	coul_xx, // 2c2e ints in coulomb metric
	erfc_xx, // 2c2e integrals in erfc metric 
	coul_bbbb, // 4c2e integrals
	coul_xx_inv,
	erfc_xx_inv,
	coul_xx_invsqrt,
	erfc_xx_invsqrt,
	dfit_coul_xbb, // fitting coefficients
	dfit_erfc_xbb,
	dfit_pari_xbb,
	dfit_qr_xbb,
	scr_xbb,
	NUM_KEYS
};
	
class aoloader {
private:	
	
	dbcsr::world m_world;
	desc::options m_opt;
	util::mpi_log LOG;
	util::mpi_time TIME;
	desc::smolecule m_mol;
	std::shared_ptr<ints::aofactory> m_aofac;

	util::key_registry<key> m_reg;

	std::array<bool,static_cast<const int>(key::NUM_KEYS)> m_to_compute;

	std::pair<dbcsr::shared_matrix<double>,dbcsr::shared_matrix<double>>
		invert(dbcsr::shared_matrix<double> mat);

	bool comp(key k) {
		return m_to_compute[static_cast<int>(k)];
	}

public:

	aoloader(dbcsr::world w, desc::smolecule smol, desc::options opt) :
		m_aofac(std::make_shared<ints::aofactory>(smol,w)),
		m_world(w), m_opt(opt), m_mol(smol),
		LOG(w.comm(), opt.get<int>("print",0)),
		TIME(w.comm(), "AO-loader") 
	{
		for (auto& a : m_to_compute) a = false;		
	}
		
	aoloader& request(key k) {
		m_to_compute[static_cast<int>(k)] = true;
		return *this;
	}
	
	void compute();
		 
	~aoloader() {}
	
	void print_info() {
		TIME.print_info();
	}
	
	const util::key_registry<key> get_registry() {
		return m_reg;
	}
	
}; // end class
	
	
} // end namespace

#endif
