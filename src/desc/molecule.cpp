#include "desc/molecule.hpp"
#include "utils/mpi_log.hpp"
#include <utility>
#include <stdexcept>
#include <iostream>
#include <cmath>

#define printvec(LOG, n, v) \
	for (auto x : v) { \
		LOG.os<n>(x, " "); \
	} LOG.os<n>('\n');

namespace megalochem {

namespace desc {
	
// Taken from PSI4
static const std::vector<int> reference_S = {  0,
											   1,                                                                                           0,
											   1, 0,                                                                         1, 2, 3, 2, 1, 0,
											   1, 0,                                                                         1, 2, 3, 2, 1, 0,
											   1, 0,                                           1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
											   1, 0,                                           1, 2, 5, 6, 5, 4, 3, 0, 1, 0, 1, 2, 3, 2, 1, 0,
											   1, 0, 1, 0, 3, 4, 5, 6, 7, 8, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0 };

static const std::vector<int> conf_orb = {0, 2, 10, 18, 36, 54, 86, 118};

molecule::molecule(molecule::create_pack&& p) {
		
	m_comm = p.p_comm;
	m_name = p.p_name;
	m_mult = p.p_mult;
	m_charge = p.p_charge;
	m_mo_split = (p.p_mo_split) ? *p.p_mo_split : MOLECULE_MO_SPLIT;
	m_atoms = p.p_atoms;
	m_cluster_basis = p.p_cluster_basis;
	
	// secondly: occ/virt info
	
	m_frac = (p.p_fractional) ? *p.p_fractional : false;
	
	if (m_frac) {
		
		if (m_atoms.size() != 1) 
			throw std::runtime_error("Fractional Occupation only for single atoms for now.");
		
		// occupied/virtual info
		int Z = m_atoms[0].atomic_number;
		
		int nbas = m_cluster_basis->nbf();
		
		// total number of electrons
		m_nele = Z - m_charge;
		
		m_mult = reference_S[Z]; // mult is overwritten
		
		if ((p.p_spin_average) ? *p.p_spin_average : true) {
			
			m_nele_alpha = m_nele_beta = 0.5 * m_nele;
			
		} else {
			
			m_nele_alpha = 0.5 * (m_nele + m_mult);
			m_nele_beta = 0.5 * (m_nele - m_mult);
		
		}
		
		auto nlimit = std::lower_bound(conf_orb.begin(), conf_orb.end(), Z);
		
		if (Z < *nlimit) --nlimit;
		
		int ncore, nact;
		if ((*nlimit) == Z) {
		 // Special case: we can hit the boundary at the end of the array
			ncore = 0;
			nact = (*nlimit) / 2;
		} else {
			ncore = (*nlimit) / 2;
			nact = (*(++nlimit)) / 2 - ncore;
		}
	
		m_nocc_alpha = m_nocc_beta = nact + ncore;
		
		double nfraca = sqrt(((double)m_nele_alpha - (double)ncore)/(double)nact);
		double nfracb = sqrt(((double)m_nele_beta - (double)ncore)/(double)nact);
		
		m_nvir_alpha= m_nvir_beta = nbas - m_nocc_alpha;
		
		// form scaled occupation vector
		std::vector<double> occ_a(m_nocc_alpha, 1.0);
		std::vector<double> occ_b(m_nocc_beta, 1.0);
		
		for (int i = ncore; i < m_nocc_alpha; ++i) occ_a[i] = nfraca;
		for (int i = ncore; i < m_nocc_beta; ++i) occ_b[i] = nfracb;
		
		m_frac_occ_alpha = occ_a;
		m_frac_occ_beta = occ_b;
		
	} else {
		
		int nbas = m_cluster_basis->nbf();
	
		// total number of electrons
		m_nele = 0;
		for (size_t i = 0; i != m_atoms.size(); ++i) {
			m_nele += m_atoms[i].atomic_number;
		}
		
		m_nele -= m_charge;
		int nue = m_mult - 1; // number of unpaired electrons
			
		if ((m_nele - nue) % 2 != 0) 
			throw std::runtime_error("Mult not compatible with charge.");
			
		m_nele_alpha = m_nocc_alpha = (m_nele - nue) / 2 + nue;
		m_nele_beta = m_nocc_beta = (m_nele - nue) / 2;
			
		m_nvir_alpha = nbas - m_nocc_alpha;
		m_nvir_beta = nbas - m_nocc_beta;
	
	}
	
	molecule::block_sizes blks(*this, m_mo_split);
	
	m_blocks = blks;
	
}

void molecule::print_info(int level) {
	
	util::mpi_log LOG(m_comm,level);
	
	LOG.os<0>("Printing relevant info for molecule...\n");
	LOG.os<0>("Charge/multiplicity: ", m_charge, "/", m_mult, '\n');
	LOG.os<0>("Number of alpha electrons: ", m_nele_alpha, '\n');
	LOG.os<0>("Number of beta electrons: ", m_nele_beta, '\n');
	LOG.os<0>("Number of basis functions: ", m_cluster_basis->nbf(), '\n');
	LOG.os<0>("Number of occ. alpha orbs: ", m_nocc_alpha, '\n');
	LOG.os<0>("Number of occ. beta orbs: ", m_nocc_beta, '\n');
	LOG.os<0>("Number of vir. alpha orbs: ", m_nvir_alpha, '\n');
	LOG.os<0>("Number of vir. beta orbs: ", m_nvir_beta, '\n');
	
	LOG.os<1>("Block sizes: \n");
	LOG.os<1>("occ. alpha: "); printvec(LOG, 1, m_blocks.oa());
	LOG.os<1>("occ. beta: "); printvec(LOG, 1, m_blocks.ob());
	LOG.os<1>("vir. alpha: "); printvec(LOG, 1, m_blocks.va());
	LOG.os<1>("vir. beta: "); printvec(LOG, 1, m_blocks.vb());
	LOG.os<1>("basis: "); printvec(LOG, 1, m_blocks.b());
	
	if (m_cluster_dfbasis) {
		LOG.os<1>("dfbasis: "); printvec(LOG, 1, m_blocks.x());
	}
	
	if (m_frac_occ_alpha) {
		LOG.os<1>("Frac. occ. alpha: "); printvec(LOG, 1, *m_frac_occ_alpha);
	}
	
	if (m_frac_occ_beta) {
		LOG.os<1>("Frac. occ. beta: "); printvec(LOG, 1, *m_frac_occ_beta);
	}
	
	LOG.os<0>('\n');
	
}

std::shared_ptr<desc::molecule> molecule::fragment(int noa, int nob, int nva,
	int nvb, std::vector<int> atom_list)
{
	
	auto frag = std::make_shared<desc::molecule>();
	
	// basic members
	
	frag->m_name = "Fragment of " + m_name;
	frag->m_comm = m_comm;
	
	frag->m_mult = m_mult;
	frag->m_charge = m_charge;
	
	frag->m_mo_split = m_mo_split;
	frag->m_nele = m_nele;
	frag->m_nele_alpha = m_nele_alpha;
	frag->m_nele_beta = m_nele_beta;

	frag->m_nocc_alpha = noa;
	frag->m_nocc_beta = nob;
	frag->m_nvir_alpha = nva;
	frag->m_nvir_beta = nvb;
	
	// atoms
	
	for (auto iatom : atom_list) {
		frag->m_atoms.push_back(m_atoms[iatom]);
	}
	
	// basis set
	
	auto get_block = [&frag](desc::cluster_basis& cbas)
	{
		
		std::vector<bool> is_included(cbas.size(),false);
			
		int off = 0;
		for (auto& cluster : cbas) {
			if (atom_of(cluster[0], frag->m_atoms) != -1) {
				is_included[off] = true;
			}
			++off;
		}
		
		std::vector<int> blklist;
		for (size_t ishell = 0; ishell != is_included.size(); ++ishell) {
			if (is_included[ishell]) blklist.push_back(ishell);
		}
		
		return blklist;
		
	};
	
	auto blklist_b = get_block(*m_cluster_basis);
	frag->m_cluster_basis = m_cluster_basis->fragment(blklist_b);
	
	if (m_cluster_dfbasis) {
		auto blklist_x = get_block(*m_cluster_dfbasis);
		frag->m_cluster_dfbasis = m_cluster_dfbasis->fragment(blklist_x);
	}
	
	frag->m_blocks = block_sizes(*frag, m_mo_split);
	
	/*auto print = [](auto v) {
		for (auto e : v) {
			std::cout << e << " ";
		} std::cout << std::endl;
	};*/
	
	frag->m_atoms = m_atoms;
	
	/*print(frag->m_blocks.oa());
	print(frag->m_blocks.va());
	print(frag->m_blocks.b());
	print(frag->m_blocks.x());*/
	
	return frag;
	
}

}

} // end mega
