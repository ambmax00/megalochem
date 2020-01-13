#ifndef DESC_MOLECULE
#define DESC_MOLECULE

#include "utils/params.hpp"
#include "desc/basis.h"

#include <vector>
#include <libint2/atom.h>

namespace desc {

class molecule {	
private:

	int m_mult;
	int m_charge;
	std::vector<libint2::Atom> m_atoms;
	cluster_basis m_cluster_basis;
	optional<cluster_basis,val> m_cluster_dfbasis;
	
	int m_nocc_alpha;
	int m_nocc_beta;
	int m_nvir_alpha;
	int m_nvir_beta;
	
	int m_nele;
	int m_nele_alpha;
	int m_nele_beta;
	
	optional<std::vector<int>,val> m_frac_occ_alpha;
	optional<std::vector<int>,val> m_frac_occ_beta;
	
	bool m_frac = false;
	
	class block_sizes {
	private:
	
		std::vector<int> m_occ_alpha_sizes;
		std::vector<int> m_occ_beta_sizes;
		std::vector<int> m_vir_alpha_sizes;
		std::vector<int> m_vir_beta_sizes;
		std::vector<int> m_bas_sizes;
		optional<std::vector<int>,val> m_dfbas_sizes;
	
	public:
	
		block_sizes() {}
	
		block_sizes(molecule& mol, int nsplit) {
			
			auto split_range = [](int n, int split) {
	
				// number of intervals
				int nblock = n%split == 0 ? n/split : n/split + 1;
				bool even = n%split == 0 ? true : false;
				
				if (even) {
					std::vector<int> out(nblock,n/split);
					return out;
				} else {
					std::vector<int> out(nblock,n/split);
					out[nblock-1] = n%split;
					return out;
				}
			};
			
			m_occ_alpha_sizes = split_range(mol.m_nocc_alpha,nsplit);
			m_occ_beta_sizes = split_range(mol.m_nocc_beta,nsplit);
			m_vir_alpha_sizes = split_range(mol.m_nvir_alpha,nsplit);
			m_vir_beta_sizes = split_range(mol.m_nvir_beta,nsplit);
			
			m_bas_sizes = mol.m_cluster_basis.cluster_sizes();
			if (mol.m_cluster_dfbasis) {
				std::cout << "ITS IN BLOCK." << std::endl;
				optional<std::vector<int>,val> opt(mol.m_cluster_dfbasis->cluster_sizes());
				m_dfbas_sizes = opt;
			}		
		}
		
		std::vector<int> oa() { return m_occ_alpha_sizes; }
		std::vector<int> ob() { return m_occ_beta_sizes; }
		std::vector<int> va() { return m_vir_alpha_sizes; }
		std::vector<int> vb() { return m_vir_beta_sizes; }
		std::vector<int> b() { return m_bas_sizes; }
		std::vector<int> x() { 
			if (m_dfbas_sizes) {
				return *m_dfbas_sizes;
			} else {
				throw std::runtime_error("Df basis not given.");
			}
		}
		
		~block_sizes() {}
		
	};
	
	block_sizes m_blocks;
		
	
public:

	molecule() {}
	
	struct mol_params {
		required<std::vector<libint2::Atom>,ref>	atoms;
		required<int,val>							charge;
		required<int,val>							mult;
		required<int,val>							split;
		required<std::vector<libint2::Shell>,ref>	basis;
		optional<std::vector<libint2::Shell>,ref>	dfbasis;
		optional<bool,val>							fractional;
	};
	molecule(mol_params&& p);

	~molecule() {}
	
	void print_info(int level = 0);
	
	cluster_basis c_basis() {
		return m_cluster_basis;
	}
	
	optional<cluster_basis,val> c_dfbasis() {
		return m_cluster_dfbasis;
	}
	
	std::vector<libint2::Atom> atoms() {
		return m_atoms;
	}
	
	block_sizes dims() {
		return m_blocks;
	}

	int nocc_alpha() {
		return m_nocc_alpha;
	}
	
	int nocc_beta() {
		return m_nocc_beta;
	}
	
	int nvir_alpha() {
		return m_nvir_alpha;
	}
	
	int nvir_beta() {
		return m_nvir_beta;
	}

};

}

#endif
