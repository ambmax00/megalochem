#ifndef DESC_MOLECULE
#define DESC_MOLECULE

#include "utils/params.hpp"
#include "utils/ppdirs.h"
#include "desc/atom.h"
#include "desc/basis.h"

#include <vector>
#include <mpi.h>
#include <memory>

namespace desc {

// defaults
static const int MOLECULE_MO_SPLIT = 5;
static const std::string MOLECULE_AO_SPLIT_METHOD = "atomic";

class create_mol_base;

class molecule {	
protected:

	std::string m_name;
	MPI_Comm m_comm;

	int m_mult;
	int m_charge;
	std::vector<Atom> m_atoms;
	shared_cluster_basis m_cluster_basis;
	shared_cluster_basis m_cluster_dfbasis;
	
	int m_nocc_alpha;
	int m_nocc_beta;
	int m_nvir_alpha;
	int m_nvir_beta;
	
	int m_mo_split;
	std::string m_ao_split_method;
	
	int m_nele;
	double m_nele_alpha;
	double m_nele_beta;
	
	std::optional<std::vector<double>> m_frac_occ_alpha;
	std::optional<std::vector<double>> m_frac_occ_beta;
	
	bool m_frac = false;
	
	class block_sizes {
	private:
	
		std::vector<int> m_occ_alpha_sizes;
		std::vector<int> m_occ_beta_sizes;
		std::vector<int> m_vir_alpha_sizes;
		std::vector<int> m_vir_beta_sizes;
		std::vector<int> m_tot_alpha_sizes;
		std::vector<int> m_tot_beta_sizes;
		std::vector<int> m_bas_sizes;
		std::vector<int> m_shell_sizes;
		std::vector<int> m_dfbas_sizes;
		std::vector<int> m_dfshell_sizes;
	
	public:
	
		block_sizes() {}
	
		block_sizes(const block_sizes& in) = default;
		
		block_sizes& operator=(const block_sizes& in) = default;
	
		block_sizes(molecule& mol, int nsplit) {
			
			m_occ_alpha_sizes = split_range(mol.m_nocc_alpha,nsplit);
			m_occ_beta_sizes = split_range(mol.m_nocc_beta,nsplit);
			m_vir_alpha_sizes = split_range(mol.m_nvir_alpha,nsplit);
			m_vir_beta_sizes = split_range(mol.m_nvir_beta,nsplit);
			
			m_tot_alpha_sizes = m_occ_alpha_sizes;
			m_tot_beta_sizes = m_occ_beta_sizes;
			
			m_tot_alpha_sizes.insert(m_tot_alpha_sizes.end(),
				m_vir_alpha_sizes.begin(),m_vir_alpha_sizes.end());
				
			m_tot_beta_sizes.insert(m_tot_beta_sizes.end(),
				m_vir_beta_sizes.begin(),m_vir_beta_sizes.end());
				
			set_b(*mol.m_cluster_basis);
			
			if (mol.m_cluster_dfbasis) {
				set_x(*mol.m_cluster_dfbasis);
			}
			
		}
		
		std::vector<int> oa() { return m_occ_alpha_sizes; }
		std::vector<int> ob() { return m_occ_beta_sizes; }
		std::vector<int> va() { return m_vir_alpha_sizes; }
		std::vector<int> vb() { return m_vir_beta_sizes; }
		std::vector<int> ma() { return m_tot_alpha_sizes; }
		std::vector<int> mb() { return m_tot_beta_sizes; }
		std::vector<int> b() { return m_bas_sizes; }
		std::vector<int> x() { return m_dfbas_sizes; }
		std::vector<int> s() { return m_shell_sizes; }
		std::vector<int> xs() { return m_dfshell_sizes; }
		
		void set_b(cluster_basis& c) {
			
			m_bas_sizes = c.cluster_sizes();
			
			for (int i = 0; i != c.size(); ++i) {
				m_shell_sizes.push_back(c[i].size());
			}
			
		}	
		
		void set_x(cluster_basis& cdf) {
			
			m_dfbas_sizes = cdf.cluster_sizes();
			
			m_dfshell_sizes.clear();
			for (int i = 0; i != cdf.size(); ++i) {
				m_dfshell_sizes.push_back(cdf[i].size());
			}
			
		}			
		
		~block_sizes() {}
		
		std::vector<int> split_range(int n, int split) {
	
			// number of intervals
			int nblock = n%split == 0 ? n/split : n/split + 1;
			bool even = n%split == 0 ? true : false;
			
			if (even) {
				std::vector<int> out(nblock,split);
				return out;
			} else {
				std::vector<int> out(nblock,split);
				out[nblock-1] = n%split;
				return out;
			}
		}
		
	};
	
	block_sizes m_blocks;
		
	
public:

	molecule() {}
	
	~molecule() {}
	
	void print_info(int level = 0);
	
	void set_cluster_dfbasis(shared_cluster_basis& df) {
		m_cluster_dfbasis = df;
		m_blocks.set_x(*df);
	}
	
	shared_cluster_basis c_basis() {
		return m_cluster_basis;
	}
	
	shared_cluster_basis c_dfbasis() {
		return m_cluster_dfbasis;
	}
	
	std::vector<Atom> atoms() {
		return m_atoms;
	}
	
	block_sizes dims() {
		return m_blocks;
	}

	int nele_alpha() {
		return m_nele_alpha;
	}
	
	int nele_beta() {
		return m_nele_beta;
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
	
	int mo_split() {
		return m_mo_split;
	}

	int charge() {
		return m_charge;
	}
	
	int mult() {
		return m_mult;
	}

	std::optional<std::vector<double>> frac_occ_alpha() {
		return m_frac_occ_alpha;
	}
	std::optional<std::vector<double>> frac_occ_beta() {
		return m_frac_occ_beta;
	}
	
	std::string name() {
		return m_name;
	}
	
	std::shared_ptr<desc::molecule> fragment(int noa, int nob, int nvo,
		int nvb, std::vector<int> atom_list);
	
	friend class create_mol_base;

};

using smolecule = std::shared_ptr<molecule>;

class create_mol_base {
private:

	shared_cluster_basis c_basis;
	shared_cluster_basis c_dfbasis;
	
	make_param(create_mol_base,comm,MPI_Comm,required,val)
	make_param(create_mol_base,name,std::string,required,val)
	make_param(create_mol_base,atoms,std::vector<Atom>,required,ref)
	make_param(create_mol_base,charge,int,required,val)
	make_param(create_mol_base,mult,int,required,val)
	make_param(create_mol_base,mo_split,int,optional,val)
	make_param(create_mol_base,fractional,bool,optional,val)
	make_param(create_mol_base,spin_average,bool,optional,val)

public:

	create_mol_base() {}

	create_mol_base& basis(shared_cluster_basis& t_basis) {
		c_basis = t_basis; return *this;
	}
	
	create_mol_base& dfbasis(shared_cluster_basis& t_dfbasis) {
		c_dfbasis = t_dfbasis; return *this;
	}
	
	smolecule get();
	
};

inline create_mol_base create_molecule() {
	return create_mol_base();
}

} // end namespace

#endif
