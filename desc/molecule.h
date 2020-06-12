#ifndef DESC_MOLECULE
#define DESC_MOLECULE

#include "utils/params.hpp"
#include "utils/ppdirs.h"
#include "desc/basis.h"

#include <vector>
#include <libint2/atom.h>
#include <mpi.h>
#include <memory>

namespace desc {

// defaults
static const int MOLECULE_MO_SPLIT = 5;
static const std::string MOLECULE_AO_SPLIT_METHOD = "atomic";

class molecule {	
private:

	std::string m_name;

	int m_mult;
	int m_charge;
	std::vector<libint2::Atom> m_atoms;
	cluster_basis m_cluster_basis;
	optional<cluster_basis,val> m_cluster_dfbasis;
	
	int m_nocc_alpha;
	int m_nocc_beta;
	int m_nvir_alpha;
	int m_nvir_beta;
	
	int m_mo_split;
	std::string m_ao_split_method;
	
	int m_nele;
	double m_nele_alpha;
	double m_nele_beta;
	
	optional<std::vector<double>,val> m_frac_occ_alpha;
	optional<std::vector<double>,val> m_frac_occ_beta;
	
	bool m_frac = false;
	
	class block_sizes {
	private:
	
		std::vector<int> m_occ_alpha_sizes;
		std::vector<int> m_occ_beta_sizes;
		std::vector<int> m_vir_alpha_sizes;
		std::vector<int> m_vir_beta_sizes;
		std::vector<int> m_bas_sizes;
		std::vector<int> m_shell_sizes;
		optional<std::vector<int>,val> m_dfbas_sizes;
		optional<std::vector<int>,val> m_dfshell_sizes;
	
	public:
	
		block_sizes() {}
	
		block_sizes(molecule& mol, int nsplit) {
			
			m_occ_alpha_sizes = split_range(mol.m_nocc_alpha,nsplit);
			m_occ_beta_sizes = split_range(mol.m_nocc_beta,nsplit);
			m_vir_alpha_sizes = split_range(mol.m_nvir_alpha,nsplit);
			m_vir_beta_sizes = split_range(mol.m_nvir_beta,nsplit);
			
			m_bas_sizes = mol.m_cluster_basis.cluster_sizes();
			
			for (int i = 0; i != mol.m_cluster_basis.size(); ++i) {
				m_shell_sizes.push_back(mol.m_cluster_basis[i].size());
			}
			
			if (mol.m_cluster_dfbasis) {
				set_x(*mol.m_cluster_dfbasis);
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
		std::vector<int> s() { return m_shell_sizes; }
		std::vector<int> xs() {
			if (m_dfshell_sizes) {
				return *m_dfshell_sizes;
			} else {
				throw std::runtime_error("Df basis not given.");
			}
		}
		
		void set_x(cluster_basis& cdf) {
			
			optional<std::vector<int>,val> opt(cdf.cluster_sizes());
			m_dfbas_sizes = opt;
			
			opt->clear();
			for (int i = 0; i != cdf.size(); ++i) {
				opt->push_back(cdf[i].size());
			}
			m_dfshell_sizes = opt;
			
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
	
	struct create {
		
		make_param(create,name,std::string,required,val)
		make_param(create,atoms,std::vector<libint2::Atom>,required,ref)
		make_param(create,basis,std::vector<libint2::Shell>,required,ref)
		make_param(create,charge,int,required,val)
		make_param(create,mult,int,required,val)
		make_param(create,mo_split,int,optional,val)
		make_param(create,ao_split_method,std::string,optional,val)
		make_param(create,dfbasis,std::vector<libint2::Shell>,optional,ref)
		make_param(create,fractional,bool,optional,val)
		make_param(create,spin_average,bool,optional,val)
		
		public:
	
		create() {}
		friend class molecule;
		
	};
		
	molecule(create& p);

	~molecule() {}
	
	void print_info(MPI_Comm comm, int level = 0);
	
	void set_dfbasis(std::vector<libint2::Shell>& dfbasis);
	
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
	
	std::string ao_split_method() {
		return m_ao_split_method;
	}

	optional<std::vector<double>,val> frac_occ_alpha() {
		return m_frac_occ_alpha;
	}
	optional<std::vector<double>,val> frac_occ_beta() {
		return m_frac_occ_beta;
	}
	
	std::string name() {
		return m_name;
	}

};

using smolecule = std::shared_ptr<molecule>;

} // end namespace

#endif
