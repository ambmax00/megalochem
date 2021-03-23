#ifndef DESC_MOLECULE
#define DESC_MOLECULE

#ifndef TEST_MACRO
#include "desc/atom.hpp"
#include "desc/basis.hpp"
#include "io/data_handler.hpp"
#include <vector>
#include <mpi.h>
#include <memory>
#endif

#include "utils/ppdirs.hpp"

namespace desc {

// defaults
static const int MOLECULE_MO_SPLIT = 5;
static const std::string MOLECULE_AO_SPLIT_METHOD = "atomic";

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

#define MOLECULE_CREATE_LIST (\
	((MPI_Comm), comm),\
	((std::string),name),\
	((std::vector<Atom>),atoms),\
	((int), charge),\
	((int),mult),\
	((util::optional<int>),mo_split),\
	((util::optional<bool>),fractional),\
	((util::optional<bool>),spin_average),\
	((shared_cluster_basis),cluster_basis))
	
	MAKE_PARAM_STRUCT(create, MOLECULE_CREATE_LIST, ())
	MAKE_BUILDER_CLASS(molecule, create, MOLECULE_CREATE_LIST, ())

	molecule(create_pack&& p);
	
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
	
};

using shared_molecule = std::shared_ptr<molecule>;

inline void write_molecule(std::string name, desc::molecule& mol, filio::data_handler& dh) {

	dh.open(filio::access_mode::rdwr);
	dh.create_group(name);
	
	// write constants
	dh.write<std::string>(name + "/name", mol.name());
	dh.write<int>(name + "/charge", mol.charge());
	dh.write<int>(name + "/mult", mol.mult());
	dh.write<int>(name + "/mo_split", mol.mo_split());
	
	// write atoms
	auto atoms = mol.atoms();
	
	hsize_t natoms = atoms.size();
	hsize_t slots = 4;
	
	std::vector<double> atom_vec;
	
	for (auto a : atoms) {
		atom_vec.push_back((double)a.atomic_number);
		atom_vec.push_back(a.x);
		atom_vec.push_back(a.y);
		atom_vec.push_back(a.z);
	}
	
	dh.write<double>(name + "/atoms", atom_vec.data(), {natoms,slots});
	
	dh.close();
	
}

} // end namespace

#endif
