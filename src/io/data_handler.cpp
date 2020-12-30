#include "io/data_handler.h"
#include <H5Cpp.h>
#include <numeric>
#include <vector>

namespace filio {
	
void write_basis(H5::H5File* file, std::string path, desc::cluster_basis& cbasis) {
	
		std::vector<int> shell_env;
		std::vector<double> shell_data;
		
		std::vector<int> csizes;
		for (auto& c : cbasis) {
			csizes.push_back(c.size());
		}
		
		std::string pathenv = path + "/shells_env";
		std::string pathdata = path + "/shells_data";
		std::string pathsizes = path + "/cluster_sizes";
		
		for (auto& cluster : cbasis) {
			for (auto& shell : cluster) {
				
				shell_env.push_back(shell.l);
				shell_env.push_back(shell.nprim());
								
				shell_data.push_back(shell.O[0]);
				shell_data.push_back(shell.O[1]);
				shell_data.push_back(shell.O[2]);
								
				auto& coeff = shell.coeff;
				auto& alpha = shell.alpha;
				
				shell_data.insert(shell_data.end(), alpha.begin(), alpha.end());
				shell_data.insert(shell_data.end(), coeff.begin(), coeff.end());
				
			}
		}
		
		int rank = 1;
		hsize_t size_env = shell_env.size();
		hsize_t size_data = shell_data.size();
		hsize_t size_csizes = csizes.size();
		
		auto type_double = H5toCpp<double>::type();
		auto type_int = H5toCpp<int>::type();
		
		H5::DataSpace dataspace_env(rank, &size_env);
		H5::DataSpace dataspace_data(rank, &size_data);
		H5::DataSpace dataspace_csizes(rank, &size_csizes);
		
		H5::DataSet dataset_env(file->createDataSet(pathenv,
			type_int, dataspace_env));
		H5::DataSet dataset_data(file->createDataSet(pathdata,
			type_double, dataspace_data));	
		H5::DataSet dataset_csizes(file->createDataSet(pathsizes,
			type_int, dataspace_csizes));
			
		dataset_env.write(shell_env.data(), type_int);
		dataset_data.write(shell_data.data(), type_double);
		dataset_csizes.write(csizes.data(), type_int);
				
}

desc::cluster_basis read_basis(H5::H5File* file, std::string path) {
	
	std::string pathenv = path + "/shells_env";
	std::string pathdata = path + "/shells_data";
	std::string pathsizes = path + "/cluster_sizes";
	
	H5::DataSet dset_env = file->openDataSet(pathenv);
	H5::DataSet dset_data = file->openDataSet(pathdata);
	H5::DataSet dset_sizes = file->openDataSet(pathsizes);
	
	H5::DataSpace dspace_env = dset_env.getSpace();
	H5::DataSpace dspace_data = dset_data.getSpace();
	H5::DataSpace dspace_sizes = dset_sizes.getSpace();
	
	int rank = 1;
	
	hsize_t size_env, size_data, size_sizes;
     
    dspace_env.getSimpleExtentDims(&size_env, nullptr);
    dspace_data.getSimpleExtentDims(&size_data, nullptr);
    dspace_sizes.getSimpleExtentDims(&size_sizes, nullptr);
	
	auto type_int = H5toCpp<int>::type();
	auto type_db = H5toCpp<double>::type();
	
	std::vector<int> env(size_env);
	std::vector<double> data(size_data);
	std::vector<int> sizes(size_sizes);
		
	dset_env.read(env.data(), type_int);
	dset_data.read(data.data(), type_db);
	dset_sizes.read(sizes.data(), type_int);
	
	int nbas = env.size() / 2;
	std::vector<desc::Shell> basis(nbas);
	
	int off = 0;
	
	for (int ishell = 0; ishell != nbas; ++ishell) {
		auto& shell = basis[ishell];
		
		shell.l = env[ishell * 2 + 0];
		int nprim = env[ishell * 2 + 1];
		
		std::vector<double> alpha;
		std::vector<double> coeff;
		
		shell.O[0] = data[off++];
		shell.O[1] = data[off++];
		shell.O[2] = data[off++];
		
		alpha.insert(alpha.end(), data.begin() + off, data.begin() + off + nprim);
		off += nprim;
		coeff.insert(coeff.end(), data.begin() + off, data.begin() + off + nprim);
		off += nprim;
	
		shell.alpha = alpha;
		shell.coeff = coeff;
		
		std::cout << shell << std::endl;
	
	}
	
	/*std::cout << "NBAS: " << nbas << std::endl;
	std::cout << "SIZES: " << std::endl;
	for (auto s : sizes) {
		std::cout << s << " ";
	} std::cout << '\n';*/
	
	std::vector<std::vector<desc::Shell>> cbasis(sizes.size());
	int shell_off = 0;
	for (int iblk = 0; iblk != sizes.size(); ++iblk) {
		
		int nshells = sizes[iblk];
	
		cbasis[iblk].insert(cbasis[iblk].end(), basis.begin() + shell_off,
			basis.begin() + (shell_off + nshells));

		shell_off += nshells;
		
	}
		
	desc::cluster_basis cbasis_out(cbasis);
	
	/*for (auto& c : cbasis_out) {
		std::cout << "CLUSTER!" << std::endl;
		for (auto& s : c) {
			std::cout << s << std::endl;
		}
	}*/
	
	return cbasis_out;
	
}

void data_handler::write_molecule(desc::smolecule& mol) {
	
	if (m_world.rank() == 0)
	{
	
		auto group = new H5::Group(m_file_ptr->createGroup("/molecule"));
		
		// normal types
		
		auto name = mol->name();
		auto charge = mol->charge();
		auto mult = mol->mult();
		auto mo_split = mol->mo_split();
		
		write<std::string>("/molecule/name", &name, {1});
		write<int>("/molecule/charge", &charge, {1});
		write<int>("/molecule/mult", &mult, {1});
		write<int>("/molecule/mo_split", &mo_split, {1});
		
		// atoms
		H5::CompType atype(sizeof(desc::Atom));
		atype.insertMember("x", HOFFSET(desc::Atom, x), H5::PredType::NATIVE_DOUBLE);
		atype.insertMember("y", HOFFSET(desc::Atom, y), H5::PredType::NATIVE_DOUBLE);
		atype.insertMember("z", HOFFSET(desc::Atom, z), H5::PredType::NATIVE_DOUBLE);
		atype.insertMember("atomic_number", HOFFSET(desc::Atom, atomic_number), 
			H5::PredType::NATIVE_INT);
		
		auto atoms = mol->atoms();
		
		int atomrank = 1;
		hsize_t atomdim = atoms.size();
	
		H5::DataSpace atomspace(atomrank, &atomdim);
		H5::DataSet atomset(m_file_ptr->createDataSet("/molecule/atoms", atype, 
			atomspace));
		atomset.write(atoms.data(), atype);
		
		// basis
		auto cbas = mol->c_basis();
		
		auto group_basis = new H5::Group(m_file_ptr->createGroup("/molecule/basis"));
		write_basis(m_file_ptr, "/molecule/basis", *cbas);
		
		delete group_basis;
		delete group;
	
	} // end if rank
	
}

void data_handler::write_hf_wfn(hf::shared_hf_wfn& hfwfn) {
	
	H5::Group* group;
	
	auto c_bo_A = hfwfn->c_bo_A();
	auto c_bv_A = hfwfn->c_bv_B();
	auto c_bo_B = hfwfn->c_bo_A();
	auto c_bv_B = hfwfn->c_bv_B();
	
	if (m_world.rank() == 0) {
		group = new H5::Group(m_file_ptr->createGroup("/hartree_fock"));
	}
	
	write_matrix<double>("/hartree_fock/c_occ_A", c_bo_A);
	write_matrix<double>("/hartree_fock/c_occ_B", c_bo_B);
	write_matrix<double>("/hartree_fock/c_vir_A", c_bv_A);
	write_matrix<double>("/hartree_fock/c_vir_B", c_bv_B);
	
	if (m_world.rank() == 0) {
		
		double scf_energy = hfwfn->scf_energy();
		double nuc_energy = hfwfn->nuc_energy();
		double wfn_energy = hfwfn->wfn_energy();
				
		write<double>("/hartree_fock/scf_energy", &scf_energy, {1});
		write<double>("/hartree_fock/nuc_energy", &nuc_energy, {1});
		write<double>("/hartree_fock/wfn_energy", &wfn_energy, {1});
	
		#define write_vec(name) \
			auto name = hfwfn->name(); \
			write<double>("/hartree_fock/" #name, name->data(), \
				{name->size()}); \
				
		write_vec(eps_occ_A)
		write_vec(eps_vir_A)
		write_vec(eps_occ_B)
		write_vec(eps_vir_B)
		
	}
	
	delete group;
	
}
	

desc::smolecule data_handler::read_molecule() {
	
	auto name = read<std::string>("/molecule/name").data[0];
	int charge = read<int>("/molecule/charge").data[0];
	int mult = read<int>("/molecule/mult").data[0];
	int mo_split = read<int>("/molecule/mo_split").data[0];
	
	// atoms
	H5::CompType atype(sizeof(desc::Atom));
	atype.insertMember("x", HOFFSET(desc::Atom, x), H5::PredType::NATIVE_DOUBLE);
	atype.insertMember("y", HOFFSET(desc::Atom, y), H5::PredType::NATIVE_DOUBLE);
	atype.insertMember("z", HOFFSET(desc::Atom, z), H5::PredType::NATIVE_DOUBLE);
	atype.insertMember("atomic_number", HOFFSET(desc::Atom, atomic_number), 
		H5::PredType::NATIVE_INT);
	
	H5::DataSet atomset = m_file_ptr->openDataSet("/molecule/atoms");
	H5::DataSpace atomspace = atomset.getSpace();
	
    hsize_t natoms;
    atomspace.getSimpleExtentDims(&natoms, nullptr);
	
	std::vector<desc::Atom> atoms(natoms);
	atomset.read(atoms.data(), atype);
	
	//basis
	auto cbasis = std::make_shared<desc::cluster_basis>(
		read_basis(m_file_ptr, "/molecule/basis"));

	auto mol = desc::create_molecule()
		.comm(m_world.comm())
		.name(name)
		.atoms(atoms)
		.basis(cbasis)
		.charge(charge)
		.mult(mult)
		.mo_split(mo_split)
		.get();
		
	return mol;
	
}

hf::shared_hf_wfn data_handler::read_hf_wfn(desc::smolecule mol) {	
	
	double scf_energy = read<double>("/hartree_fock/scf_energy").data[0];
	double wfn_energy = read<double>("/hartree_fock/wfn_energy").data[0];
	double nuc_energy = read<double>("/hartree_fock/nuc_energy").data[0];
	
	svector<double> eps_occ_A = std::make_shared<std::vector<double>>(
		read<double>("/hartree_fock/eps_occ_A").data);
	svector<double> eps_occ_B = std::make_shared<std::vector<double>>(
		read<double>("/hartree_fock/eps_occ_B").data);
	svector<double> eps_vir_A = std::make_shared<std::vector<double>>(
		read<double>("/hartree_fock/eps_vir_A").data);
	svector<double> eps_vir_B = std::make_shared<std::vector<double>>(
		read<double>("/hartree_fock/eps_vir_B").data);
	
	auto b = mol->dims().b();
	auto oa = mol->dims().oa();
	auto ob = mol->dims().ob();
	auto va = mol->dims().va();
	auto vb = mol->dims().vb();
	
	auto coA = read_matrix<double>("/hartree_fock/c_occ_A", 
		"c_bo_A", b, oa, dbcsr::type::no_symmetry);
	auto coB = read_matrix<double>("/hartree_fock/c_occ_B", 
		"c_bo_B", b, ob, dbcsr::type::no_symmetry);
	auto cvA = read_matrix<double>("/hartree_fock/c_vir_A", 
		"c_bv_A", b, va, dbcsr::type::no_symmetry);
	auto cvB = read_matrix<double>("/hartree_fock/c_vir_B", 
		"c_bv_B", b, vb, dbcsr::type::no_symmetry);
	
	auto hfwfn = hf::create_hf_wfn()
		.c_bo_A(coA)
		.c_bv_A(cvA)
		.c_bo_B(coB)
		.c_bv_B(cvB)
		.mol(mol)
		.scf_energy(scf_energy)
		.nuc_energy(nuc_energy)
		.wfn_energy(wfn_energy)
		.eps_occ_A(eps_occ_A)
		.eps_occ_B(eps_occ_B)
		.eps_vir_A(eps_vir_A)
		.eps_vir_B(eps_vir_B)
		.get();
		
	return hfwfn;
	
}
	
} // end namespace
