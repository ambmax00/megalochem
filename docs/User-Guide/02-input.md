# Running MEGALOChem

## Parallel Runtime Environment

After installing MEGALOChem, the executable `bin/chem` is found in the build directory. The program is launched with `mpirun`, `mpiexec` or `srun`, and takes two arguments
````
mpirun -n <NP> bin/chem <INPUT> <DIRECTORY>
````
MEGALOchem then reads from the file `<INPUT>.json` and optionally from `<INPUT>.hdf5` located in `<DIRECTORY>`. The JSON file contains information on the job control, while the HDF5 file contains information from previous calculations. If the HDF5 file is present, MEGALOchem moves `<INPUT>.hdf5` to `<INPUT>.back.hdf5`. Information from the current run is then saved to the new HDF5 file.

MEGALOchem is currently optimized to 1 OpenMP-thread per 1 MPI process. You should also use a square number of MPI processes for optimal performance, even if that means some processes may run idly. Please also consult the [CP2K documentation](https://xconfigure.readthedocs.io/en/latest/cp2k/#running-cp2k), which also uses DBCSR, for further information

## Input Files

Job control is specified by an input file in the JSON format. It is inspired by the [BAGLE](https://github.com/qsimulate-open/bagel) quantum chemistry software and follows a similar scheme to the object-oriented python interfaces like Psi4 or PySCF. Consider the following example for running a Hartree-Fock calculation on a water molecule with the cc-pVDZ basis set
````
{
  "megalochem": [
  {
    "type": "atoms",
    "tag": "xyz",
    "unit": "angstrom",
    "geometry": [
      0.00000,        0.00000,        0.11779,
      0.00000,        0.75545,       -0.47116,
      0.00000,       -0.75545,       -0.47116
    ],
    "symbols": ["O", "H", "H"]
  },
  {
    "type": "basis",
    "atoms": "xyz",
    "tag": "basis1",
    "name": "cc-pvdz"
  },
  {
    "type": "molecule",
    "tag": "mol",
    "atoms": "xyz",
    "basis": "basis1",
    "mult": 1,
    "charge": 0
  },
  {
    "type": "hfwfn",
    "tag": "hfwfn",
    "molecule": "mol",
    "guess": "SAD",
    "build_J": "exact",
    "build_K": "exact"
   }]
}
````
The JSON file contains a main structure `megalochem` which specifies an ordered array of objects. Each object must contain the fields `type` and  `tag`. The entry `type` specifies what kind of object it is, and `tag` is a user-defined name to uniquely define that object. This is similar to declaring a variable in a programming language. The available types are

1. `global`: Specifies global variables like block thresholds or damping factors for electron integrals

2. `atoms`: Specifies the atomic structure of the system. It can either contain an array with coordinates or reference an XYZ file

3. `basis`: Specifies the basis set and what kind of clustering is used. Clustering indicates how atomic orbitals are grouped into blocks. A cutoff parameter can also be specified for removing linear dependencies

4. `molecule`: Specifies molecular information like multiplicity and charge

5. `hfwfn`: Job control for computing the HF wave function

6. `mpwfn`: Job control for computing the MP2 wave function

7. `adcwfn`: Job control for computing CIS and ADC(2) excitation energies

8. `moprint` Job control for localizing and plotting orbitals (CMOs, LMOs, NTOs) from different calculations

## List of Keywords

The `values` cell either contains the default value or all possible values for that keyword with the default in first position.

Also have a look at [this file](../../src/megalochem_driver.cpp) where all the keywords are defined.

### global
keyword | type | values | explanation
--- | --- | --- | ---
`block_threshold` | double | 1e-8 | Blocks with Frobenius norm below this threshold are ignored in sparse matrix and tensor operations
`integral_omega` | double | 0.1 | Damping factor for electron integrals
`qr_T` | double | 1e-5 | Threshold for choosing fitting functions in the quasi-robust density-fitting procedure
`qr_R` | double | 1e-5 | Threshold for choosing test functions in the quasi-robust density-fitting procedure
`basis_lindep` | double | 1e-6 | Threshold for canonical orthogonalization

### atoms

keyword | type | values | explanation
--- | --- | --- | ---
`unit` | "string" | "angstrom" | Indicates the unit for the xyz coordinates.
`geometry` | double-array | - | Linear array containing the atomic xyz-coordinates
`symbols` | string-array | - | Array containing the atomic symbols
`file`| string | - | Coordinates are read from the xyz file indicated by this keyword
`reorder` | bool | true/false | Whether atoms are reordered using the reverse Cuthill-McKee algorithm

### basis
keyword | type | values | explanation
--- | --- | --- | ---
`atoms` | "string | - | Name of the atoms object for which to construct the basis set
`name` | string | - | Name of the basis set (see basis folder for available basis sets)
`augmentation` | bool | false/true | Whether to use augmented version of the basis set
`cutoff` | double | 1e-6 | Cutoff for removing linear dependencies
`ao_split_method` | string | "atomic", "shell", "multi_shell", "multi_shell_strict", "multi_shell_strict_sp" | Inicates the clustering method for the basis set.
`ao_split` | integer | 8 | Maximum number of basis functions within one cluster
`symbols` | string-array | ["H", "C", ...] | Array which contains the unique atom types
`names` | string-array | ["cc-pvdz", ...] | Alternative to `name` where the `i`th entry indicates the basis set that is put on the atom type in `i`th position of the `symbols` array
`augmentations` | bool-array | [false, ...] | Alternative to `augmentation` where the `i`th entry indicates whether an augmented basis set is used for the atom type in `i`th position of the `symbols` array

### molecule

keyword | type | values | explanation
--- | --- | --- | ---
`atoms` | "string" | - | Name of the `atoms`-object for which to construct the molecule
`basis` | "string" | - | Name of the `basis`-object for this molecule
`basis2` | "string" | - | Name of the secondary `basis`-object for this molecule
`mult` | integer | - | Multiplicity of this molecule
`charge` | integer | - | Charge of this molecule
`mo_split` | integer | 8 | Split factor for the occupied and virtual MO dimensions of matrices and tensors

### hfwfn

keyword | type | values | explanation
--- | --- | --- | ---
`molecule` | string | - | Name of the `molecule` object
`df_basis` | string | - | Name of the `basis` object for density fitting
`df_basis2` | string | - | Name of the `basis` object for density fitting when using the `project` guess
`df_metric` | string | "coulomb", "erfc_coulomb", "qr_fit", "pari" | Density fitting method
`diis` | bool | true/false | Use DIIS acceleration
`diis_max_vecs` | integer | 8 | Maximum number of vectors in the DIIS space
`diis_min_vecs` | integer | 1 | Minimum number of vectors needed before DIIS starts
`diis_start` | integer | 0 | Iteration where DIIS starts to collect error vectors
`diis_beta` | bool | true/false | Do separate DIIS for beta space
`guess` | string | "SAD", "core", "project" | Hartree-Fock guess. For `project`, a secondary basis in the molecule is needed
`scf_thresh` | double | 1e-6 | SCF threshold
`unrestricted` | bool | false/true | Force unrestricted calculation even for closed-shell molecules
`build_J` | string | "exact", "dfao" | Method to construct Coulomb matrix
`build_K` | string | "exact", "dfao", "dfmo", "dfmem", "robust" | Method to construct exchange matrix
`eris` | string | "core", "disk", "direct" | How electron integrals are stored in memory
`intermeds` | string | "core", "disk" | How intermediates are stored in memory
`print` | integer | 0 | Print level
`nbatches_b` | integer | 5 | Number of batches along the AO dimension
`nbatches_x` | integer | 5 | Number of batches along the auxiliary AO dimension
`occ_nbatches` | integer | 2 | Number of batches along the MO dimension (important for dfmo)
`read` | bool | false,true | Read the Hartree-Fock wave function from the HDF5 file. This skips the HF calculation.
`max_iter` | integer | 100 | Maximum number of SCF iterations
`SAD_diis` | bool | false/true | Whether to do DIIS acceleration during atomistic HF calculations
`SAD_spin_average` | bool | true/false | Use spin-averaged occupation for atomistic HF calculations

### mp_wfn

keyword | type | values | explanation
--- | --- | --- | ---
`wfn` | string | - | Name of the object which is used as the reference
`df_basis` | string | - | Name of the `basis` object for density fitting
`print` | integer | 0 | Print level
`nbatches_b` | integer | 5 | Number of batches along the AO dimension
`nbatches_x` | integer | 5 | Number of batches along the auxiliary AO dimension
`nlap` | integer | 5 | number of Laplace points
`c_os` | double | 1.3 | SOS coefficient
`eris` | string | "core", "disk", "direct" | How electron integrals are stored in memory
`intermeds` | string | "core", "disk" | How intermediates are stored in memory
`df_metric` | string | "coulomb", "erfc_coulomb", "qr_fit", "pari" | Density fitting method
`build_Z` | string | "llmp_full", "llmp_mem" | Method to construct Z matrix in SOS-MP2

### adc_wfn

keyword | type | values | explanation
--- | --- | --- | ---
`wfn` | string | - | Name of the object which is used as the reference
`df_basis` | string | - | Name of the `basis` object for density fitting
`print` | integer | 0 | Print level
`method` | string | "ri_ao_adc1", "sos_cd_ri_adc2" | ADC method
`nbatches_b` | integer | 5 | Number of batches along the AO dimension
`nbatches_x` | integer | 5 | Number of batches along the auxiliary AO dimension
`nlap` | integer | 5 | Number of Laplace points
`nroots` | integer | 1 | Number of root(s) to compute
`c_os` | double | 1.3 | SOS coefficient
`c_os_coupling` | double | 1.17 | SOS coupling coefficient
`eris` | string | "core", "disk", "direct" | How electron integrals are stored in memory
`imeds` | string | "core", "disk" | How intermediates are stored in memory
`df_metric` | string | "coulomb", "erfc_coulomb", "qr_fit", "pari" | Density fitting method
`read` | bool | false/true | Read the transition matrices and excitation energies from the HDF5 file. This skips the ADC calculation.
`block` | bool | false/true | Use Davdison-Liu to compute all roots from 1 to `nroots`
`balanced` | bool | false/true | Use balanced Davdison algorithm
`conv` | double | 1e-5 | Convergence threshold
`nguesses` | integer | Number of guesses to use for Davidson
`guess` | string | "hf", "adc" | Type of guess
`build_J` | string | "dfao" | Method to construct Coulomb-like matrix
`build_K` | string | "dfao", dfmem", "robust" | Method to construct exchange-like matrix
`build_Z` | string | "llmp_full", "llmp_mem" | Method to construct Z-like matrix in SOS-ADC(2)
`dav_max_iter` | integer | 50 | Maximum number of Davdison iterations
`diis_max_iter` | integer | 50 | Maximum number of DIIS iterations
`use_doubles_ob` | bool | false/true | Use "OB"-type algorithm for computing doubles-contributions
`test_mvp` | bool | false/true | Run a single Davdison iteration, then stop and output timings. Good for benchmarking

### moprint

keyword | type | values | explanation
--- | --- | --- | ---
`wfn` | string | - | Name of the object which is used as the reference
`job_type` | string | "cmo", "local", "nto" | What orbitals will be computed/printed
`lmo_occ` | string | "boys", "pao", "cholesky" | Localization method for occupied MOs
`lmo_vir` | string | "boys", "pao", "cholesky" | Localization method for virtual MOs
`file` | string | - | Results are saved in a file with the name \<job_type\>_\<file\>.molden
