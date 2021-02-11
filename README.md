# MEGALOCHEM

WORK IN PROGRESS!
A quantum chemistry software package that is being designed for computing excitation energies of large molecules using ADC, using sparse matrix algebra. 

## Getting Started

Coming soon...

### Prerequisites

Currently depends on:

```
libcint >= 4.0.7 
ScaLAPACK (>= 2.0 for reference ScaLAPACK)
HDF5 (with parallel libraries enabled)
DBCSR (fork from ambmax00)
fypp (Fortran preprocessor)
python >= 3.6
```

### Installing

In the build directory:

```
export PKG_CONFIG_PATH=/path/to/libxsmm/lib
pkg-config libxsmm --libs

CMAKE_PREFIX_PATH=/my/dbcsr/install/location/usr/local/lib/cmake
cmake 
    -DLIBCINT_DIR=(mylibcint_dir) 
    -DSCALAPACK_DIR=(myscalapack_dir)  
    -DCMAKE_CXX_COMPILER=(g++/icpc ...)
    -DMPI_CXX_COMPILER=(mpic++,...)
    -DMPI_VENDOR=(openmpi/intelmpi ...)
    -DCMAKE_Fortran_COMPILER=(gfortran/ifort)
    -DEigen3_DIR=(Eigen3_dir) 
    ..
```

Compilers need MPI/OpenMP support. C++ needs to have at least C++17 standard.

## Implemented

* Hartree Fock (exact/df/df-mem/pari/lmo)
* MP2 (SOS-AOMP2)
* ADC (AO-ADC1, SOS-AO-ADC2)

## License

Coming soon....

## Other

* json.hpp from [here](https://github.com/nlohmann/json)
* documentation is going to follow
