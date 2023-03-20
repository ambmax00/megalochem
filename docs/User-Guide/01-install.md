# Installation Guide

## Perequisites

MEGALOchem currently depends on:
1. BLAS & LAPACK
2. ScaLAPACK
3. HDF5 (MPI enabled)
4. DBCSR, configured with LIBXSMM
5. libcint (version 4.4.5 or higher), available [here](https://github.com/sunqm/libcint). Configure with CMake flags `WITH_RANGE_COULOMB=ON` and `WITH_COULOMB_ERF=ON`
6. Eigen matrix library
7. (Optional) Clang-format

Furthermore you will need
1. CMake (version 3.17 or higher)
2. Fortran compiler which supports the 2008 standard including the TS 29113 for C-bindings
3. C++ compiler with c++17 standard (GCC >9.2.0 or Intel >19.1) and MPI support (version >4.0)

## Installing

The executable is built using cmake. Inside the megalochem directory, run
```
mkdir build
cd build
cmake ..
````
Prior to running cmake, remember to link the LIBXSMM library:
````
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:PATH/TO/LIBXSMMM
pkg-config libxsmm libxsmmext --libs
````
Additional configuration flags for cmake are:
````
-DDBCSR_DIR = <location of DBCSR config file>
-DLIBCINT_DIR = <location of libcint>
-DEigen3_DIR = <location of Eigen3 headers>
-DSCALAPACK_DIR = <location of ScaLAPACK>
-DSCALAPACK_IMPLICIT = <false|true>
-DMPI_VENDOR = <openmpi|intelmpi|...>
-DHDF5_DIR = <location of hdf5 config file>
-DCLANG_FORMAT_EXE = <clang-format executable>
````

When building with compiler wrappers that implicitly link to ScaLAPACK, pass `-DSCALAPACK_IMPLICIT=true` as a flag.
