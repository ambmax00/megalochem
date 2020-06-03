# MEGALOCHEM

A quantum chemistry software package designed for computing excitation energies of large molecules using ADC, using sparse matrix algebra. 

## Getting Started

Coming soon...

### Prerequisites

Currently depends on:

```
libint2
ScaLAPACK
boost 
TBB
DBCSR (fork)
fypp (Fortran preprocessor)
```
Exact versions are going to follow

### Installing

In the build directory:

```
CMAKE_PREFIX_PATH=/my/dbcsr/install/location/usr/local/lib/cmake
cmake 
    -DLibint2_DIR=(mylibint2_dir) 
    -DSCALAPACK_DIR=(myscalapack_dir)  
    -DCMAKE_C_COMPILER=(gcc/icc ...) 
    -DCMAKE_CXX_COMPILER=(g++/icpc ...)
    -DCMAKE_Fortran_COMPILER=(gfortran/ifort) 
    ..
```

Cmake recipie will be improved.

Compilers need MPI/OpenMP support. C++ needs to have at least C++17 standard.

## License

Coming soon....

## Other

* json.hpp from [here](https://github.com/nlohmann/json)
* documentation is going to follow
