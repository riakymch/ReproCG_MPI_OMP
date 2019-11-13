# ReproCG

## Introduction

ReproCG aims to ensure reproducibility and accuracy of the pure MPI implementation of the preconditioned Conjugate Gradient method. The project is composed of the following branches:
- orig containing the original version of the code

- master with the reproducible and accurate implementation using the ExBLAS approach

- opt_exblas optimizes the previous version by relying only on floating-point expansions (FPE, short arrays of doubles) with error-free transformations (`twosum' and `twoprod'). This version employs FPEs of size 8 with the early-exit technique

- opt_exblas_fpe3 works with FPEs of size 3. This version throws warning in case reproducibility may not be ensured

- mpfr provides highly accurate sequential implementation using the MPFR library. It serves as a reference

Currently, we also consider to apply vectorization to the opt_exblas using the VCL library

## Installation

#### Requirements:
- `c++ 11` (presently with `gcc-5.3.0` or higher)

- support of fma, especially for the opt_exblas branch

- separate installation of the MPFR library

#### Building ReproCG

1. clone the git-repository into `<ReproCG_root>`

2. inside the src directory, invoke `make` to create CG_MPI_OMP executable

## Example
The code can be run using two modes
- automatically generated matrix arising from the finite-difference method of a 3D Poisson’s equation with 27 stencil points. This matrix has the number of rows/columns of the matrix equal to 159^3, which is roughly 4M, and has the bandwidth of 200.

`mpirun -np P --bind-to core --report-bindings ./ReproCG/src/CG_MPI_OMP ../Matrices/$mat 0 2 159 27`

- matrix from the Suite Sparse Matrix Collection

`mpirun -np P --bind-to core --report-bindings ./ReproCG/src/CG_MPI_OMP MAT.rsa 1`
 
