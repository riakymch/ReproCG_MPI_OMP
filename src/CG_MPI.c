#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mkl_blas.h>
#include <mpi.h>
#include <hb_io.h>

#include "reloj.h"
#include "ScalarVectors.h"
#include "SparseProduct.h"
#include "ToolsMPI.h"
#include "matrix.h"
#include "common.h"

#include <cstddef>
#include <mpfr.h>

double dot_mpfr(int *N, double *a, int *inca, double *b, int *incb) {
    mpfr_t sum, dot, op1, op2;
    mpfr_init2(op1, 64);
    mpfr_init2(op2, 64);
    mpfr_init2(dot, 192);
    mpfr_init2(sum, 2048);

    mpfr_set_zero(sum, 0.0);

    for (int i = 0; i < *N; i++) {
        mpfr_set_d(op1, a[i], MPFR_RNDN);
        mpfr_set_d(op2, b[i], MPFR_RNDN);

        mpfr_set_zero(dot, 0.0);
        mpfr_mul(dot, op1, op2, MPFR_RNDN);

        mpfr_add(sum, sum, dot, MPFR_RNDN);
    }
    double dacc = mpfr_get_d(sum, MPFR_RNDN);

    mpfr_clear(op1);
    mpfr_clear(op2);
    mpfr_clear(dot);
    mpfr_clear(sum);
    mpfr_free_cache();

    return dacc;
}

// ================================================================================

#define DIRECT_ERROR 1
#define PRECOND 1
#define VECTOR_OUTPUT 0

void ConjugateGradient (SparseMatrix mat, double *x, double *b, int *sizes, int *dspls, int myId) {
    int size = mat.dim2, sizeR = mat.dim1; 
    int IONE = 1; 
    double DONE = 1.0, DMONE = -1.0, DZERO = 0.0;
    int n, n_dist, iter, maxiter, nProcs;
    double beta, tol, rho, alpha, umbral;
    double *res = NULL, *z = NULL, *d = NULL, *y = NULL;
    double *aux = NULL;
    double t1, t2, t3, t4;
#if PRECOND
    int i, *posd = NULL;
    double *diags = NULL;
#endif

    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    n = size; n_dist = sizeR; maxiter = size; umbral = 1.0e-8;
    CreateDoubles (&res, n_dist); CreateDoubles (&z, n_dist); 
    CreateDoubles (&d, n_dist);  
#ifdef DIRECT_ERROR
    // init exact solution
    double *res_err = NULL, *x_exact = NULL;
	CreateDoubles (&x_exact, n_dist);
	CreateDoubles (&res_err, n_dist);
    InitDoubles(x_exact, n_dist, DONE, DZERO);
#endif // DIRECT_ERROR 

#if PRECOND
    CreateDoubles (&y, n_dist);
    CreateInts (&posd, n_dist);
    CreateDoubles (&diags, n_dist);
    GetDiagonalSparseMatrix2 (mat, dspls[myId], diags, posd);
#pragma omp parallel for
    for (i=0; i<n_dist; i++) 
        diags[i] = DONE / diags[i];
#endif
    CreateDoubles (&aux, n); 

#if VECTOR_OUTPUT
    // write to file for testing purpose
    FILE *fp;
    if (myId == 0) {
        char name[50];
        sprintf(name, "%d.txt", nProcs);
        fp = fopen(name,"w");
    }
#endif

    iter = 0;
    MPI_Allgatherv (x, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (z, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, z);            			// z = A * x
    dcopy (&n_dist, b, &IONE, res, &IONE);                          		// res = b
    daxpy (&n_dist, &DMONE, z, &IONE, res, &IONE);                      // res -= z
#if PRECOND
    VvecDoubles (DONE, diags, res, DZERO, y, n_dist);                    // y = D^-1 * res
#else
    y = res;
#endif
    dcopy (&n_dist, y, &IONE, d, &IONE);                                // d = y

#if PRECOND
    beta = dot_mpfr (&n_dist, res, &IONE, y, &IONE);
    tol = dot_mpfr (&n_dist, res, &IONE, res, &IONE);

    tol = sqrt (tol);
#else
    beta = dot_mpfr (&n_dist, res, &IONE, res, &IONE);                        // tol = res' * res

    tol = sqrt (beta);
#endif

#ifdef DIRECT_ERROR
    // compute direct error
    double direct_err;
	dcopy (&n_dist, x_exact, &IONE, res_err, &IONE);                        // res_err = x_exact
	daxpy (&n_dist, &DMONE, x, &IONE, res_err, &IONE);                      // res_err -= x

    // compute inf norm
    direct_err = norm_inf(n_dist, res_err);

//    // compute euclidean norm
//    direct_err = dot_mpfr (&n_dist, res_err, &IONE, res_err, &IONE);       // direct_err = res_err' * res_err
//    direct_err = sqrt(direct_err);
#endif // DIRECT_ERROR

    MPI_Barrier(MPI_COMM_WORLD);
    if (myId == 0) 
        reloj (&t1, &t2);
    while ((iter < maxiter) && (tol > umbral)) {

        MPI_Allgatherv (d, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (z, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, z);            		// z = A * d

        if (myId == 0) 
#if DIRECT_ERROR
            printf ("%d \t %a \t %a \n", iter, tol, direct_err);
            //printf ("%d \t %20.10e \t %20.10e \n", iter, tol, direct_err);
#else        
            printf ("%d \t %a \n", iter, tol);
            //printf ("%d \t %20.10e \n", iter, tol);
#endif // DIRECT_ERROR

        rho = dot_mpfr (&n_dist, d, &IONE, z, &IONE);

        rho = beta / rho;
        daxpy (&n_dist, &rho, d, &IONE, x, &IONE);                      	// x += rho * d;
        rho = -rho;
        daxpy (&n_dist, &rho, z, &IONE, res, &IONE);                      // res -= rho * z

#if PRECOND
        VvecDoubles (DONE, diags, res, DZERO, y, n_dist);                 // y = D^-1 * res
#else
        y = res;
#endif
        alpha = beta;                                                 		// alpha = beta

#if PRECOND
        beta = dot_mpfr (&n_dist, res, &IONE, y, &IONE);
        tol = dot_mpfr (&n_dist, res, &IONE, res, &IONE);

        tol = sqrt (tol);
#else
        beta = dot_mpfr (&n_dist, res, &IONE, y, &IONE);                      // beta = res' * y                     
        
        tol = sqrt (beta);
#endif

#ifdef DIRECT_ERROR
        // compute direct error
        dcopy (&n_dist, x_exact, &IONE, res_err, &IONE);                        // res_err = x_exact
        daxpy (&n_dist, &DMONE, x, &IONE, res_err, &IONE);                      // res_err -= x

        // compute inf norm
        direct_err = norm_inf(n_dist, res_err);

//        // compute euclidean norm
//        direct_err = dot_mpfr (&n_dist, res_err, &IONE, res_err, &IONE);
//        direct_err = sqrt(direct_err);
#endif // DIRECT_ERROR

        alpha = beta / alpha;                                                   // alpha = beta / alpha
        dscal (&n_dist, &alpha, d, &IONE);                                // d = alpha * d
        daxpy (&n_dist, &DONE, y, &IONE, d, &IONE);                       // d += y

        iter++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myId == 0)
        reloj (&t3, &t4);

#if VECTOR_OUTPUT
    // print aux
    MPI_Allgatherv (x, n_dist, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    if (myId == 0) {
        fprintf(fp, "%d ", iter);
        for (int ip = 0; ip < n; ip++)
            fprintf(fp, "%20.10e ", aux[ip]);
        fprintf(fp, "\n");
    }
#endif

    if (myId == 0) {
        printf ("Size: %d \n", n);
        printf ("Iter: %d \n", iter);
        //printf ("Tol: %a \n", tol);
        printf ("Tol: %20.10e \n", tol);
        printf ("Time_loop: %20.10e\n", (t3-t1));
        printf ("Time_iter: %20.10e\n", (t3-t1)/iter);
    }

    RemoveDoubles (&aux); RemoveDoubles (&res); RemoveDoubles (&z); RemoveDoubles (&d);
#if PRECOND
    RemoveDoubles (&diags); RemoveInts (&posd); RemoveDoubles(&y);
#endif
}

/*********************************************************************************/

int main (int argc, char **argv) {
    int dim; 
    double *vec = NULL, *sol1 = NULL, *sol2 = NULL;
    int index = 0, indexL = 0;
    SparseMatrix mat  = {0, 0, NULL, NULL, NULL}, sym = {0, 0, NULL, NULL, NULL};

    int root = 0, myId, nProcs;
    int dimL, dspL, *vdimL = NULL, *vdspL = NULL;
    SparseMatrix matL = {0, 0, NULL, NULL, NULL};
    double *vecL = NULL, *sol1L = NULL, *sol2L = NULL;
    int mat_from_file, nodes, size_param, stencil_points;

    if (argc == 3) {
        mat_from_file = atoi(argv[2]);
    } else {
        mat_from_file = atoi(argv[2]);
        nodes = atoi(argv[3]);
        size_param = atoi(argv[4]);
        stencil_points = atoi(argv[5]);
    }

    /***************************************/

    MPI_Init (&argc, &argv);

    // Definition of the variables nProcs and myId
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    root = nProcs-1;
    root = 0;

    /***************************************/

    CreateInts (&vdimL, nProcs); CreateInts (&vdspL, nProcs); 
    if(mat_from_file) {
        if (myId == root) {
            // Creating the matrix
            ReadMatrixHB (argv[1], &sym);
            DesymmetrizeSparseMatrices (sym, 0, &mat, 0);
            dim = mat.dim1;
        }

        // Distributing the matrix
        dim = DistributeMatrix (mat, index, &matL, indexL, vdimL, vdspL, root, MPI_COMM_WORLD);
        dimL = vdimL[myId]; dspL = vdspL[myId];
    }
    else {
        dim = size_param * size_param * size_param;
        int divL, rstL, i;
        divL = (dim / nProcs); rstL = (dim % nProcs);
        for (i=0; i<nProcs; i++) vdimL[i] = divL + (i < rstL);
        vdspL[0] = 0; for (i=1; i<nProcs; i++) vdspL[i] = vdspL[i-1] + vdimL[i-1];
        dimL = vdimL[myId]; dspL = vdspL[myId];
        int band_width = size_param * (size_param + 1) + 1;
        band_width = 100 * nodes;
        long nnz_here = ((long) (stencil_points + 2 * band_width)) * dimL;
        printf ("dimL: %d, nodes: %d, size_param: %d, band_width: %d, stencil_points: %d, nnz_here: %ld\n",
                dimL, nodes, size_param, band_width, stencil_points, nnz_here);
        allocate_matrix(dimL, dim, nnz_here, &matL);
        generate_Poisson3D_filled(&matL, size_param, stencil_points, band_width, dspL, dimL, dim);

//        // To generate ill-conditioned matrices
//        double factor = 1.0e6;
//        ScaleFirstRowCol(matL, dspL, dimL, myId, root, factor);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Creating the vectors
    if (myId == root) {
        CreateDoubles (&vec , dim);
        CreateDoubles (&sol1, dim);
        CreateDoubles (&sol2, dim);
        InitRandDoubles (vec, dim, -1.0, 1.0);
        InitDoubles (sol1, dim, 0.0, 0.0);
        InitDoubles (sol2, dim, 0.0, 0.0);
    } else {
        CreateDoubles (&vec , dim);
        CreateDoubles (&sol2, dim);
        InitDoubles (vec , dim, 0.0, 0.0);
        InitDoubles (sol2, dim, 0.0, 0.0);
    }
    CreateDoubles (&vecL , dimL);
    CreateDoubles (&sol1L, dimL);
    CreateDoubles (&sol2L, dimL);
    InitDoubles (vecL , dimL, 0.0, 0.0);
    InitDoubles (sol1L, dimL, 0.0, 0.0);
    InitDoubles (sol2L, dimL, 0.0, 0.0);

    /***************************************/

    int i, IONE = 1;
    double beta;
    if (myId == root) {
        InitDoubles (vec, dim, 1.0, 0.0);
        InitDoubles (sol1, dim, 0.0, 0.0);
        InitDoubles (sol2, dim, 0.0, 0.0);
    }
    int k=0;
    int *vptrM = matL.vptr;
    for (int i=0; i < matL.dim1; i++) {
        for(int j=vptrM[i]; j<vptrM[i+1]; j++) {
            sol1L[k] += matL.vval[j];
        }
        k++;
    }

    MPI_Scatterv (sol2, vdimL, vdspL, MPI_DOUBLE, sol2L, dimL, MPI_DOUBLE, root, MPI_COMM_WORLD);

    ConjugateGradient (matL, sol2L, sol1L, vdimL, vdspL, myId);

    // Error computation
    for (i=0; i<dimL; i++) sol2L[i] -= 1.0;

    beta = dot_mpfr (&dimL, sol2L, &IONE, sol2L, &IONE);            
    beta = sqrt(beta);
    if (myId == 0) 
        printf ("Error: %a\n", beta);
        //printf ("Error: %10.5e\n", beta);

    /***************************************/
    // Freeing memory
    RemoveDoubles (&sol2L); RemoveDoubles (&sol1L); RemoveDoubles (&vecL);
    RemoveInts (&vdspL); RemoveInts (&vdimL); 
    if (myId == root) {
        RemoveDoubles (&sol2); RemoveDoubles (&sol1); RemoveDoubles (&vec);
        RemoveSparseMatrix (&mat); RemoveSparseMatrix (&sym);
    } else {
        RemoveDoubles (&sol2); RemoveDoubles (&vec);
    }

    MPI_Finalize ();

    return 0;
}

