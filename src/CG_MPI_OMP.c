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
#include "cg_aux.h"
#include "matrix.h"
#include "common.h"

// ================================================================================

#define DIRECT_ERROR 1
#define PRECOND 1
#define VECTOR_OUTPUT 0

void ConjugateGradient (SparseMatrix mat, double *x, double *b, int *sizes, int *dspls, int myId, int bm) {
	int size = mat.dim2, sizeR = mat.dim1; 
	int IONE = 1; 
	double DONE = 1.0, DMONE = -1.0;
	int n, n_dist, iter, maxiter;
	double beta, tol, rho, alpha, umbral;
	double *res = NULL, *z = NULL, *d = NULL, *y = NULL;
	double *aux = NULL;
	double t1, t2, t3, t4;
#ifdef PRECOND
    int i, *posd = NULL;
    double *diags = NULL;
#endif

	n = size; n_dist = sizeR; maxiter = size; umbral = 1.0e-8;
  maxiter = 500;
	CreateDoubles (&res, n_dist); CreateDoubles (&z, n_dist); 
	CreateDoubles (&d, n_dist);  

#ifdef DIRECT_ERROR
    // init exact solution
    double *res_err = NULL, *x_exact = NULL;
	CreateDoubles (&x_exact, n_dist);
	CreateDoubles (&res_err, n_dist);
    InitDoubles(x_exact, n_dist, DONE, 0);
#endif // DIRECT_ERROR 

#ifdef PRECOND
    CreateDoubles (&y, n_dist);
    CreateInts (&posd, n_dist);
    CreateDoubles (&diags, n_dist);
    GetDiagonalSparseMatrix2 (mat, dspls[myId], diags, posd);
    for (i=0; i<n_dist; i++)
        diags[i] = DONE / diags[i];
#endif

	CreateDoubles (&aux, n); 

	iter = 0;
	MPI_Allgatherv (x, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
	for ( int ii=0; ii<n_dist; ii+=bm) {
        int cs = n_dist - ii;
        int c = cs < bm ? cs : bm;
        double *zptr = &z[ii];
        #pragma omp task depend(out:z[ii:ii+c-1]) firstprivate(ii, c) 
        InitDoubles (zptr, c, 0.0, 0.0);
    }
	ProdSparseMatrixVectorByRows_OMPTasks (mat, 0, aux, z, bm);         // z = A * x
    bblas_dcopy(bm, n_dist, b, res);																		// res = b
	bblas_daxpy(bm, n_dist, DMONE, z, res);                             // res -= z
#ifdef PRECOND
    VvecDoublesTasks(bm, n_dist, diags, res, y); 											// y = D^-1 * res
#else
	y = res;																														// y = res
#endif
    bblas_dcopy(bm, n_dist, y, d);                                      // d = y

    double reduce[2];
#ifdef PRECOND
    bblas_ddot(bm, n_dist, res, y, &beta);                              // beta = res' * y
	tol = ddot (&n_dist, res, &IONE, res, &IONE);                      // beta = res' * res                     
    #pragma omp taskwait

    reduce[0] = beta;
    reduce[1] = tol;

	MPI_Allreduce (MPI_IN_PLACE, reduce, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
    beta = reduce[0];
    tol = reduce[1];

	tol = sqrt (tol);                              									// tol = norm (res)
#else
    bblas_ddot(bm, n_dist, res, y, &beta);                              // beta = res' * y
    #pragma omp taskwait
	MPI_Allreduce (MPI_IN_PLACE, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	tol = sqrt (beta);
#endif

#ifdef DIRECT_ERROR
    // compute direct error
    double direct_err;
    bblas_dcopy(bm, n_dist, x_exact, res_err);								// res_err = x_exact
	bblas_daxpy(bm, n_dist, DMONE, x, res_err);                             // res_err -= x;

    // compute inf norm
    direct_err = norm_inf(n_dist, res_err);
    MPI_Allreduce(MPI_IN_PLACE, &direct_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif // DIRECT_ERROR

    MPI_Barrier(MPI_COMM_WORLD);
	if (myId == 0)
        reloj (&t1, &t2);

	while ((iter < maxiter) && (tol > umbral)) {
		MPI_Allgatherv (d, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
		for ( int ii=0; ii<n_dist; ii+=bm) {
    	    int cs = n_dist - ii;
    	    int c = cs < bm ? cs : bm;
    	    double *zptr = &z[ii];
    	    #pragma omp task depend(out:z[ii:ii+c-1]) firstprivate(ii, c)
    	    InitDoubles (zptr, c, 0.0, 0.0);
  	    }
		ProdSparseMatrixVectorByRows_OMPTasks (mat, 0, aux, z, bm);            		// z = A * d

		if (myId == 0) 
#ifdef DIRECT_ERROR
            printf ("%d \t %a \t %a \n", iter, tol, direct_err);
#else        
            printf ("%d \t %20.10e \n", iter, tol);
#endif // DIRECT_ERROR

        bblas_ddot(bm, n_dist, d, z, &rho);
  	    #pragma omp taskwait

		MPI_Allreduce (MPI_IN_PLACE, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		rho = beta / rho;
		bblas_daxpy(bm, n_dist, rho, d, x);                                       // x += rho * d;
		rho = -rho;
		bblas_daxpy(bm, n_dist, rho, z, res);                                     // res -= rho * z
#ifdef PRECOND
        VvecDoublesTasks(bm, n_dist, diags, res, y);                               // y = D^-1 * res
#else
		y = res;
#endif
		alpha = beta;                                                 		        // alpha = beta

#ifdef PRECOND
        bblas_ddot(bm, n_dist, res, y, &beta);                                    // beta = res' * y  
		tol = ddot (&n_dist, res, &IONE, res, &IONE);                      // beta = res' * res                     

        reduce[0] = beta;
        reduce[1] = tol;

  	    #pragma omp taskwait
		MPI_Allreduce (MPI_IN_PLACE, reduce, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      
        beta = reduce[0];
        tol = reduce[1];

		tol = sqrt (tol);                              									// tol = norm (res)
#else
		tol = sqrt (beta);                              									// tol = norm (res)
#endif

#ifdef DIRECT_ERROR
        // compute direct error
        double direct_err;
        bblas_dcopy(bm, n_dist, x_exact, res_err);								// res_err = x_exact
        bblas_daxpy(bm, n_dist, DMONE, x, res_err);                             // res_err -= x;

        // compute inf norm
        direct_err = norm_inf(n_dist, res_err);
        MPI_Allreduce(MPI_IN_PLACE, &direct_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif // DIRECT_ERROR

		alpha = beta / alpha;                                         		        // alpha = beta / alpha
		bblas_dscal(bm, n_dist, alpha, d);                                        // d = alpha * d
		bblas_daxpy(bm, n_dist, DONE, y, d);                                      // d += y
  	    #pragma omp taskwait 

		iter++;
	}
    #pragma omp taskwait
	MPI_Barrier(MPI_COMM_WORLD);

	if (myId == 0)
        reloj (&t3, &t4);

	if (myId == 0) {
		printf ("Size: %d \n", n);
		printf ("Iter: %d \n", iter);
		printf ("Tol: %20.10e \n", tol);
		printf ("Time_loop: %20.10e\n", (t3-t1)); 
		printf ("Time_iter: %20.10e\n", (t3-t1)/iter);
    }

	RemoveDoubles (&aux); RemoveDoubles (&res); RemoveDoubles (&z); RemoveDoubles (&d);
#ifdef PRECOND
	RemoveDoubles(&y);
    RemoveDoubles (&diags); RemoveInts (&posd); 
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
	int bm = atoi(argv[1]);

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

    #pragma omp parallel 
    {
  	    #pragma omp single
  	    {
	        ConjugateGradient (matL, sol2L, sol1L, vdimL, vdspL, myId, bm);
        }
    }

	// Error computation
	for (i=0; i<dimL; i++)
        sol2L[i] -= 1.0;

    // TODO: why not with OMP tasks?
	beta = ddot (&dimL, sol2L, &IONE, sol2L, &IONE);
	MPI_Allreduce (MPI_IN_PLACE, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	beta = sqrt(beta);
	if (myId == 0)
        printf ("Error: %a\n", beta);
		
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

