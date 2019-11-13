#ifndef __CG_AUX_H__
#define __CG_AUX_H__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>


#define FP_SQRT 	sqrt
#define FP_RAND 	drand48
#define FP_SEED 	srand48
#define FP_ABS 		fabs	
#define FP_EXP 		frexp
#define FP_LOG10 	log10
#define FP_POW 		pow

#define FP_SCANSPEC	scan_dconspec


#include "mkl.h"
#define BLAS_cp(n, dx, incx, dy, incy) 					cblas_dcopy(n, dx, incx, dy, incy)
#define BLAS_dot(n, dx, incx, dy, incy) 				cblas_ddot(n, dx, incx, dy, incy) 
#define BLAS_axpy(n, da, dx, incx, dy, incy) 				cblas_daxpy(n, da, dx, incx, dy, incy)
#define BLAS_scal(n, da, dx, incx) 				        cblas_dscal(n, da, dx, incx)


void bblas_dcopy(int bm, int m, double *X, double *Y);
void bblas_ddot(int bm, int m, double *X, double *Y, double *result);
void bblas_daxpy(int bm, int m, double f, double *X, double *Y);
void bblas_dscal(int bm, int m, double f, double *X);
void VvecDoublesTasks(int bm, int m, double *src1, double *src2, double *dst);

//#pragma omp task depend(in:X[initx:initx+bm-1]) depend(out:Y[inity:inity+bm-1]) 
void __t_copy(int bm, int m, double *X, double *Y, int initx, int inity);

//#pragma omp task depend(in:X[initx:initx+bm-1], Y[inity:inity+bm-1]) out(result)//concurrent(result[0:bn-1]) 
void __t_dot(int bm, int m, double *X, double *Y, int initx, int inity, double *result);

//#pragma omp task depend(in:X[0:bm-1], f) out(Y[0:bm-1]) 
void __t_axpy(int bm, int m, double f, double *X, double *Y);

//#pragma omp task depend(inout:X[0:bm-1],f) 
void __t_scal(int bm, int m, double f, double *X);


#endif //__CG_AUX_H__
