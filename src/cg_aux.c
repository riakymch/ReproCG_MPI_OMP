
#include "exblas/exdot.hpp"

#include "cg_aux.h"

#define NBFPE 8

//static inline void __attribute__((always_inline)) bblas_dcopy(int bm, int m, double *X, double *Y) 
void bblas_dcopy(int bm, int m, double *X, double *Y) 
{
	int i;
	for ( i=0; i<m; i+=bm ) {
		int cs = m - i;
		int c = cs < bm ? cs : bm;
		#pragma omp task depend(in:X[i:i+c-1]) depend(out:Y[i:i+c-1]) firstprivate(i,c,m)
		  __t_copy(c, m, X, Y, i, i);
	}
}


//static inline void __attribute__((always_inline)) bblas_ddot(int bm, int m, double *X, double *Y, double *result) 
void bblas_ddot(int bm, int m, double *X, double *Y, double *result) 
{
	*result = 0; //M
	int i;
	for ( i=0; i<m; i+=bm ) {
		int cs = m - i;
		int c = cs < bm ? cs : bm;
		#pragma omp task depend(in:X[i:i+c-1], Y[i:i+c-1]) depend(out:result) firstprivate(i,c,m) //private(result)
		  __t_dot(c, m, X, Y, i, i, result);
	}
}


//static inline void __attribute__((always_inline)) bblas_daxpy(int bm, int m, double f, double *X, double *Y) 
void bblas_daxpy(int bm, int m, double f, double *X, double *Y) 
{
	int i;
	for ( i=0; i<m; i+=bm ) {
		int cs = m - i;
		int c = cs < bm ? cs : bm;
		#pragma omp task depend(in:X[i:i+c-1], f) depend(inout:Y[i:i+c-1]) firstprivate(i, c, m, f) //depend(in:f)
		  __t_axpy(c, m, f, &X[i], &Y[i]);
	}
}

//static inline void __attribute__((always_inline)) bblas_dscal(int bm, int m, double f, double *X) 
void bblas_dscal(int bm, int m, double f, double *X) 
{
	int i;
	for ( i=0; i<m; i+=bm ) {
		int cs = m - i;
		int c = cs < bm ? cs : bm;
		#pragma omp task depend(inout:X[i:i+c-1]) depend(in:f) firstprivate(i, c, m, f)
		  __t_scal(c, m, f, &X[i]);
	}
}


void VvecDoublesTasks (int bm, int m, double *src1, double *src2, double *dst) {
	int i;
	for ( i=0; i<m; i+=bm ) {
		int cs = m - i;
		int c = cs < bm ? cs : bm;
		#pragma omp task depend(in:src1[i:i+c-1], src2[i:i+c-1]) depend(out:dst[i:i+c-1]) firstprivate(i, c, m)
    {
      double *D = &dst[i];
      double *S1 = &src1[i];
      double *S2 = &src2[i];
      for(int ii=0; ii < c; ii++){
		  	D[ii] = S1[ii] * S2[ii];  
			}
    } 
	}
}
/* 
 * BLAS/LAPACK task wrappers
 * */
//#pragma omp task depend(in:x[initx:initx+bm-1]) depend(out:y[inity:inity+bm-1]) 
void __t_copy(int bm, int m, double *x, double *y, int initx, int inity) 
{
	double *X = &x[initx];
	double *Y = &y[inity];
	int i_one = 1;
	BLAS_cp(bm, X, i_one, Y, i_one);
}

//#pragma omp task depend(in:x[initx:initx+bm-1], y[inity:inity+bm-1]) depend(out:result)//concurrent(result[0:bn-1]) 
void __t_dot(int bm, int m, double *x, double *y, int initx, int inity, double *result) 
{
	double *X = &x[initx];
	double *Y = &y[inity];
	int i_one = 1;
	double local_result;
    exblas::cpu::exdot<double*, double*, NBFPE> (bm, X, Y, result);
    //local_result = BLAS_dot(bm, X, i_one, Y, i_one);

	//#pragma omp atomic //critical
	//  *result += local_result;
}


//#pragma omp task depend(in:X[0:bm-1], f) depend(out:Y[0:bm-1]) 
void __t_axpy(int bm, int m, double f, double *X, double *Y) 
{
	int i_one = 1;
	BLAS_axpy(bm, f, X, i_one, Y, i_one);
}

//#pragma omp task depend(inout:X[0:bm-1],f) 
void __t_scal(int bm, int m, double f, double *X) 
{
	int i_one = 1;
	BLAS_scal(bm, f, X, i_one);
}

