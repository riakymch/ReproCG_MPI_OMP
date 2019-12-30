
#include <vector>
#include <omp.h>

#include "exblas/exdot.hpp"
#include "cg_aux.h"

/* 
 * operation to reduce fpes 
 */ 
void fpeSum_omp( double *in, double *inout, int len) { 

    double s;
    for (int j = 0; j < len; ++j) { 
        if (in[j] == 0.0)
            return;

        for (int i = 0; i < len; ++i) { 
            inout[i] = exblas::cpu::FMA2Sum(inout[i], in[j], s);
            in[j] = s;
            if(true && !(in[j] != 0))
                break;
        }
    }
}

void fpeSum( double *in, double *inout, int *len, MPI_Datatype *dptr ) { 

    double s;
    for (int j = 0; j < *len; ++j) { 
        if (in[j] == 0.0)
            return;

        for (int i = 0; i < *len; ++i) { 
            inout[i] = exblas::cpu::FMA2Sum(inout[i], in[j], s);
            in[j] = s;
            if(true && !(in[j] != 0))
                break;
        }
    }
}

void fpeSum2( double *in, double *inout, int *len, MPI_Datatype *dptr ) { 

    double s;
    // for the first fpe
    for (int j = 0; j < *len/2; ++j) { 
        if (in[j] == 0.0)
            break;

        for (int i = 0; i < *len/2; ++i) { 
            inout[i] = exblas::cpu::FMA2Sum(inout[i], in[j], s);
            in[j] = s;
            if(true && !(in[j] != 0))
                break;
        }
    }

    // for the second fpe
    for (int j = *len/2; j < *len; ++j) { 
        if (in[j] == 0.0)
            return;

        for (int i = *len/2; i < *len; ++i) { 
            inout[i] = exblas::cpu::FMA2Sum(inout[i], in[j], s);
            in[j] = s;
            if(true && !(in[j] != 0))
                break;
        }
    }
}

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
    int i;
    //#pragma omp task depend(inout:result[omp_get_thread_num()*NBFPE:omp_get_thread_num()*NBFPE+NBFPE-1]) firstprivate(i)
    for (i=0; i < NBFPE * omp_get_num_threads(); i++) 
        result[i] = 0.0;
        //result[i + omp_get_thread_num()*NBFPE] = 0.0;

	for (int i=0; i<m; i+=bm ) {
		int cs = m - i;
		int c = cs < bm ? cs : bm;
		#pragma omp task depend(in:X[i:i+c-1], Y[i:i+c-1]) depend(out:result) firstprivate(i,c,m) //private(result)
		  __t_dot(c, m, X, Y, i, i, &result[NBFPE * omp_get_thread_num()]);
	}
    #pragma omp taskwait

    // reduction
    for (int i=1; i < omp_get_num_threads(); i++) 
        fpeSum_omp(&result[NBFPE*i], &result[0], NBFPE);
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

    exblas::cpu::exdot<double*, double*, NBFPE> (bm, X, Y, &result[0]);
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

