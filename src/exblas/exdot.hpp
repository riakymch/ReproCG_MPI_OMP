/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */

/**
 *  @file exdot.h
 *  @brief Serial version of exdot
 *
 *  @authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <iostream>

#include "accumulate.h"
#include "ExSUM.FPE.hpp"

namespace exblas{
///@cond
namespace cpu{

template<typename CACHE, typename PointerOrValue1, typename PointerOrValue2>
void ExDOTFPE(int bm, int N, PointerOrValue1 a, PointerOrValue2 b, int NBFPE, int64_t* acc) {
    CACHE cache(acc);
    double *fpe = (double *) calloc(NBFPE*omp_get_num_threads(), sizeof(double));

#ifndef _WITHOUT_VCL
    int r = (( int64_t(N) ) & ~7ul);
    for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
        asm ("# myloop");
#endif
        vcl::Vec8d r1 ;
        vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,i), make_vcl_vec8d(b,i), r1);
        //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load(a+i), vcl::Vec8d().load(b+i), r1);
        //vcl::Vec8d x  = vcl::Vec8d().load(a+i)*vcl::Vec8d().load(b+i);
        cache.Accumulate(x);
        cache.Accumulate(r1);
    }
    if( r != N) {
        //accumulate remainder
        vcl::Vec8d r1;
        vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,r,N-r), make_vcl_vec8d(b,r,N-r), r1);
        //vcl::Vec8d x  = TwoProductFMA(vcl::Vec8d().load_partial(N-r, a+r), vcl::Vec8d().load_partial(N-r,b+r), r1);
        //vcl::Vec8d x  = vcl::Vec8d().load_partial(N-r, a+r)*vcl::Vec8d().load_partial(N-r,b+r);
        cache.Accumulate(x);
        cache.Accumulate(r1);
    }
#else// _WITHOUT_VCL
	for (int i = 0; i < N; i += bm ) {
		int cs = N - i;
		int c = cs < bm ? cs : bm;

		#pragma omp task depend(in:a[i:i+c-1], b[i:i+c-1]) depend(out:fpe) firstprivate(i,c) 
        for(int j = i; j < (i+c); j++) {
            double r1;
            double x = TwoProductFMA(get_element(a,j),get_element(b,j),r1);
            cache.Accumulate(&fpe[omp_get_thread_num()*NBFPE], x);
            cache.Accumulate(&fpe[omp_get_thread_num()*NBFPE], r1);
        }
	}
#endif// _WITHOUT_VCL
    #pragma omp taskwait

    //#pragma omp task private(fpe)
    for (int i=0; i < omp_get_num_threads(); i++) {
        cache.Flush(&fpe[i*NBFPE]);
    }
}

/*!@brief serial version of exact dot product
 *
 * Computes the exact sum \f[ \sum_{i=0}^{N-1} x_i y_i \f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param bm size of each subproblem for each task
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @param h_superacc pointer to an array of 64 bit integers (the superaccumulator) in host memory with size at least \c exblas::BIN_COUNT (39) (contents are overwritten)
*/
template<class PointerOrValue1, class PointerOrValue2, size_t NBFPE=8>
void exdot(unsigned bm, unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, int64_t* h_superacc){
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");
    for( int i=0; i<exblas::BIN_COUNT; i++)
        h_superacc[i] = 0;
#ifndef _WITHOUT_VCL
    cpu::ExDOTFPE<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)bm,(int)size,x1_ptr,x2_ptr, NBFPE,h_superacc);
#else
    cpu::ExDOTFPE<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)bm,(int)size,x1_ptr,x2_ptr, NBFPE, h_superacc);
#endif//_WITHOUT_VCL
}

}//namespace cpu
///@endcond

}//namespace exblas
