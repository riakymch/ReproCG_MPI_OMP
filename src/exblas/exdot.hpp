/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */

/**
 *  @file exdot_serial.h
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
#include <mpi.h>
#include <iostream>

#include "accumulate.h"

namespace exblas{
///@cond
namespace cpu{

/**
 * \struct FPExpansionTraits
 * \ingroup lowlevel
 * \brief This struct is meant to specify optimization or other technique used
 */
template<bool EX=false, bool FLUSHHI=false, bool H2SUM=false, bool CRF=false, bool CSWAP=false, bool B2SUM=true, bool SORT=false, bool VICT=false>
struct FPExpansionTraits
{
    static bool constexpr EarlyExit = EX;
    static bool constexpr FlushHi = FLUSHHI;
    static bool constexpr Horz2Sum = H2SUM;
    static bool constexpr CheckRangeFirst = CRF;
    static bool constexpr ConditionalSwap = CSWAP;
    static bool constexpr Biased2Sum = B2SUM;
    static bool constexpr Sort = SORT;
    static bool constexpr Victimcache = VICT;
};

/**
 * \struct FPExpansionVect
 * \ingroup lowlevel
 * \brief This struct is meant to introduce functionality for working with
 *  floating-point expansions in conjuction with superaccumulators
 */
template<typename T, int N, typename TRAITS=FPExpansionTraits<false,false> >
struct FPExpansionVect
{
    /**
     * Constructor
     * \param fpe
     */
    FPExpansionVect(T* fpeinit);
    
    /**
     * This function accumulates value x to the floating-point expansion
     * \param x input value
     */
    void Accumulate(T x);

private:
    static T twosum(T a, T b, T & s);

    T* fpe;
};

template<typename T, int N, typename TRAITS>
FPExpansionVect<T,N,TRAITS>::FPExpansionVect(T* fpeinit) :
    fpe(fpeinit)
{}

// Knuth 2Sum.
template<typename T>
inline static T KnuthTwoSum(T a, T b, T & s)
{
    T r = a + b;
    T z = r - a;
    s = (a - (r - z)) + (b - z);
    return r;
}
template<typename T>
inline static T TwoProductFMA(T a, T b, T &d) {
    T p = a * b;
#ifdef _WITHOUT_VCL
    d = fma(a,b,-p);
#else
    d = vcl::mul_sub_x(a, b, p); //extra precision even if FMA is not available
#endif//_WITHOUT_VCL
    return p;
}

// Knuth 2Sum with FMAs
template<typename T>
inline static T FMA2Sum(T a, T b, T & s)
{
#ifndef _WITHOUT_VCL
    T r = a + b;
    T z = vcl::mul_sub(1., r, a);
    s = vcl::mul_add(1., a - vcl::mul_sub(1., r, z), b - z);
    return r;
#else
    T r = a + b;
    T z = r - a;
    s = (a - (r - z)) + (b - z);
    return r;
#endif//_WITHOUT_VCL
}

template<typename T, int N, typename TRAITS> UNROLL_ATTRIBUTE
void FPExpansionVect<T,N,TRAITS>::Accumulate(T x)
{
    // Experimental
    if(TRAITS::CheckRangeFirst && horizontal_or(abs(x) < abs(fpe[N-1]))) {
        return;
    }

    T s;
    for(unsigned int i = 0; i != N; ++i) {
        fpe[i] = twosum(fpe[i], x, s);
        x = s;
        if(TRAITS::EarlyExit && !horizontal_or(x))
	    return;
    }

//    if (horizontal_or(x)) {
//        fprintf(stderr, "WARN: with the FPE-based implementation we cannot keep every bit of information for this problem due to either high condition number (ill-cond), too broad dynamic range, or both. Thus, we cannot ensure correct-rounding and bitwise reproducibility. Hence, we advise to switch to the ExBLAS-based implementation for this particular (rather rare) case.\n");
//    }
}

template<typename T, int N, typename TRAITS>
T FPExpansionVect<T,N,TRAITS>::twosum(T a, T b, T & s)
{
//#if INSTRSET > 7                       // AVX2 and later
	// Assume Haswell-style architecture with parallel Add and FMA pipelines
	return FMA2Sum(a, b, s);
//#else
    //if(TRAITS::Biased2Sum) {
    //    return BiasedSIMD2Sum(a, b, s);
    //}
    //else {
    //    return KnuthTwoSum(a, b, s);
    //}
//#endif
}

template<typename CACHE, typename PointerOrValue1, typename PointerOrValue2>
void ExDOTFPE(int N, PointerOrValue1 a, PointerOrValue2 b, int NBFPE, double* fpe) {
    // declare fpe for accumulating errors
    double fperr[NBFPE];
    for( int i=0; i<NBFPE; i++)
        fperr[i] = 0.0;
    CACHE cache(fpe);
    CACHE cacherr(fperr);
#ifndef _WITHOUT_VCL
    int r = (( int64_t(N) ) & ~7ul);
    for(int i = 0; i < r; i+=8) {
#ifndef _MSC_VER
        asm ("# myloop");
#endif
        vcl::Vec8d r1 ;
        vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,i), make_vcl_vec8d(b,i), r1);
        cache.Accumulate(x);
        cacherr.Accumulate(r1);
    }
    if( r != N) {
        //accumulate remainder
        vcl::Vec8d r1;
        vcl::Vec8d x  = TwoProductFMA(make_vcl_vec8d(a,r,N-r), make_vcl_vec8d(b,r,N-r), r1);
        cache.Accumulate(x);
        cacherr.Accumulate(r1);
    }
#else// _WITHOUT_VCL
    for(int i = 0; i < N; i++) {
        double r1;
        double x = TwoProductFMA(a[i],b[i],r1);
        cache.Accumulate(x);
        cacherr.Accumulate(r1);
    }
#endif// _WITHOUT_VCL

    // merge fpe and fperr
    for(int i = 0; i < NBFPE; i++) {
        cache.Accumulate(fperr[i]);
    }
}

/*!@brief serial version of exact dot product
 *
 * Computes the exact dot \f[ \sum_{i=0}^{N-1} x_i y_i \f]
 * @ingroup highlevel
 * @tparam NBFPE size of the floating point expansion (should be between 3 and 8)
 * @tparam PointerOrValue must be one of <tt> T, T&&, T&, const T&, T* or const T* </tt>, where \c T is either \c float or \c double. If it is a pointer type, then we iterate through the pointed data from 0 to \c size, else we consider the value constant in every iteration.
 * @param size size N of the arrays to sum
 * @param x1_ptr first array
 * @param x2_ptr second array
 * @sa \c exblas::cpu::Round  to convert the superaccumulator into a double precision number
*/
template<class PointerOrValue1, class PointerOrValue2, size_t NBFPE=8>
void exdot(unsigned size, PointerOrValue1 x1_ptr, PointerOrValue2 x2_ptr, double* fpe){
    static_assert( has_floating_value<PointerOrValue1>::value, "PointerOrValue1 needs to be T or T* with T one of (const) float or (const) double");
    static_assert( has_floating_value<PointerOrValue2>::value, "PointerOrValue2 needs to be T or T* with T one of (const) float or (const) double");

#ifndef _WITHOUT_VCL
    cpu::ExDOTFPE<cpu::FPExpansionVect<vcl::Vec8d, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, NBFPE, fpe);
#else
    cpu::ExDOTFPE<cpu::FPExpansionVect<double, NBFPE, cpu::FPExpansionTraits<true> > >((int)size,x1_ptr,x2_ptr, NBFPE, fpe);
#endif//_WITHOUT_VCL
}

}//namespace cpu
///@endcond

}//namespace exblas
