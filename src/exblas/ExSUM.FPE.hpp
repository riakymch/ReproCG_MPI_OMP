/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file ExSUM.FPE.hpp
 *  \brief A set of routines concerning floating-point expansions
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#ifndef EXSUM_FPE_HPP_
#define EXSUM_FPE_HPP_
#include "accumulate.h"

namespace exblas
{
namespace cpu
{

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
     * \param sa superaccumulator
     */
    FPExpansionVect(int64_t* sa);

    /**
     * This function accumulates value x to the floating-point expansion
     * \param a fpe
     * \param x input value
     */
    void Accumulate(T *a, T x);

    /**
     * This function is used to flush the floating-point expansion to the superaccumulator
     * \param a fpe
     */
    void Flush(T *a);

private:
    void FlushVector(T x) const;
    static void Swap(T & x1, T & x2);
    static T twosum(T a, T b, T & s);

    int64_t* superacc;

    // Most significant digits first!
    T victim;
};

template<typename T, int N, typename TRAITS>
FPExpansionVect<T,N,TRAITS>::FPExpansionVect(int64_t * sa) :
    superacc(sa),
    victim(0)
{
//    std::fill(a, a + N, 0);
}

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
void FPExpansionVect<T,N,TRAITS>::Accumulate(T *fpe, T x)
{
    // Experimental
    if(TRAITS::CheckRangeFirst && horizontal_or(abs(x) < abs(fpe[N-1]))) {
        FlushVector(x);
        return;
    }
    T s;
    for(unsigned int i = 0; i != N; ++i) {
        fpe[i] = twosum(fpe[i], x, s);
        x = s;
        if(TRAITS::EarlyExit && !horizontal_or(x))
            return;
    }
    if(horizontal_or(x)) {
        FlushVector(x);
    }
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

#undef IACA
#undef IACA_START
#undef IACA_END

template<typename T, int N, typename TRAITS>
void FPExpansionVect<T,N,TRAITS>::Flush(T *a)
{
    for(unsigned int i = 0; i != N; ++i)
    {
        FlushVector(a[i]);
        a[i] = 0;
    }
    if(TRAITS::Victimcache) {
        FlushVector(victim);
    }
}

template<typename T, int N, typename TRAITS> inline
void FPExpansionVect<T,N,TRAITS>::FlushVector(T x) const
{
    // TODO: update status, handle Inf/Overflow/NaN cases
    // TODO: make it work for other values of 4
    #pragma omp critical
    exblas::cpu::Accumulate(superacc, x);
}

}//namespace cpu
}//namespace exblas
#endif // EXSUM_FPE_HPP_
