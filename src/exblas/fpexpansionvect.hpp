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
#ifndef FPEXPANSIONVECT_HPP_INCLUDED
#define FPEXPANSIONVECT_HPP_INCLUDED
#include "accumulate.h"
#include "nearsum.hpp"

namespace exblas
{
namespace cpu
{

template<typename T, int N>
inline static T Round( T *fpe ) { 
    return NearSum(N, fpe, 1);
}

/**
* @brief Convert a fpe to the nearest double precision number (CPU version)
*
* @ingroup highlevel
* @param fpe a pointer to N doubles on the CPU (representing the fpe)
* @return the double precision number nearest to the fpe
*/
template<typename T>
inline static T Round( const T *fpe ) {

    // Now add3(hi, mid, lo)
    // Adapted from:
    // Sylvie Boldo, and Guillaume Melquiond. "Emulation of a FMA and correctly rounded sums: proved algorithms using rounding to odd." IEEE Transactions on Computers, 57, no. 4 (2008): 462-471.
    union {
        T d;
        int64_t l;
    } thdb;

    T tl;
    T th = FMA2Sum(fpe[1], fpe[2], tl);
   
    if (tl != 0.0) {
        thdb.d = th;
        // if the mantissa of th is odd, there is nothing to do
        if (!(thdb.l & 1)) {
            // choose the rounding direction
            // depending of the signs of th and tl
            if ((tl > 0.0) ^ (th < 0.0))
                thdb.l++;
            else
                thdb.l--;
            th = thdb.d;
        }
        
    } 

    // final addition rounded to nearest
    return fpe[0] + th;
}

#undef IACA
#undef IACA_START
#undef IACA_END

}//namespace cpu
}//namespace exblas
#endif
