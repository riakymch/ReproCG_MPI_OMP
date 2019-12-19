/*
 * %%%%%%%%%%%%%%%%%%%%%%%Original development%%%%%%%%%%%%%%%%%%%%%%%%%
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 * %%%%%%%%%%%%%%%%%%%%%%%Modifications and further additions%%%%%%%%%%
 *  Matthias Wiesenberger, 2017, within FELTOR and EXBLAS licenses
 */
/**
 *  @file accumulate.h
 *  @brief Primitives for accumulation into superaccumulator
 *
 *  @authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once
#include "config.h"
#include "mylibm.hpp"
//this file has a direct correspondance to gpu code accumulate.cuh

namespace exblas {
namespace cpu {
///////////////////////////////////////////////////////////////////////////
//********* Here, the change from float to double happens ***************//
///////////////////////////////////////////////////////////////////////////
#ifndef _WITHOUT_VCL
static inline vcl::Vec8d make_vcl_vec8d( double x, int i){
    return vcl::Vec8d(x);
}
static inline vcl::Vec8d make_vcl_vec8d( const double* x, int i){
    return vcl::Vec8d().load( x+i);
}
static inline vcl::Vec8d make_vcl_vec8d( double x, int i, int num){
    return vcl::Vec8d(x);
}
static inline vcl::Vec8d make_vcl_vec8d( const double* x, int i, int num){
    return vcl::Vec8d().load_partial( num, x+i);
}
static inline vcl::Vec8d make_vcl_vec8d( float x, int i){
    return vcl::Vec8d((double)x);
}
static inline vcl::Vec8d make_vcl_vec8d( const float* x, int i){
    double tmp[8];
    for(int i=0; i<8; i++)
        tmp[i] = (double)x[i];
    return vcl::Vec8d().load( tmp);
}
static inline vcl::Vec8d make_vcl_vec8d( float x, int i, int num){
    return vcl::Vec8d((double)x);
}
static inline vcl::Vec8d make_vcl_vec8d( const float* x, int i, int num){
    double tmp[8];
    for(int i=0; i<num; i++)
        tmp[i] = (double)x[i];
    return vcl::Vec8d().load_partial( num, tmp);
}
#endif//_WITHOUT_VCL

}//namespace cpu
} //namespace exblas
