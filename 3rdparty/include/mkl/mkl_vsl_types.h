/* file: mkl_vsl_types.h */
/*******************************************************************************
* Copyright (c) 2006-2018, Intel Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/*
//++
//  This file contains user-level type definitions.
//--
*/

#ifndef __MKL_VSL_TYPES_H__
#define __MKL_VSL_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "mkl_types.h"

/*
//++
//  TYPEDEFS
//--
*/

/*
//  POINTER TO STREAM STATE STRUCTURE
//  This is a void pointer to hide implementation details.
*/
typedef void* VSLStreamStatePtr;
typedef void* VSLConvTaskPtr;
typedef void* VSLCorrTaskPtr;
typedef void* VSLSSTaskPtr;

/*
//  POINTERS TO BASIC RANDOM NUMBER GENERATOR FUNCTIONS
//  Each BRNG must have following implementations:
//
//  * Stream initialization (InitStreamPtr)
//  * Integer-value recurrence implementation (iBRngPtr)
//  * Single precision implementation (sBRngPtr) - for random number generation
//    uniformly distributed on the [a,b] interval
//  * Double precision implementation (dBRngPtr) - for random number generation
//    uniformly distributed on the [a,b] interval
*/
typedef int (*InitStreamPtr)( int method, VSLStreamStatePtr stream, \
        int n, const unsigned int params[] );
typedef int (*sBRngPtr)( VSLStreamStatePtr stream, int n, float r[], \
        float a, float b );
typedef int (*dBRngPtr)( VSLStreamStatePtr stream, int n, double r[], \
        double a, double b );
typedef int (*iBRngPtr)( VSLStreamStatePtr stream, int n, unsigned int r[] );

/*********** Pointers to callback functions for abstract streams *************/
typedef int (*iUpdateFuncPtr)( VSLStreamStatePtr stream, int* n, \
     unsigned int ibuf[], int* nmin, int* nmax, int* idx );
typedef int (*dUpdateFuncPtr)( VSLStreamStatePtr stream, int* n,
     double dbuf[], int* nmin, int* nmax, int* idx );
typedef int (*sUpdateFuncPtr)( VSLStreamStatePtr stream, int* n, \
     float sbuf[], int* nmin, int* nmax, int* idx );


/*
//  BASIC RANDOM NUMBER GENERATOR PROPERTIES STRUCTURE
//  The structure describes the properties of given basic generator, e.g. size
//  of the stream state structure, pointers to function implementations, etc.
//
//  BRNG properties structure fields:
//  StreamStateSize - size of the stream state structure (in bytes)
//  WordSize        - size of base word (in bytes). Typically this is 4 bytes.
//  NSeeds          - number of words necessary to describe generator's state
//  NBits           - number of bits actually used in base word. For example,
//                    only 31 least significant bits are actually used in
//                    basic random number generator MCG31m1 with 4-byte base
//                    word. NBits field is useful while interpreting random
//                    words as a sequence of random bits.
//  IncludesZero    - FALSE if 0 cannot be generated in integer-valued
//                    implementation; TRUE if 0 can be potentially generated in
//                    integer-valued implementation.
//  InitStream      - pointer to stream state initialization function
//  sBRng           - pointer to single precision implementation
//  dBRng           - pointer to double precision implementation
//  iBRng           - pointer to integer-value implementation
*/
typedef struct _VSLBRngProperties {
    int StreamStateSize;       /* Stream state size (in bytes) */
    int NSeeds;                /* Number of seeds */
    int IncludesZero;          /* Zero flag */
    int WordSize;              /* Size (in bytes) of base word */
    int NBits;                 /* Number of actually used bits */
    InitStreamPtr InitStream;  /* Pointer to InitStream func */
    sBRngPtr sBRng;            /* Pointer to S func */
    dBRngPtr dBRng;            /* Pointer to D func */
    iBRngPtr iBRng;            /* Pointer to I func */
} VSLBRngProperties;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VSL_TYPES_H__ */
