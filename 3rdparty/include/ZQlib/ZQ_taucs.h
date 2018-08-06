#ifndef _ZQ_TAUCS_H_
#define _ZQ_TAUCS_H_

#pragma once
/******************************************************
WARNING: when using "ZQ_taucs.h", DON'T use "taucs.h"
******************************************************/

#define taucs_double double
#define taucs_single float

#define TAUCS_SUCCESS                       0
#define TAUCS_ERROR                        -1
#define TAUCS_ERROR_NOMEM                  -2
#define TAUCS_ERROR_BADARGS                -3
#define TAUCS_ERROR_INDEFINITE             -4
#define TAUCS_ERROR_MAXDEPTH               -5

#define TAUCS_INT       1024
#define TAUCS_DOUBLE    2048
#define TAUCS_SINGLE    4096
#define TAUCS_DCOMPLEX  8192
#define TAUCS_SCOMPLEX 16384

#define TAUCS_LOWER      1
#define TAUCS_UPPER      2
#define TAUCS_TRIANGULAR 4
#define TAUCS_SYMMETRIC  8
#define TAUCS_HERMITIAN  16
#define TAUCS_PATTERN    32

#define TAUCS_METHOD_LLT  1
#define TAUCS_METHOD_LDLT 2
#define TAUCS_METHOD_PLU  3

#define TAUCS_VARIANT_SNMF 1
#define TAUCS_VARIANT_SNLL 2

typedef struct
{
	int     n;    /* columns                      */
	int     m;    /* rows; don't use if symmetric   */
	int     flags;
	int*    colptr; /* pointers to where columns begin in rowind and values. */
	/* 0-based. Length is (n+1). */
	int*    rowind; /* row indices */

	union {
		void*           v;
		taucs_double*   d;
		taucs_single*   s;
	} values;

} taucs_ccs_matrix;

#endif