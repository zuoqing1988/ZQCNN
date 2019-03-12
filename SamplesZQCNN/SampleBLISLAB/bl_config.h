/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_config.h
 *
 *
 * Purpose:
 * this header file contains configuration parameters.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */

#ifndef BLISLAB_CONFIG_H
#define BLISLAB_CONFIG_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

#define GEMM_SIMD_ALIGN_SIZE 32

#define DGEMM_MC 96
#define DGEMM_NC 2048
#define DGEMM_KC 256
#define DGEMM_MR 8
#define DGEMM_NR 4


//#define DGEMM_MC 72
//#define DGEMM_NC 4080
//#define DGEMM_KC 256
////#define DGEMM_MR 8
////#define DGEMM_NR 6
//#define DGEMM_MR 12
//#define DGEMM_NR 4


//Ivy Bridge
//#define SGEMM_MC 128
//#define SGEMM_NC 4096
//#define SGEMM_KC 384
//#define SGEMM_MR 8
//#define SGEMM_NR 8

//// Haswell, 16x6
//#define SGEMM_MC 144
//#define SGEMM_NC 4080
//#define SGEMM_KC 256
//#define SGEMM_MR 16
//#define SGEMM_NR 6


//// Haswell, 24x4
//#define SGEMM_MC 264
//#define SGEMM_NC 128
//#define SGEMM_KC 256
//#define SGEMM_MR 24
//#define SGEMM_NR 4


// ARM a-15, 4x4
//#define SGEMM_MC 336
#define SGEMM_MC 160
#define SGEMM_NC 4096
#define SGEMM_KC 528
#define SGEMM_MR 4
#define SGEMM_NR 4


//#define BL_MICRO_KERNEL bl_sgemm_ukr_ref
//#define BL_MICRO_KERNEL bl_sgemm_asm_8x8
//#define BL_MICRO_KERNEL bl_sgemm_asm_16x6
//#define BL_MICRO_KERNEL bl_sgemm_asm_24x4
#define BL_MICRO_KERNEL bl_sgemm_opt_4x4

// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif

