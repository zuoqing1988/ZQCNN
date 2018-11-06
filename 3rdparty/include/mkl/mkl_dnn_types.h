/*******************************************************************************
* Copyright (c) 2015-2018, Intel Corporation
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

#ifndef _MKL_DNN_TYPES_H
#define _MKL_DNN_TYPES_H

#include <stdlib.h>

#if defined(__cplusplus_cli)
struct _uniPrimitive_s {};
struct _dnnLayout_s {};
#endif

typedef struct _uniPrimitive_s* dnnPrimitive_t;
typedef struct _dnnLayout_s* dnnLayout_t;
typedef void* dnnPrimitiveAttributes_t;

#define DNN_MAX_DIMENSION       32
#define DNN_QUERY_MAX_LENGTH    128

typedef enum {
    E_SUCCESS                   =  0,
    E_INCORRECT_INPUT_PARAMETER = -1,
    E_UNEXPECTED_NULL_POINTER   = -2,
    E_MEMORY_ERROR              = -3,
    E_UNSUPPORTED_DIMENSION     = -4,
    E_UNIMPLEMENTED             = -127
} dnnError_t;

typedef enum {
    /** GEMM base convolution (unimplemented) */
    dnnAlgorithmConvolutionGemm,
    /** Direct convolution */
    dnnAlgorithmConvolutionDirect,
    /** FFT based convolution (unimplemented) */
    dnnAlgorithmConvolutionFFT,
    /** Maximum pooling */
    dnnAlgorithmPoolingMax,
    /** Minimum pooling */
    dnnAlgorithmPoolingMin,
    /** Average pooling (padded values are not taken into account) */
    dnnAlgorithmPoolingAvgExcludePadding,
    /** Alias for average pooling (padded values are not taken into account) */
    dnnAlgorithmPoolingAvg = dnnAlgorithmPoolingAvgExcludePadding,
    /** Average pooling (padded values are taken into account) */
    dnnAlgorithmPoolingAvgIncludePadding
} dnnAlgorithm_t;

typedef enum {
    dnnResourceSrc            = 0,
    dnnResourceFrom           = 0,
    dnnResourceDst            = 1,
    dnnResourceTo             = 1,
    dnnResourceFilter         = 2,
    dnnResourceScaleShift     = 2,
    dnnResourceBias           = 3,
    dnnResourceMean           = 3,
    dnnResourceDiffSrc        = 4,
    dnnResourceDiffFilter     = 5,
    dnnResourceDiffScaleShift = 5,
    dnnResourceDiffBias       = 6,
    dnnResourceVariance       = 6,
    dnnResourceDiffDst        = 7,
    dnnResourceWorkspace      = 8,
    dnnResourceMultipleSrc    = 16,
    dnnResourceMultipleDst    = 24,
    dnnResourceNumber         = 32
} dnnResourceType_t;

typedef enum {
    dnnBorderZeros          = 0x0,
    dnnBorderZerosAsymm     = 0x100,
    dnnBorderExtrapolation  = 0x3
} dnnBorder_t;

typedef enum {
    dnnUseInputMeanVariance = 0x1U,
    dnnUseScaleShift        = 0x2U
} dnnBatchNormalizationFlag_t;

#endif
