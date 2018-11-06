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

#ifndef _MKL_DNN_H
#define _MKL_DNN_H

#include <stdarg.h>
#include <stddef.h>

#include "mkl_dnn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * F32 section: single precision
 ******************************************************************************/

dnnError_t dnnLayoutCreate_F32(
        dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t dnnLayoutCreateFromPrimitive_F32(
        dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type);

/** Returns the size of buffer required to serialize dnnLayout_t structure. */
size_t dnnLayoutSerializationBufferSize_F32();

/** Serializes given @p layout into buffer @p buf. User-provided buffer @p buf
 * should have enough space to store dnnLayout_t structure.
 * @sa dnnLayoutSerializationBufferSize_F32 */
dnnError_t dnnLayoutSerialize_F32(const dnnLayout_t layout, void *buf);

/** Creates new layout restored from previously serialized one. */
dnnError_t dnnLayoutDeserialize_F32(dnnLayout_t *pLayout, const void *buf);

size_t dnnLayoutGetMemorySize_F32(
        const dnnLayout_t layout);
int dnnLayoutCompare_F32(
        const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t dnnAllocateBuffer_F32(
        void **pPtr, dnnLayout_t layout);
dnnError_t dnnReleaseBuffer_F32(
        void *ptr);
dnnError_t dnnLayoutDelete_F32(
        dnnLayout_t layout);

dnnError_t dnnPrimitiveAttributesCreate_F32(
        dnnPrimitiveAttributes_t *attributes);
dnnError_t dnnPrimitiveAttributesDestroy_F32(
        dnnPrimitiveAttributes_t attributes);
dnnError_t dnnPrimitiveGetAttributes_F32(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes);

dnnError_t dnnExecute_F32(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnExecuteAsync_F32(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnWaitFor_F32(
        dnnPrimitive_t primitive);
dnnError_t dnnDelete_F32(
        dnnPrimitive_t primitive);

dnnError_t dnnConversionCreate_F32(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);
dnnError_t dnnConversionExecute_F32(
        dnnPrimitive_t conversion, void *from, void *to);

dnnError_t dnnSumCreate_F32(
        dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
        dnnLayout_t layout, float *coefficients);
dnnError_t dnnConcatCreate_F32(
        dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src);
dnnError_t dnnSplitCreate_F32(
        dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
        dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t dnnScaleCreate_F32(
        dnnPrimitive_t *pScale,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float alpha);

dnnError_t dnnConvolutionCreateForward_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateForwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardData_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardFilter_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[]);

dnnError_t dnnGroupsConvolutionCreateForward_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateForwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardData_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardFilter_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[]);

dnnError_t dnnReLUCreateForward_F32(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t dnnReLUCreateBackward_F32(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, float negativeSlope);

dnnError_t dnnLRNCreateForward_F32(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);
dnnError_t dnnLRNCreateBackward_F32(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);

dnnError_t dnnBatchNormalizationCreateForward_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps);
dnnError_t dnnBatchNormalizationCreateBackwardScaleShift_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps);
dnnError_t dnnBatchNormalizationCreateBackwardData_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps);

dnnError_t dnnBatchNormalizationCreateForward_v2_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps,
        unsigned int flags);
dnnError_t dnnBatchNormalizationCreateBackward_v2_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps,
        unsigned int flags);

dnnError_t dnnPoolingCreateForward_F32(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnPoolingCreateBackward_F32(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);

dnnError_t dnnInnerProductCreateForward_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateForwardBias_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardData_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardFilter_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardBias_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t dstSize[]);

/*******************************************************************************
 * F64 section: double precision
 ******************************************************************************/

dnnError_t dnnLayoutCreate_F64 (
        dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t dnnLayoutCreateFromPrimitive_F64(
        dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type);

/** Returns the size of buffer required to serialize dnnLayout_t structure. */
size_t dnnLayoutSerializationBufferSize_F64();

/** Serializes given @p layout into buffer @p buf. User-provided buffer @p buf
 * should have enough space to store dnnLayout_t structure.
 * @sa dnnLayoutSerializationBufferSize_F64 */
dnnError_t dnnLayoutSerialize_F64(const dnnLayout_t layout, void *buf);

/** Creates new layout restored from previously serialized one. */
dnnError_t dnnLayoutDeserialize_F64(dnnLayout_t *pLayout, const void *buf);

size_t dnnLayoutGetMemorySize_F64(
        const dnnLayout_t layout);
int dnnLayoutCompare_F64(
        const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t dnnAllocateBuffer_F64(
        void **pPtr, dnnLayout_t layout);
dnnError_t dnnReleaseBuffer_F64(
        void *ptr);
dnnError_t dnnLayoutDelete_F64(
        dnnLayout_t layout);

dnnError_t dnnPrimitiveAttributesCreate_F64(
        dnnPrimitiveAttributes_t *attributes);
dnnError_t dnnPrimitiveAttributesDestroy_F64(
        dnnPrimitiveAttributes_t attributes);
dnnError_t dnnPrimitiveGetAttributes_F64(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes);

dnnError_t dnnExecute_F64(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnExecuteAsync_F64(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnWaitFor_F64(
        dnnPrimitive_t primitive);
dnnError_t dnnDelete_F64(
        dnnPrimitive_t primitive);

dnnError_t dnnConversionCreate_F64(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);
dnnError_t dnnConversionExecute_F64(
        dnnPrimitive_t conversion, void *from, void *to);

dnnError_t dnnSumCreate_F64(
        dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
        dnnLayout_t layout, double *coefficients);
dnnError_t dnnConcatCreate_F64(
         dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src);
dnnError_t dnnSplitCreate_F64(
        dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
        dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t dnnScaleCreate_F64(
        dnnPrimitive_t *pScale,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double alpha);

dnnError_t dnnConvolutionCreateForward_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateForwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardData_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardFilter_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[]);

dnnError_t dnnGroupsConvolutionCreateForward_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateForwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardData_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardFilter_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[]);

dnnError_t dnnReLUCreateForward_F64(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t dnnReLUCreateBackward_F64(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, double negativeSlope);

dnnError_t dnnLRNCreateForward_F64(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta, double k);
dnnError_t dnnLRNCreateBackward_F64(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta, double k);

dnnError_t dnnBatchNormalizationCreateForward_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps);
dnnError_t dnnBatchNormalizationCreateBackwardScaleShift_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps);
dnnError_t dnnBatchNormalizationCreateBackwardData_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps);

dnnError_t dnnBatchNormalizationCreateForward_v2_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps,
        unsigned int flags);
dnnError_t dnnBatchNormalizationCreateBackward_v2_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps,
        unsigned int flags);

dnnError_t dnnPoolingCreateForward_F64(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnPoolingCreateBackward_F64(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);

dnnError_t dnnInnerProductCreateForward_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateForwardBias_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardData_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardFilter_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardBias_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t dstSize[]);

#ifdef __cplusplus
}
#endif

#endif
