/*******************************************************************************
* Copyright 2015-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
