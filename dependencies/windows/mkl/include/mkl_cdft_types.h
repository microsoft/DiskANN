/*******************************************************************************
* Copyright 2002-2019 Intel Corporation.
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

/*
!  Content:
!      Intel(R) Math Kernel Library (Intel(R) MKL) interface for Cluster DFT routines
!******************************************************************************/

/* Avoid multiple definition */
#ifndef _MKL_CDFT_TYPES_H_
#define _MKL_CDFT_TYPES_H_

/* Include header-files */
#include "mkl_dfti.h"

/* Keep C++ compilers from getting confused */
#ifdef __cplusplus
extern "C" {
#endif

/* Codes of errors */
#define CDFT_MPI_ERROR      1000
#define CDFT_SPREAD_ERROR   1001

/* Codes of parameters for DftiGetValueDM / DftiSetValueDM */
enum CDFT_CONFIG_PARAM {
    CDFT_LOCAL_SIZE         =1000,
    CDFT_LOCAL_X_START      =1001,
    CDFT_LOCAL_NX           =1002,
    CDFT_MPI_COMM           =1003,
    CDFT_WORKSPACE          =1004,
    CDFT_LOCAL_OUT_X_START  =1005,
    CDFT_LOCAL_OUT_NX       =1006
};

/* Definition of handle to descriptor */
typedef struct _DFTI_DESCRIPTOR_DM* DFTI_DESCRIPTOR_DM_HANDLE;

/* Keep C++ compilers from getting confused (extern "C" {) */
#ifdef __cplusplus
}
#endif

/* Avoid multiple definition (#ifndef _MKL_CDFT_TYPES_H_) */
#endif

