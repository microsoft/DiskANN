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
#ifndef _MKL_CDFT_H_
#define _MKL_CDFT_H_

/* Include header-files */
#include "mpi.h"
#include "mkl_cdft_types.h"

/* Keep C++ compilers from getting confused */
#ifdef __cplusplus
extern "C" {
#endif

/* Prototypes of routines */
extern MKL_LONG DftiCreateDescriptorDM(MPI_Comm,DFTI_DESCRIPTOR_DM_HANDLE*,enum DFTI_CONFIG_VALUE,enum DFTI_CONFIG_VALUE,MKL_LONG,...);
extern MKL_LONG DftiGetValueDM(DFTI_DESCRIPTOR_DM_HANDLE,int,...);
extern MKL_LONG DftiSetValueDM(DFTI_DESCRIPTOR_DM_HANDLE,int,...);
extern MKL_LONG DftiCommitDescriptorDM(DFTI_DESCRIPTOR_DM_HANDLE);
extern MKL_LONG DftiComputeForwardDM(DFTI_DESCRIPTOR_DM_HANDLE,void*,...);
extern MKL_LONG DftiComputeBackwardDM(DFTI_DESCRIPTOR_DM_HANDLE,void*,...);
extern MKL_LONG DftiFreeDescriptorDM(DFTI_DESCRIPTOR_DM_HANDLE*);

/* Keep C++ compilers from getting confused (extern "C" {) */
#ifdef __cplusplus
}
#endif

/* Avoid multiple definition (#ifndef _MKL_CDFT_H_) */
#endif
