/*******************************************************************************
* Copyright 1999-2019 Intel Corporation.
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
!      Intel(R) Math Kernel Library (Intel(R) MKL) C/C++ interface for
!	   Cluster Sparse Solver
!******************************************************************************/

#if !defined( __MKL_CLUSTER_SPARSE_SOLVER_H )

#include "mkl_types.h"

#define __MKL_CLUSTER_SPARSE_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void cluster_sparse_solver(
     void *, const MKL_INT *, const MKL_INT *, const MKL_INT *, const MKL_INT *, const MKL_INT *,
     const void *, const MKL_INT *, const MKL_INT *, MKL_INT *, const MKL_INT *, MKL_INT *,
     const MKL_INT *,       void *,       void *, const int *, MKL_INT *);

void CLUSTER_SPARSE_SOLVER(
     void *, const MKL_INT *, const MKL_INT *, const MKL_INT *, const MKL_INT *, const MKL_INT *,
     const void *, const MKL_INT *, const MKL_INT *, MKL_INT *, const MKL_INT *, MKL_INT *,
     const MKL_INT *,       void *,       void *, const int *, MKL_INT *);

void cluster_sparse_solver_64(
     void *, const long long int *, const long long int *, const long long int *, const long long int *, const long long int *,
     const void *, const long long int *, const long long int *, long long int *, const long long int *, long long int *,
     const long long int *,       void *,       void *, const int *, long long int *);

void CLUSTER_SPARSE_SOLVER_64(
     void *, const long long int *, const long long int *, const long long int *, const long long int *, const long long int *,
     const void *, const long long int *, const long long int *, long long int *, const long long int *, long long int *,
     const long long int *,       void *,       void *, const int *, long long int *);

typedef enum mkl_dss_export_data {DSS_EXPORT_DATA_MIN = 0,
                                  SPARSE_PTLUQT_L = 0,
                                  SPARSE_PTLUQT_U,
                                  SPARSE_PTLUQT_P,
                                  SPARSE_PTLUQT_Q,
                                  SPARSE_DPTLUQT_L,
                                  SPARSE_DPTLUQT_U,
                                  SPARSE_DPTLUQT_P,
                                  SPARSE_DPTLUQT_Q,
                                  SPARSE_DPTLUQT_D,
                                  DSS_EXPORT_DATA_MAX = SPARSE_DPTLUQT_D} _MKL_DSS_EXPORT_DATA;

typedef enum mkl_dss_export_operation {DSS_EXPORT_OPERATION_MIN = 0,
                                       SPARSE_PTLUQT = 0,
                                       SPARSE_DPTLUQT,
                                       DSS_EXPORT_OPERATION_MAX = SPARSE_DPTLUQT} _MKL_DSS_EXPORT_OPERATION;

void cluster_sparse_solver_get_csr_size(void *, const int, MKL_INT *, MKL_INT *, const int *, MKL_INT *);

void cluster_sparse_solver_set_csr_ptrs(void *, const int, MKL_INT *, MKL_INT *, void *, const int *, MKL_INT *);

void cluster_sparse_solver_set_ptr(void *, const int, void *, const int *, MKL_INT *);

void cluster_sparse_solver_export(void *, const int, const int *, MKL_INT *);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
