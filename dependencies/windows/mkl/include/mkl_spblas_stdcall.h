/*******************************************************************************
* Copyright 2005-2019 Intel Corporation.
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
!    Intel(R) Math Kernel Library (Intel(R) MKL) stdcall interface for Sparse BLAS level 2,3
!    routines
!******************************************************************************/

#ifndef _MKL_SPBLAS_STDCALL_H_
#define _MKL_SPBLAS_STDCALL_H_

#include "mkl_types.h"

#ifdef __GNUC__
#define MKL_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define MKL_DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: MKL SpBLAS was declared deprecated. Use MKL IE SpBLAS instead")
#define MKL_DEPRECATED
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if defined(MKL_STDCALL)

/* Float */
/* Sparse BLAS Level2 lower case */
MKL_DEPRECATED void __stdcall mkl_scsrmv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall mkl_scsrsv(const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_scsrgemv(const char *transa, int transa_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_scsrgemv(const char *transa, int transa_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_scsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_scsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_scsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_scsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);

MKL_DEPRECATED void __stdcall mkl_scscmv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall mkl_scscsv(const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);

MKL_DEPRECATED void __stdcall mkl_scoomv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall mkl_scoosv(const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_scoogemv(const char *transa, int transa_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_scoogemv(const char *transa, int transa_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_scoosymv(const char *uplo, int uplo_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_scoosymv(const char *uplo, int uplo_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_scootrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_scootrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);

MKL_DEPRECATED void __stdcall mkl_sdiamv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall mkl_sdiasv (const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_sdiagemv(const char *transa, int transa_len, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_sdiasymv(const char *uplo, int uplo_len, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_sdiatrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);

MKL_DEPRECATED void __stdcall mkl_sskymv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *pntr, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall mkl_sskysv(const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *pntr, const float *x, float *y);

MKL_DEPRECATED void __stdcall mkl_sbsrmv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall mkl_sbsrsv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_sbsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_sbsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_sbsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_sbsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_sbsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_sbsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);

/* Sparse BLAS Level3 lower case */
MKL_DEPRECATED void __stdcall mkl_scsrmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_scsrsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_scscmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_scscsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_scoomm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_scoosm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_sdiamm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_sdiasm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_sskysm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *pntr, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_sskymm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *pntr, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_sbsrmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_sbsrsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

/* Upper case declaration */
/* Sparse BLAS Level2 upper case */
MKL_DEPRECATED void __stdcall MKL_SCSRMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall MKL_SCSRSV(const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SCSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_SCSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SCSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_SCSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SCSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_SCSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);

MKL_DEPRECATED void __stdcall MKL_SCSCMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall MKL_SCSCSV(const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);

MKL_DEPRECATED void __stdcall MKL_SCOOMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall MKL_SCOOSV(const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SCOOGEMV(const char *transa, int transa_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_SCOOGEMV(const char *transa, int transa_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SCOOSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_SCOOSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SCOOTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_SCOOTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *x, float *y);

MKL_DEPRECATED void __stdcall MKL_SDIAMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall MKL_SDIASV (const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SDIAGEMV(const char *transa, int transa_len, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SDIASYMV(const char *uplo, int uplo_len, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SDIATRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *x, float *y);

MKL_DEPRECATED void __stdcall MKL_SSKYMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *pntr, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall MKL_SSKYSV(const char *transa, int transa_len, const MKL_INT *m, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *pntr, const float *x, float *y);

MKL_DEPRECATED void __stdcall MKL_SBSRMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, const float *beta, float *y);
MKL_DEPRECATED void __stdcall MKL_SBSRSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SBSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_SBSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SBSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_SBSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_SBSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_SBSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const float *a, const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);

/* Sparse BLAS Level3 upper case */
MKL_DEPRECATED void __stdcall MKL_SCSRMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_SCSRSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_SCSCMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_SCSCSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_SCOOMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_SCOOSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_SDIAMM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_SDIASM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_SSKYSM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *pntr, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_SSKYMM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *pntr, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_SBSRMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_SBSRSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const float *alpha, const char *matdescra, int matdescra_len, const float *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const float *b, const MKL_INT *ldb, float *c, const MKL_INT *ldc);

/* Double */
/* Sparse BLAS Level2 lower case */
MKL_DEPRECATED void __stdcall mkl_dcsrmv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall mkl_dcsrsv(const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_dcsrgemv(const char *transa, int transa_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_dcsrgemv(const char *transa, int transa_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_dcsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_dcsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_dcsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_dcsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);

MKL_DEPRECATED void __stdcall mkl_dcscmv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall mkl_dcscsv(const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);

MKL_DEPRECATED void __stdcall mkl_dcoomv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall mkl_dcoosv(const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_dcoogemv(const char *transa, int transa_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_dcoogemv(const char *transa, int transa_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_dcoosymv(const char *uplo, int uplo_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_dcoosymv(const char *uplo, int uplo_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_dcootrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_dcootrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);

MKL_DEPRECATED void __stdcall mkl_ddiamv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall mkl_ddiasv (const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_ddiagemv(const char *transa, int transa_len, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_ddiasymv(const char *uplo, int uplo_len, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_ddiatrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);

MKL_DEPRECATED void __stdcall mkl_dskymv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *pntr, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall mkl_dskysv(const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *pntr, const double *x, double *y);

MKL_DEPRECATED void __stdcall mkl_dbsrmv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall mkl_dbsrsv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_dbsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_dbsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_dbsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_dbsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_dbsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_dbsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);

/* Sparse BLAS Level3 lower case */
MKL_DEPRECATED void __stdcall mkl_dcsrmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_dcsrsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_dcscmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_dcscsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_dcoomm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_dcoosm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_ddiamm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_ddiasm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_dskysm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *pntr, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_dskymm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *pntr, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_dbsrmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_dbsrsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

/* Upper case declaration */
/* Sparse BLAS Level2 upper case */
MKL_DEPRECATED void __stdcall MKL_DCSRMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall MKL_DCSRSV(const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DCSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_DCSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DCSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_DCSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DCSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_DCSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);

MKL_DEPRECATED void __stdcall MKL_DCSCMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall MKL_DCSCSV(const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);

MKL_DEPRECATED void __stdcall MKL_DCOOMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall MKL_DCOOSV(const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DCOOGEMV(const char *transa, int transa_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_DCOOGEMV(const char *transa, int transa_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DCOOSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_DCOOSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DCOOTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_DCOOTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *x, double *y);

MKL_DEPRECATED void __stdcall MKL_DDIAMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall MKL_DDIASV (const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DDIAGEMV(const char *transa, int transa_len, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DDIASYMV(const char *uplo, int uplo_len, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DDIATRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *x, double *y);

MKL_DEPRECATED void __stdcall MKL_DSKYMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *pntr, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall MKL_DSKYSV(const char *transa, int transa_len, const MKL_INT *m, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *pntr, const double *x, double *y);

MKL_DEPRECATED void __stdcall MKL_DBSRMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, const double *beta, double *y);
MKL_DEPRECATED void __stdcall MKL_DBSRSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DBSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_DBSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DBSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_DBSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_DBSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_DBSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const double *a, const MKL_INT *ia, const MKL_INT *ja, const double *x, double *y);

/* Sparse BLAS Level3 upper case */
MKL_DEPRECATED void __stdcall MKL_DCSRMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_DCSRSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_DCSCMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_DCSCSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_DCOOMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_DCOOSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_DDIAMM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_DDIASM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_DSKYSM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *pntr, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_DSKYMM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *pntr, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_DBSRMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_DBSRSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const double *alpha, const char *matdescra, int matdescra_len, const double *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const double *b, const MKL_INT *ldb, double *c, const MKL_INT *ldc);

/* MKL_Complex8 */
/* Sparse BLAS Level2 lower case */
MKL_DEPRECATED void __stdcall mkl_ccsrmv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_ccsrsv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_ccsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_ccsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_ccsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_ccsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_ccsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_ccsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall mkl_ccscmv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_ccscsv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall mkl_ccoomv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_ccoosv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_ccoogemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_ccoogemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_ccoosymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_ccoosymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_ccootrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_ccootrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall mkl_cdiamv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cdiasv (const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cdiagemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cdiasymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cdiatrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall mkl_cskymv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cskysv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall mkl_cbsrmv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cbsrsv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cbsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_cbsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cbsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_cbsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cbsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_cbsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);

/* Sparse BLAS Level3 lower case */
MKL_DEPRECATED void __stdcall mkl_ccsrmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_ccsrsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_ccscmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_ccscsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_ccoomm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_ccoosm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_cdiamm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_cdiasm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_cskysm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_cskymm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_cbsrmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_cbsrsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

/* Upper case declaration */
/* Sparse BLAS Level2 upper case */
MKL_DEPRECATED void __stdcall MKL_CCSRMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CCSRSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CCSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_CCSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CCSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_CCSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CCSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_CCSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall MKL_CCSCMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CCSCSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall MKL_CCOOMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CCOOSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CCOOGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_CCOOGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CCOOSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_CCOOSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CCOOTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_CCOOTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall MKL_CDIAMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CDIASV (const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CDIAGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CDIASYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CDIATRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall MKL_CSKYMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSKYSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *x, MKL_Complex8 *y);

MKL_DEPRECATED void __stdcall MKL_CBSRMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, const MKL_Complex8 *beta, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CBSRSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CBSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_CBSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CBSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_CBSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CBSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_CBSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex8 *x, MKL_Complex8 *y);

/* Sparse BLAS Level3 upper case */
MKL_DEPRECATED void __stdcall MKL_CCSRMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_CCSRSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_CCSCMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_CCSCSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_CCOOMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_CCOOSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_CDIAMM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_CDIASM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_CSKYSM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_CSKYMM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *pntr, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_CBSRMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_CBSRSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const MKL_Complex8 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex8 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex8 *b, const MKL_INT *ldb, MKL_Complex8 *c, const MKL_INT *ldc);

/* MKL_Complex16 */
/* Sparse BLAS Level2 lower case */
MKL_DEPRECATED void __stdcall mkl_zcsrmv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zcsrsv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zcsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_zcsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zcsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_zcsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zcsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_zcsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall mkl_zcscmv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zcscsv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall mkl_zcoomv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zcoosv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zcoogemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_zcoogemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zcoosymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_zcoosymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zcootrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_zcootrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall mkl_zdiamv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zdiasv (const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zdiagemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zdiasymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zdiatrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall mkl_zskymv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zskysv(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall mkl_zbsrmv (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zbsrsv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zbsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_zbsrgemv(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zbsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_zbsrsymv(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_zbsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall mkl_cspblas_zbsrtrsv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);

/* Sparse BLAS Level3 lower case */
MKL_DEPRECATED void __stdcall mkl_zcsrmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_zcsrsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_zcscmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_zcscsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_zcoomm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_zcoosm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_zdiamm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_zdiasm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_zskysm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_zskymm (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall mkl_zbsrmm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_zbsrsm(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

/* Upper case declaration */
/* Sparse BLAS Level2 upper case */
MKL_DEPRECATED void __stdcall MKL_ZCSRMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZCSRSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZCSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_ZCSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZCSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_ZCSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZCSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_ZCSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall MKL_ZCSCMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZCSCSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall MKL_ZCOOMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZCOOSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZCOOGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_ZCOOGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZCOOSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_ZCOOSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZCOOTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_ZCOOTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall MKL_ZDIAMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZDIASV (const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZDIAGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZDIASYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZDIATRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall MKL_ZSKYMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZSKYSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *x, MKL_Complex16 *y);

MKL_DEPRECATED void __stdcall MKL_ZBSRMV (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, const MKL_Complex16 *beta, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZBSRSV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZBSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_ZBSRGEMV(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZBSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_ZBSRSYMV(const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_ZBSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);
MKL_DEPRECATED void __stdcall MKL_CSPBLAS_ZBSRTRSV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *m, const MKL_INT *lb, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_Complex16 *x, MKL_Complex16 *y);

/* Sparse BLAS Level3 upper case */
MKL_DEPRECATED void __stdcall MKL_ZCSRMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_ZCSRSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_ZCSCMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_ZCSCSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_ZCOOMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_ZCOOSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *rowind, const MKL_INT *colind, const MKL_INT *nnz, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_ZDIAMM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_ZDIASM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *lval, const MKL_INT *idiag, const MKL_INT *ndiag, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_ZSKYSM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_ZSKYMM (const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *pntr, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);

MKL_DEPRECATED void __stdcall MKL_ZBSRMM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_ZBSRSM(const char *transa, int transa_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *lb, const MKL_Complex16 *alpha, const char *matdescra, int matdescra_len, const MKL_Complex16 *val, const MKL_INT *indx, const MKL_INT *pntrb, const MKL_INT *pntre, const MKL_Complex16 *b, const MKL_INT *ldb, MKL_Complex16 *c, const MKL_INT *ldc);

/*Converters lower case*/
MKL_DEPRECATED void __stdcall mkl_dcsrbsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, double *Acsr, MKL_INT *AJ, MKL_INT *AI, double *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_dcsrcoo(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, double *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_ddnscsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, double *Adns, const MKL_INT *lda, double *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_dcsrcsc(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0, double *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_dcsrdia(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0, double *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, double *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_dcsrsky(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  double *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void __stdcall mkl_scsrbsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, float *Acsr, MKL_INT *AJ, MKL_INT *AI, float *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_scsrcoo(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, float *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_sdnscsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, float *Adns, const MKL_INT *lda, float *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_scsrcsc(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0, float *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_scsrdia(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0, float *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, float *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_scsrsky(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  float *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void __stdcall mkl_ccsrbsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, MKL_Complex8 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_Complex8 *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_ccsrcoo(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, MKL_Complex8 *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_cdnscsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, MKL_Complex8 *Adns, const MKL_INT *lda, MKL_Complex8 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_ccsrcsc(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex8 *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_ccsrdia(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex8 *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, MKL_Complex8 *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_ccsrsky(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  MKL_Complex8 *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void __stdcall mkl_zcsrbsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, MKL_Complex16 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_Complex16 *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_zcsrcoo(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, MKL_Complex16 *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_zdnscsr(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, MKL_Complex16 *Adns, const MKL_INT *lda, MKL_Complex16 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_zcsrcsc(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex16 *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_zcsrdia(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex16 *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, MKL_Complex16 *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void __stdcall mkl_zcsrsky(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  MKL_Complex16 *Asky, MKL_INT *pointers, MKL_INT *info);

/*Converters upper case*/
MKL_DEPRECATED void __stdcall MKL_DCSRBSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, double *Acsr, MKL_INT *AJ, MKL_INT *AI, double *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_DCSRCOO(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, double *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_DDNSCSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, double *Adns, const MKL_INT *lda, double *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_DCSRCSC(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0, double *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_DCSRDIA(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0, double *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, double *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_DCSRSKY(const MKL_INT *job, const MKL_INT *n, double *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  double *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void __stdcall MKL_SCSRBSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, float *Acsr, MKL_INT *AJ, MKL_INT *AI, float *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_SCSRCOO(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, float *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_SDNSCSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, float *Adns, const MKL_INT *lda, float *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_SCSRCSC(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0, float *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_SCSRDIA(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0, float *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, float *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_SCSRSKY(const MKL_INT *job, const MKL_INT *n, float *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  float *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void __stdcall MKL_CCSRBSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, MKL_Complex8 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_Complex8 *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_CCSRCOO(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, MKL_Complex8 *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_CDNSCSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, MKL_Complex8 *Adns, const MKL_INT *lda, MKL_Complex8 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_CCSRCSC(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex8 *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_CCSRDIA(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex8 *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, MKL_Complex8 *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_CCSRSKY(const MKL_INT *job, const MKL_INT *n, MKL_Complex8 *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  MKL_Complex8 *Asky, MKL_INT *pointers, MKL_INT *info);

MKL_DEPRECATED void __stdcall MKL_ZCSRBSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *mblk, const MKL_INT *ldAbsr, MKL_Complex16 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_Complex16 *Absr,  MKL_INT *AJB,  MKL_INT *AIB,  MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_ZCSRCOO(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJR, MKL_INT *AIR, MKL_INT *nnz, MKL_Complex16 *Acoo,  MKL_INT *ir,  MKL_INT *jc,  MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_ZDNSCSR(const MKL_INT *job, const MKL_INT *m, const MKL_INT *n, MKL_Complex16 *Adns, const MKL_INT *lda, MKL_Complex16 *Acsr, MKL_INT *AJ, MKL_INT *AI, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_ZCSRCSC(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex16 *Acsc, MKL_INT *AJ1, MKL_INT *AI1, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_ZCSRDIA(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0, MKL_Complex16 *Adia, const MKL_INT *ndiag, MKL_INT *distance, MKL_INT *idiag, MKL_Complex16 *Acsr_rem, MKL_INT *AJ0_rem, MKL_INT *AI0_rem, MKL_INT *info);
MKL_DEPRECATED void __stdcall MKL_ZCSRSKY(const MKL_INT *job, const MKL_INT *n, MKL_Complex16 *Acsr, MKL_INT *AJ0, MKL_INT *AI0,  MKL_Complex16 *Asky, MKL_INT *pointers, MKL_INT *info);


/*Sparse BLAS Level2 (CSR-CSR or CSR-DNS) lower case */
MKL_DEPRECATED void __stdcall mkl_dcsrmultcsr(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, double *a,  MKL_INT *ja, MKL_INT *ia, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void __stdcall mkl_dcsrmultd(const char *transa, int transa_len,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_dcsradd(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, double *a, MKL_INT *ja, MKL_INT *ia,  const double *beta, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void __stdcall mkl_scsrmultcsr(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia, float *b, MKL_INT *jb, MKL_INT *ib,  float *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void __stdcall mkl_scsrmultd(const char *transa, int transa_len,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia, float *b, MKL_INT *jb,  MKL_INT *ib, float *c,  MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_scsradd(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, float *a, MKL_INT *ja, MKL_INT *ia,  const float *beta, float *b, MKL_INT *jb, MKL_INT *ib, float *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void __stdcall mkl_ccsrmultcsr(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, const MKL_INT *k, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib, MKL_Complex8 *c, MKL_INT *jc, MKL_INT *ic, const MKL_INT *nnzmax, MKL_INT *ierr);
MKL_DEPRECATED void __stdcall mkl_ccsrmultd(const char *transa, int transa_len,   const MKL_INT *m, const MKL_INT *n,  const MKL_INT *k, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib, MKL_Complex8 *c, MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_ccsradd(const char *transa, int transa_len,  const MKL_INT *job, const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, const MKL_Complex8 *beta, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex8 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void __stdcall mkl_zcsrmultcsr(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex16 *a, MKL_INT *ja,  MKL_INT *ia, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void __stdcall mkl_zcsrmultd(const char *transa, int transa_len,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *ldc);
MKL_DEPRECATED void __stdcall mkl_zcsradd(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia,  const MKL_Complex16 *beta, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);


/*Sparse BLAS Level2 (CSR-CSR or CSR-DNS) upper case */
MKL_DEPRECATED void __stdcall MKL_DCSRMULTCSR(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void __stdcall MKL_DCSRMULTD(const char *transa, int transa_len,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, double *a, MKL_INT *ja, MKL_INT *ia, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_DCSRADD(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, double *a, MKL_INT *ja, MKL_INT *ia, const double *beta, double *b, MKL_INT *jb, MKL_INT *ib,  double *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void __stdcall MKL_SCSRMULTCSR(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia, float *b, MKL_INT *jb, MKL_INT *ib,  float *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void __stdcall MKL_SCSRMULTD(const char *transa, int transa_len,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, float *a, MKL_INT *ja, MKL_INT *ia, float *b, MKL_INT *jb, MKL_INT *ib,  float *c,  MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_SCSRADD(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, float *a, MKL_INT *ja, MKL_INT *ia,  const float *beta, float *b, MKL_INT *jb, MKL_INT *ib,  float *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void __stdcall MKL_CCSRMULTCSR(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex8 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void __stdcall MKL_CCSRMULTD(const char *transa, int transa_len,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex8 *c,  MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_CCSRADD(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, MKL_Complex8 *a, MKL_INT *ja, MKL_INT *ia,  const MKL_Complex8 *beta, MKL_Complex8 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex8 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);

MKL_DEPRECATED void __stdcall MKL_ZCSRMULTCSR(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);
MKL_DEPRECATED void __stdcall MKL_ZCSRMULTD(const char *transa, int transa_len,   const MKL_INT *m,  const MKL_INT *n,  const MKL_INT *k, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *ldc);
MKL_DEPRECATED void __stdcall MKL_ZCSRADD(const char *transa, int transa_len,  const MKL_INT *job,  const MKL_INT *sort,  const MKL_INT *m,  const MKL_INT *n, MKL_Complex16 *a, MKL_INT *ja, MKL_INT *ia,  const MKL_Complex16 *beta, MKL_Complex16 *b, MKL_INT *jb, MKL_INT *ib,  MKL_Complex16 *c,  MKL_INT *jc,  MKL_INT *ic,  const MKL_INT *nnzmax,  MKL_INT *ierr);




/*****************************************************************************************/
/************** Basic types and constants for inspector-executor SpBLAS API **************/
/*****************************************************************************************/

    /* status of the routines */
    typedef enum
    {
        SPARSE_STATUS_SUCCESS           = 0,    /* the operation was successful */
        SPARSE_STATUS_NOT_INITIALIZED   = 1,    /* empty handle or matrix arrays */
        SPARSE_STATUS_ALLOC_FAILED      = 2,    /* internal error: memory allocation failed */
        SPARSE_STATUS_INVALID_VALUE     = 3,    /* invalid input value */
        SPARSE_STATUS_EXECUTION_FAILED  = 4,    /* e.g. 0-diagonal element for triangular solver, etc. */
        SPARSE_STATUS_INTERNAL_ERROR    = 5,    /* internal error */
        SPARSE_STATUS_NOT_SUPPORTED     = 6     /* e.g. operation for double precision doesn't support other types */
    } sparse_status_t;

    /* sparse matrix operations */
    typedef enum
    {
        SPARSE_OPERATION_NON_TRANSPOSE       = 10,
        SPARSE_OPERATION_TRANSPOSE           = 11,
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12
    } sparse_operation_t;

    /* supported matrix types */
    typedef enum
    {
        SPARSE_MATRIX_TYPE_GENERAL            = 20,   /*    General case                    */
        SPARSE_MATRIX_TYPE_SYMMETRIC          = 21,   /*    Triangular part of              */
        SPARSE_MATRIX_TYPE_HERMITIAN          = 22,   /*    the matrix is to be processed   */
        SPARSE_MATRIX_TYPE_TRIANGULAR         = 23,
        SPARSE_MATRIX_TYPE_DIAGONAL           = 24,   /* diagonal matrix; only diagonal elements will be processed */
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR   = 25,
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL     = 26    /* block-diagonal matrix; only diagonal blocks will be processed */
    } sparse_matrix_type_t;

    /* sparse matrix indexing: C-style or Fortran-style */
    typedef enum
    {
        SPARSE_INDEX_BASE_ZERO  = 0,           /* C-style */
        SPARSE_INDEX_BASE_ONE   = 1            /* Fortran-style */
    } sparse_index_base_t;

    /* applies to triangular matrices only ( SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_MATRIX_TYPE_HERMITIAN, SPARSE_MATRIX_TYPE_TRIANGULAR ) */
    typedef enum
    {
        SPARSE_FILL_MODE_LOWER  = 40,           /* lower triangular part of the matrix is stored */
        SPARSE_FILL_MODE_UPPER  = 41,            /* upper triangular part of the matrix is stored */
        SPARSE_FILL_MODE_FULL   = 42            /* upper triangular part of the matrix is stored */
    } sparse_fill_mode_t;

    /* applies to triangular matrices only ( SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_MATRIX_TYPE_HERMITIAN, SPARSE_MATRIX_TYPE_TRIANGULAR ) */
    typedef enum
    {
        SPARSE_DIAG_NON_UNIT    = 50,           /* triangular matrix with non-unit diagonal */
        SPARSE_DIAG_UNIT        = 51            /* triangular matrix with unit diagonal */
    } sparse_diag_type_t;

    /* applicable for Level 3 operations with dense matrices; describes storage scheme for dense matrix (row major or column major) */
    typedef enum
    {
        SPARSE_LAYOUT_ROW_MAJOR    = 101,       /* C-style */
        SPARSE_LAYOUT_COLUMN_MAJOR = 102        /* Fortran-style */
    } sparse_layout_t;

    /* verbose mode; if verbose mode activated, handle should collect and report profiling / optimization info */
    typedef enum
    {
        SPARSE_VERBOSE_OFF      = 70,
        SPARSE_VERBOSE_BASIC    = 71,           /* output contains high-level information about optimization algorithms, issues, etc. */
        SPARSE_VERBOSE_EXTENDED = 72            /* provide detailed output information */
    } verbose_mode_t;

    /* memory optimization hints from user: describe how much memory could be used on optimization stage */
    typedef enum
    {
        SPARSE_MEMORY_NONE          = 80,       /* no memory should be allocated for matrix values and structures; auxiliary structures could be created only for workload balancing, parallelization, etc. */
        SPARSE_MEMORY_AGGRESSIVE    = 81        /* matrix could be converted to any internal format */
    } sparse_memory_usage_t;

    typedef enum
    {
        SPARSE_STAGE_FULL_MULT            = 90,
        SPARSE_STAGE_NNZ_COUNT            = 91,
        SPARSE_STAGE_FINALIZE_MULT        = 92,
        SPARSE_STAGE_FULL_MULT_NO_VAL     = 93,
        SPARSE_STAGE_FINALIZE_MULT_NO_VAL = 94
    } sparse_request_t;

/*************************************************************************************************/
/*** Opaque structure for sparse matrix in internal format, further D - means double precision ***/
/*************************************************************************************************/

    struct  sparse_matrix;
    typedef struct sparse_matrix *sparse_matrix_t;

    /* descriptor of main sparse matrix properties */
    struct matrix_descr {
        sparse_matrix_type_t  type;       /* matrix type: general, diagonal or triangular / symmetric / hermitian */
        sparse_fill_mode_t    mode;       /* upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case) */
        sparse_diag_type_t    diag;       /* unit or non-unit diagonal ( for triangular / symmetric / hermitian case) */
    };

/*****************************************************************************************/
/*************************************** Creation routines *******************************/
/*****************************************************************************************/

/*
    Matrix handle is used for storing information about the matrix and matrix values

    Create matrix from one of the existing sparse formats by creating the handle with matrix info and copy matrix values if requested.
    Collect high-level info about the matrix. Need to use this interface for the case with several calls in program for performance reasons,
    where optimizations are not required.

    coordinate format,
    SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the mkl_sparse_order() or mkl_sparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
    sparse_status_t __stdcall
                    mkl_sparse_s_create_coo( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             nnz,
                                                   MKL_INT             *row_indx,
                                                   MKL_INT             *col_indx,
                                                   float               *values );

    sparse_status_t __stdcall
                    mkl_sparse_d_create_coo( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             nnz,
                                                   MKL_INT             *row_indx,
                                                   MKL_INT             *col_indx,
                                                   double              *values );

    sparse_status_t __stdcall
                    mkl_sparse_c_create_coo( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             nnz,
                                                   MKL_INT             *row_indx,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex8        *values );

    sparse_status_t __stdcall
                    mkl_sparse_z_create_coo( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             nnz,
                                                   MKL_INT             *row_indx,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex16       *values );


/*
    compressed sparse row format (4-arrays version),
    SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the mkl_sparse_order() or mkl_sparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
    sparse_status_t __stdcall
                    mkl_sparse_s_create_csr( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   float               *values );

    sparse_status_t __stdcall
                    mkl_sparse_d_create_csr( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   double              *values );

    sparse_status_t __stdcall
                    mkl_sparse_c_create_csr( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex8        *values );

    sparse_status_t __stdcall
                    mkl_sparse_z_create_csr( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex16       *values );

/*
    compressed sparse column format (4-arrays version),
    SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the mkl_sparse_order() or mkl_sparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
    sparse_status_t __stdcall
                    mkl_sparse_s_create_csc( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *cols_start,
                                                   MKL_INT             *cols_end,
                                                   MKL_INT             *row_indx,
                                                   float               *values );

    sparse_status_t __stdcall
                    mkl_sparse_d_create_csc( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *cols_start,
                                                   MKL_INT             *cols_end,
                                                   MKL_INT             *row_indx,
                                                   double              *values );

    sparse_status_t __stdcall
                    mkl_sparse_c_create_csc( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *cols_start,
                                                   MKL_INT             *cols_end,
                                                   MKL_INT             *row_indx,
                                                   MKL_Complex8        *values );

    sparse_status_t __stdcall
                    mkl_sparse_z_create_csc( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *cols_start,
                                                   MKL_INT             *cols_end,
                                                   MKL_INT             *row_indx,
                                                   MKL_Complex16       *values );

/*
    compressed block sparse row format (4-arrays version, square blocks),
    SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the mkl_sparse_order() or mkl_sparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
    sparse_status_t __stdcall
                    mkl_sparse_s_create_bsr( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing,       /* indexing: C-style or Fortran-style */
                                             const sparse_layout_t     block_layout,   /* block storage: row-major or column-major */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             block_size,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   float               *values );

    sparse_status_t __stdcall
                    mkl_sparse_d_create_bsr( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing,       /* indexing: C-style or Fortran-style */
                                             const sparse_layout_t     block_layout,   /* block storage: row-major or column-major */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             block_size,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   double              *values );

    sparse_status_t __stdcall
                    mkl_sparse_c_create_bsr( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing,       /* indexing: C-style or Fortran-style */
                                             const sparse_layout_t     block_layout,   /* block storage: row-major or column-major */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             block_size,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex8        *values );

    sparse_status_t __stdcall
                    mkl_sparse_z_create_bsr( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing,       /* indexing: C-style or Fortran-style */
                                             const sparse_layout_t     block_layout,   /* block storage: row-major or column-major */
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                             const MKL_INT             block_size,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex16       *values );

/*
    Create copy of the existing handle; matrix properties could be changed.
    For example it could be used for extracting triangular or diagonal parts from existing matrix.
*/
    sparse_status_t __stdcall
                    mkl_sparse_copy( const sparse_matrix_t     source,
                                     const struct matrix_descr descr,        /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     sparse_matrix_t           *dest );

/*
    destroy matrix handle; if sparse matrix was stored inside the handle it also deallocates the matrix
    It is user's responsibility not to delete the handle with the matrix, if this matrix is shared with other handles
*/
    sparse_status_t __stdcall
                    mkl_sparse_destroy( sparse_matrix_t  A );
/*
    return extended error information from last operation;
    e.g. info about wrong input parameter, memory sizes that couldn't be allocated
*/
    sparse_status_t __stdcall
                    mkl_sparse_get_error_info( sparse_matrix_t  A, MKL_INT *info ); /* unsupported currently */


/*****************************************************************************************/
/************************ Converters of internal representation  *************************/
/*****************************************************************************************/

    /* converters from current format to another */
    sparse_status_t __stdcall
                    mkl_sparse_convert_csr ( const sparse_matrix_t    source,         /* convert original matrix to CSR representation */
                                             const sparse_operation_t operation,      /* as is, transposed or conjugate transposed */
                                             sparse_matrix_t          *dest );

    sparse_status_t __stdcall
                    mkl_sparse_convert_bsr ( const sparse_matrix_t    source,         /* convert original matrix to BSR representation */
                                             const MKL_INT            block_size,
                                             const sparse_layout_t    block_layout,   /* block storage: row-major or column-major */
                                             const sparse_operation_t operation,      /* as is, transposed or conjugate transposed */
                                             sparse_matrix_t          *dest );

    sparse_status_t __stdcall
                    mkl_sparse_s_export_bsr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             sparse_layout_t        *block_layout,  /* block storage: row-major or column-major */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                *block_size,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             float                  **values );

    sparse_status_t __stdcall
                    mkl_sparse_d_export_bsr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             sparse_layout_t        *block_layout,  /* block storage: row-major or column-major */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                *block_size,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             double                 **values );

    sparse_status_t __stdcall
                    mkl_sparse_c_export_bsr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             sparse_layout_t        *block_layout,  /* block storage: row-major or column-major */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                *block_size,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             MKL_Complex8           **values );

    sparse_status_t __stdcall
                    mkl_sparse_z_export_bsr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             sparse_layout_t        *block_layout,  /* block storage: row-major or column-major */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                *block_size,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             MKL_Complex16          **values );

    sparse_status_t __stdcall
                    mkl_sparse_s_export_csr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             float                  **values );

    sparse_status_t __stdcall
                    mkl_sparse_d_export_csr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             double                 **values );

    sparse_status_t __stdcall
                    mkl_sparse_c_export_csr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             MKL_Complex8           **values );

    sparse_status_t __stdcall
                    mkl_sparse_z_export_csr( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **rows_start,
                                             MKL_INT                **rows_end,
                                             MKL_INT                **col_indx,
                                             MKL_Complex16          **values );

    sparse_status_t __stdcall
                    mkl_sparse_s_export_csc( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **cols_start,
                                             MKL_INT                **cols_end,
                                             MKL_INT                **row_indx,
                                             float                  **values );

    sparse_status_t __stdcall
                    mkl_sparse_d_export_csc( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **cols_start,
                                             MKL_INT                **cols_end,
                                             MKL_INT                **row_indx,
                                             double                 **values );

    sparse_status_t __stdcall
                    mkl_sparse_c_export_csc( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **cols_start,
                                             MKL_INT                **cols_end,
                                             MKL_INT                **row_indx,
                                             MKL_Complex8           **values );

    sparse_status_t __stdcall
                    mkl_sparse_z_export_csc( const sparse_matrix_t  source,
                                             sparse_index_base_t    *indexing,      /* indexing: C-style or Fortran-style */
                                             MKL_INT                *rows,
                                             MKL_INT                *cols,
                                             MKL_INT                **cols_start,
                                             MKL_INT                **cols_end,
                                             MKL_INT                **row_indx,
                                             MKL_Complex16          **values );


/*****************************************************************************************/
/************************** Step-by-step modification routines ***************************/
/*****************************************************************************************/


    /* update existing value in the matrix ( for internal storage only, should not work with user-allocated matrices) */
    sparse_status_t __stdcall
                    mkl_sparse_s_set_value( const sparse_matrix_t A,
                                            const MKL_INT         row,
                                            const MKL_INT         col,
                                            const float           value );

    sparse_status_t __stdcall
                    mkl_sparse_d_set_value( const sparse_matrix_t A,
                                            const MKL_INT         row,
                                            const MKL_INT         col,
                                            const double          value );

    sparse_status_t __stdcall
                    mkl_sparse_c_set_value( const sparse_matrix_t A,
                                            const MKL_INT         row,
                                            const MKL_INT         col,
                                            const MKL_Complex8    value );

    sparse_status_t __stdcall
                    mkl_sparse_z_set_value( const sparse_matrix_t A,
                                            const MKL_INT         row,
                                            const MKL_INT         col,
                                            const MKL_Complex16   value );

/*****************************************************************************************/
/****************************** Verbose mode routine *************************************/
/*****************************************************************************************/

    /* allow to switch on/off verbose mode */
    sparse_status_t __stdcall
                    mkl_sparse_set_verbose_mode ( verbose_mode_t verbose ); /* unsupported currently */

/*****************************************************************************************/
/****************************** Optimization routines ************************************/
/*****************************************************************************************/

    /* Describe expected operations with amount of iterations */
    sparse_status_t __stdcall
                    mkl_sparse_set_mv_hint    ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,  /* SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const MKL_INT             expected_calls );

    sparse_status_t __stdcall
                    mkl_sparse_set_dotmv_hint ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation, /* SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                                const struct matrix_descr descr,     /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const MKL_INT             expectedCalls );

    sparse_status_t __stdcall
                    mkl_sparse_set_mm_hint    ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                                const MKL_INT             dense_matrix_size, /* amount of columns in dense matrix */
                                                const MKL_INT             expected_calls );

    sparse_status_t __stdcall
                    mkl_sparse_set_sv_hint    ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,  /* SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const MKL_INT             expected_calls );

    sparse_status_t __stdcall
                    mkl_sparse_set_sm_hint    ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                                const MKL_INT             dense_matrix_size, /* amount of columns in dense matrix */
                                                const MKL_INT             expected_calls );

    sparse_status_t __stdcall
                    mkl_sparse_set_symgs_hint ( const sparse_matrix_t     A,
                                                const sparse_operation_t  operation,  /* SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                                const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                                const MKL_INT             expected_calls );


    /* Describe memory usage model */
    sparse_status_t __stdcall
                    mkl_sparse_set_memory_hint ( const sparse_matrix_t       A,
                                                 const sparse_memory_usage_t policy );    /* SPARSE_MEMORY_AGGRESSIVE is default value */

/*
    Optimize matrix described by the handle. It uses hints (optimization and memory) that should be set up before this call.
    If hints were not explicitly defined, default vales are:
    SPARSE_OPERATION_NON_TRANSPOSE for matrix-vector multiply with infinite number of expected iterations.
*/
    sparse_status_t __stdcall
                    mkl_sparse_optimize ( sparse_matrix_t  A );

/*****************************************************************************************/
/****************************** Computational routines ***********************************/
/*****************************************************************************************/

    sparse_status_t __stdcall
                    mkl_sparse_order( const sparse_matrix_t A );

/*
    Perform computations based on created matrix handle

    Level 2
*/
    /*   Computes y = alpha * A * x + beta * y   */
    sparse_status_t __stdcall
                    mkl_sparse_s_mv ( const sparse_operation_t  operation,
                                      const float               alpha,
                                      const sparse_matrix_t     A,
                                      const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                      const float               *x,
                                      const float               beta,
                                      float                     *y );

    sparse_status_t __stdcall
                    mkl_sparse_d_mv ( const sparse_operation_t  operation,
                                      const double              alpha,
                                      const sparse_matrix_t     A,
                                      const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                      const double              *x,
                                      const double              beta,
                                      double                    *y );

    sparse_status_t __stdcall
                    mkl_sparse_c_mv ( const sparse_operation_t  operation,
                                      const MKL_Complex8        alpha,
                                      const sparse_matrix_t     A,
                                      const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                      const MKL_Complex8        *x,
                                      const MKL_Complex8        beta,
                                      MKL_Complex8              *y );

    sparse_status_t __stdcall
                    mkl_sparse_z_mv ( const sparse_operation_t  operation,
                                      const MKL_Complex16       alpha,
                                      const sparse_matrix_t     A,
                                      const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                      const MKL_Complex16       *x,
                                      const MKL_Complex16       beta,
                                      MKL_Complex16             *y );

    /*    Computes y = alpha * A * x + beta * y  and d = <x, y> , the l2 inner product */ 
    sparse_status_t __stdcall
                    mkl_sparse_s_dotmv( const sparse_operation_t  transA,
                                        const float               alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const float               *x,
                                        const float               beta,
                                        float                     *y,
                                        float                     *d);

    sparse_status_t __stdcall
                    mkl_sparse_d_dotmv( const sparse_operation_t  transA,
                                        const double              alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const double              *x,
                                        const double              beta,
                                        double                    *y,
                                        double                    *d);

    sparse_status_t __stdcall
                    mkl_sparse_c_dotmv( const sparse_operation_t  transA,
                                        const MKL_Complex8        alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const MKL_Complex8        *x,
                                        const MKL_Complex8        beta,
                                        MKL_Complex8              *y,
                                        MKL_Complex8              *d);

    sparse_status_t __stdcall
                    mkl_sparse_z_dotmv( const sparse_operation_t  transA,
                                        const MKL_Complex16       alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const MKL_Complex16       *x,
                                        const MKL_Complex16       beta,
                                        MKL_Complex16             *y,
                                        MKL_Complex16             *d);


    /*   Solves triangular system y = alpha * A^{-1} * x   */
    sparse_status_t __stdcall
                    mkl_sparse_s_trsv ( const sparse_operation_t  operation,
                                        const float               alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const float               *x,
                                        float                     *y );

    sparse_status_t __stdcall
                    mkl_sparse_d_trsv ( const sparse_operation_t  operation,
                                        const double              alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const double              *x,
                                        double                    *y );

    sparse_status_t __stdcall
                    mkl_sparse_c_trsv ( const sparse_operation_t  operation,
                                        const MKL_Complex8        alpha,
                                        const sparse_matrix_t    A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const MKL_Complex8        *x,
                                        MKL_Complex8              *y );

    sparse_status_t __stdcall
                    mkl_sparse_z_trsv ( const sparse_operation_t  operation,
                                        const MKL_Complex16       alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const MKL_Complex16       *x,
                                        MKL_Complex16             *y );

    /*   Applies symmetric Gauss-Seidel preconditioner to symmetric system A * x = b, */
    /*   that is, it solves:                                                          */
    /*      x0       = alpha*x                                                        */
    /*      (L+D)*x1 = b - U*x0                                                       */
    /*      (D+U)*x  = b - L*x1                                                       */
    /*                                                                                */
    /*   SYMGS_MV also returns y = A*x                                                */
    sparse_status_t __stdcall
                    mkl_sparse_s_symgs ( const sparse_operation_t  op,
                                         const sparse_matrix_t     A,
                                         const struct matrix_descr descr,
                                         const float               alpha,
                                         const float               *b,
                                         float                     *x);

    sparse_status_t __stdcall
                    mkl_sparse_d_symgs ( const sparse_operation_t  op,
                                         const sparse_matrix_t     A,
                                         const struct matrix_descr descr,
                                         const double              alpha,
                                         const double              *b,
                                         double                    *x);

    sparse_status_t __stdcall
                    mkl_sparse_c_symgs ( const sparse_operation_t  op,
                                         const sparse_matrix_t     A,
                                         const struct matrix_descr descr,
                                         const MKL_Complex8        alpha,
                                         const MKL_Complex8        *b,
                                         MKL_Complex8              *x);
   
    sparse_status_t __stdcall
                    mkl_sparse_z_symgs ( const sparse_operation_t  op,
                                         const sparse_matrix_t     A,
                                         const struct matrix_descr descr,
                                         const MKL_Complex16       alpha,
                                         const MKL_Complex16       *b,
                                         MKL_Complex16             *x);
    
    sparse_status_t __stdcall
                    mkl_sparse_s_symgs_mv ( const sparse_operation_t  op,
                                            const sparse_matrix_t     A,
                                            const struct matrix_descr descr,
                                            const float               alpha,
                                            const float               *b,
                                            float                     *x,
                                            float                     *y);
 
    sparse_status_t __stdcall
                    mkl_sparse_d_symgs_mv ( const sparse_operation_t  op,
                                            const sparse_matrix_t     A,
                                            const struct matrix_descr descr,
                                            const double              alpha,
                                            const double              *b,
                                            double                    *x,
                                            double                    *y);

    sparse_status_t __stdcall
                    mkl_sparse_c_symgs_mv ( const sparse_operation_t  op,
                                            const sparse_matrix_t     A,
                                            const struct matrix_descr descr,
                                            const MKL_Complex8        alpha,
                                            const MKL_Complex8        *b,
                                            MKL_Complex8              *x,
                                            MKL_Complex8              *y);
    
    sparse_status_t __stdcall
                    mkl_sparse_z_symgs_mv ( const sparse_operation_t  op,
                                            const sparse_matrix_t     A,
                                            const struct matrix_descr descr,
                                            const MKL_Complex16       alpha,
                                            const MKL_Complex16       *b,
                                            MKL_Complex16             *x,
                                            MKL_Complex16             *y);


    /* Level 3 */

    /*   Computes y = alpha * A * x + beta * y   */
    sparse_status_t __stdcall
                    mkl_sparse_s_mm( const sparse_operation_t  operation,
                                     const float               alpha,
                                     const sparse_matrix_t     A,
                                     const struct matrix_descr descr,          /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     const sparse_layout_t     layout,         /* storage scheme for the dense matrix: C-style or Fortran-style */
                                     const float               *x,
                                     const MKL_INT             columns,
                                     const MKL_INT             ldx,
                                     const float               beta,
                                     float                     *y,
                                     const MKL_INT             ldy );

    sparse_status_t __stdcall
                    mkl_sparse_d_mm( const sparse_operation_t  operation,
                                     const double              alpha,
                                     const sparse_matrix_t     A,
                                     const struct matrix_descr descr,          /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     const sparse_layout_t     layout,         /* storage scheme for the dense matrix: C-style or Fortran-style */
                                     const double              *x,
                                     const MKL_INT             columns,
                                     const MKL_INT             ldx,
                                     const double              beta,
                                     double                    *y,
                                     const MKL_INT             ldy );

    sparse_status_t __stdcall
                    mkl_sparse_c_mm( const sparse_operation_t  operation,
                                     const MKL_Complex8        alpha,
                                     const sparse_matrix_t     A,
                                     const struct matrix_descr descr,          /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     const sparse_layout_t     layout,         /* storage scheme for the dense matrix: C-style or Fortran-style */
                                     const MKL_Complex8        *x,
                                     const MKL_INT             columns,
                                     const MKL_INT             ldx,
                                     const MKL_Complex8        beta,
                                     MKL_Complex8              *y,
                                     const MKL_INT             ldy );

    sparse_status_t __stdcall
                    mkl_sparse_z_mm( const sparse_operation_t  operation,
                                     const MKL_Complex16       alpha,
                                     const sparse_matrix_t     A,
                                     const struct matrix_descr descr,          /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                     const sparse_layout_t     layout,         /* storage scheme for the dense matrix: C-style or Fortran-style */
                                     const MKL_Complex16       *x,
                                     const MKL_INT             columns,
                                     const MKL_INT             ldx,
                                     const MKL_Complex16       beta,
                                     MKL_Complex16             *y,
                                     const MKL_INT             ldy );

    /*   Solves triangular system y = alpha * A^{-1} * x   */
    sparse_status_t __stdcall
                    mkl_sparse_s_trsm ( const sparse_operation_t  operation,
                                        const float               alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                        const float               *x,
                                        const MKL_INT             columns,
                                        const MKL_INT             ldx,
                                        float                     *y,
                                        const MKL_INT             ldy );

    sparse_status_t __stdcall
                    mkl_sparse_d_trsm ( const sparse_operation_t  operation,
                                        const double              alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                        const double              *x,
                                        const MKL_INT             columns,
                                        const MKL_INT             ldx,
                                        double                    *y,
                                        const MKL_INT             ldy );

    sparse_status_t __stdcall
                    mkl_sparse_c_trsm ( const sparse_operation_t  operation,
                                        const MKL_Complex8        alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                        const MKL_Complex8        *x,
                                        const MKL_INT             columns,
                                        const MKL_INT             ldx,
                                        MKL_Complex8              *y,
                                        const MKL_INT             ldy );

    sparse_status_t __stdcall
                    mkl_sparse_z_trsm ( const sparse_operation_t  operation,
                                        const MKL_Complex16       alpha,
                                        const sparse_matrix_t     A,
                                        const struct matrix_descr descr,      /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
                                        const sparse_layout_t     layout,     /* storage scheme for the dense matrix: C-style or Fortran-style */
                                        const MKL_Complex16       *x,
                                        const MKL_INT             columns,
                                        const MKL_INT             ldx,
                                        MKL_Complex16             *y,
                                        const MKL_INT             ldy );

    /* Sparse-sparse functionality */


    /*   Computes sum of sparse matrices: C = alpha * op(A) + B, result is sparse   */
    sparse_status_t __stdcall
                    mkl_sparse_s_add( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const float              alpha,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    sparse_status_t __stdcall
                    mkl_sparse_d_add( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const double             alpha,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    sparse_status_t __stdcall
                    mkl_sparse_c_add( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const MKL_Complex8       alpha,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    sparse_status_t __stdcall
                    mkl_sparse_z_add( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const MKL_Complex16      alpha,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    /*   Computes product of sparse matrices: C = op(A) * B, result is sparse   */
    sparse_status_t __stdcall
                    mkl_sparse_spmm ( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      const sparse_matrix_t    B,
                                      sparse_matrix_t          *C );

    /*   Computes product of sparse matrices: C = opA(A) * opB(B), result is sparse   */
    sparse_status_t __stdcall
                    mkl_sparse_sp2m ( const sparse_operation_t  transA, 
                                      const struct matrix_descr descrA, 
                                      const sparse_matrix_t     A,
                                      const sparse_operation_t  transB, 
                                      const struct matrix_descr descrB, 
                                      const sparse_matrix_t     B,
                                      const sparse_request_t    request, 
                                      sparse_matrix_t           *C );

    /*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is sparse   */
    sparse_status_t __stdcall
                    mkl_sparse_syrk ( const sparse_operation_t operation,
                                      const sparse_matrix_t    A,
                                      sparse_matrix_t          *C );

    
    /*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is sparse   */
    sparse_status_t __stdcall
                    mkl_sparse_sypr ( const sparse_operation_t  transA,
                                      const sparse_matrix_t     A,
                                      const sparse_matrix_t     B,
                                      const struct matrix_descr descrB,
                                      sparse_matrix_t           *C,
                                      const sparse_request_t    request );

    /*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is dense */
    sparse_status_t __stdcall
                    mkl_sparse_s_syprd ( const sparse_operation_t op,
                                         const sparse_matrix_t    A,
                                         const float              *B,
                                         const sparse_layout_t    layoutB,
                                         const MKL_INT            ldb,
                                         const float              alpha,
                                         const float              beta,
                                         float                    *C,
                                         const sparse_layout_t    layoutC,
                                         const MKL_INT            ldc );

    sparse_status_t __stdcall
                    mkl_sparse_d_syprd ( const sparse_operation_t op,
                                         const sparse_matrix_t    A,
                                         const double             *B,
                                         const sparse_layout_t    layoutB,
                                         const MKL_INT            ldb,
                                         const double             alpha,
                                         const double             beta,
                                         double                   *C,
                                         const sparse_layout_t    layoutC,
                                         const MKL_INT            ldc );

    sparse_status_t __stdcall
                    mkl_sparse_c_syprd ( const sparse_operation_t op,
                                         const sparse_matrix_t    A,
                                         const MKL_Complex8       *B,
                                         const sparse_layout_t    layoutB,
                                         const MKL_INT            ldb,
                                         const MKL_Complex8       alpha,
                                         const MKL_Complex8       beta,
                                         MKL_Complex8             *C,
                                         const sparse_layout_t    layoutC,
                                         const MKL_INT            ldc );

    sparse_status_t __stdcall
                    mkl_sparse_z_syprd ( const sparse_operation_t op,
                                         const sparse_matrix_t    A,
                                         const MKL_Complex16      *B,
                                         const sparse_layout_t    layoutB,
                                         const MKL_INT            ldb,
                                         const MKL_Complex16      alpha,
                                         const MKL_Complex16      beta,
                                         MKL_Complex16            *C,
                                         const sparse_layout_t    layoutC,
                                         const MKL_INT            ldc );


    /*   Computes product of sparse matrices: C = op(A) * B, result is dense   */
    sparse_status_t __stdcall
                    mkl_sparse_s_spmmd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const sparse_matrix_t    B,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        float                    *C,
                                        const MKL_INT            ldc );

    sparse_status_t __stdcall
                    mkl_sparse_d_spmmd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const sparse_matrix_t    B,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        double                   *C,
                                        const MKL_INT            ldc );

    sparse_status_t __stdcall
                    mkl_sparse_c_spmmd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const sparse_matrix_t    B,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        MKL_Complex8             *C,
                                        const MKL_INT            ldc );

    sparse_status_t __stdcall
                    mkl_sparse_z_spmmd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const sparse_matrix_t    B,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        MKL_Complex16            *C,
                                        const MKL_INT            ldc );

    /*   Computes product of sparse matrices: C = opA(A) * opB(B), result is dense*/
    sparse_status_t __stdcall
                    mkl_sparse_s_sp2md ( const sparse_operation_t  transA,
                                         const struct matrix_descr descrA,
                                         const sparse_matrix_t     A,
                                         const sparse_operation_t  transB,
                                         const struct matrix_descr descrB,
                                         const sparse_matrix_t     B,
                                         const float               alpha,
                                         const float               beta,
                                         float                     *C,
                                         const sparse_layout_t     layout,
                                         const MKL_INT             ldc );

    sparse_status_t __stdcall
                    mkl_sparse_d_sp2md ( const sparse_operation_t  transA,
                                         const struct matrix_descr descrA,
                                         const sparse_matrix_t     A,
                                         const sparse_operation_t  transB,
                                         const struct matrix_descr descrB,
                                         const sparse_matrix_t     B,
                                         const double              alpha,
                                         const double              beta,
                                         double                    *C,
                                         const sparse_layout_t     layout,
                                         const MKL_INT             ldc );

    sparse_status_t __stdcall
                    mkl_sparse_c_sp2md ( const sparse_operation_t  transA,
                                         const struct matrix_descr descrA,
                                         const sparse_matrix_t     A,
                                         const sparse_operation_t  transB,
                                         const struct matrix_descr descrB,
                                         const sparse_matrix_t     B,
                                         const MKL_Complex8        alpha,
                                         const MKL_Complex8        beta,
                                         MKL_Complex8              *C,
                                         const sparse_layout_t     layout,
                                         const MKL_INT             ldc );

    sparse_status_t __stdcall
                    mkl_sparse_z_sp2md ( const sparse_operation_t  transA,
                                         const struct matrix_descr descrA,
                                         const sparse_matrix_t     A,
                                         const sparse_operation_t  transB,
                                         const struct matrix_descr descrB,
                                         const sparse_matrix_t     B,
                                         const MKL_Complex16       alpha,
                                         const MKL_Complex16       beta,
                                         MKL_Complex16             *C,
                                         const sparse_layout_t     layout,
                                         const MKL_INT             ldc );

    /*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is dense */
    sparse_status_t __stdcall
                    mkl_sparse_s_syrkd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const float              alpha,
                                        const float              beta,
                                        float                    *C,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        const MKL_INT            ldc );

    sparse_status_t __stdcall
                    mkl_sparse_d_syrkd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const double             alpha,
                                        const double             beta,
                                        double                   *C,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        const MKL_INT            ldc );

    sparse_status_t __stdcall
                    mkl_sparse_c_syrkd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const MKL_Complex8       alpha,
                                        const MKL_Complex8       beta,
                                        MKL_Complex8             *C,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        const MKL_INT            ldc );

    sparse_status_t __stdcall
                    mkl_sparse_z_syrkd( const sparse_operation_t operation,
                                        const sparse_matrix_t    A,
                                        const MKL_Complex16      alpha,
                                        const MKL_Complex16      beta,
                                        MKL_Complex16            *C,
                                        const sparse_layout_t    layout,       /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                        const MKL_INT            ldc );


#endif /* MKL_STDCALL */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_SPBLAS_STDCALL_H_ */
