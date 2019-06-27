/*******************************************************************************
* Copyright 2007-2019 Intel Corporation.
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
!      Intel(R) Math Kernel Library (Intel(R) MKL) include for transposition routines
!******************************************************************************/

#if !defined(_MKL_TRANS_H)
#define _MKL_TRANS_H

/* for size_t */
#include <stddef.h>
#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* In-place transposition routines */

#define mkl_simatcopy MKL_Simatcopy
void MKL_Simatcopy(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const float alpha,
    float * AB, size_t lda, size_t ldb);

#define mkl_dimatcopy MKL_Dimatcopy
void MKL_Dimatcopy(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const double alpha,
    double * AB, size_t lda, size_t ldb);

#define mkl_cimatcopy MKL_Cimatcopy
void MKL_Cimatcopy(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    MKL_Complex8 * AB, size_t lda, size_t ldb);

#define mkl_zimatcopy MKL_Zimatcopy
void MKL_Zimatcopy(
    const char ordering, const char trans,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    MKL_Complex16 * AB, size_t lda, size_t ldb);

/* Out-of-place transposition routines */

#define mkl_somatcopy MKL_Somatcopy
void MKL_Somatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const float alpha,
    const float * A, size_t lda,
    float * B, size_t ldb);

#define mkl_domatcopy MKL_Domatcopy
void MKL_Domatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const double alpha,
    const double * A, size_t lda,
    double * B, size_t ldb);

#define mkl_comatcopy MKL_Comatcopy
void MKL_Comatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    const MKL_Complex8 * A, size_t lda,
    MKL_Complex8 * B, size_t ldb);

#define mkl_zomatcopy MKL_Zomatcopy
void MKL_Zomatcopy(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    const MKL_Complex16 * A, size_t lda,
    MKL_Complex16 * B, size_t ldb);

/* Out-of-place transposition routines (all-strided case) */

#define mkl_somatcopy2 MKL_Somatcopy2
void MKL_Somatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const float alpha,
    const float * A, size_t lda, size_t stridea,
    float * B, size_t ldb, size_t strideb);

#define mkl_domatcopy2 MKL_Domatcopy2
void MKL_Domatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const double alpha,
    const double * A, size_t lda, size_t stridea,
    double * B, size_t ldb, size_t strideb);

#define mkl_comatcopy2 MKL_Comatcopy2
void MKL_Comatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    const MKL_Complex8 * A, size_t lda, size_t stridea,
    MKL_Complex8 * B, size_t ldb, size_t strideb);

#define mkl_zomatcopy2 MKL_Zomatcopy2
void MKL_Zomatcopy2(
    char ordering, char trans,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    const MKL_Complex16 * A, size_t lda, size_t stridea,
    MKL_Complex16 * B, size_t ldb, size_t strideb);

/* Out-of-place memory movement routines */

#define mkl_somatadd MKL_Somatadd
void MKL_Somatadd(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const float alpha,
    const float * A, size_t lda,
    const float beta,
    const float * B, size_t ldb,
    float * C, size_t ldc);

#define mkl_domatadd MKL_Domatadd
void MKL_Domatadd(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const double alpha,
    const double * A, size_t lda,
    const double beta,
    const double * B, size_t ldb,
    double * C, size_t ldc);

#define mkl_comatadd MKL_Comatadd
void MKL_Comatadd(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex8 alpha,
    const MKL_Complex8 * A, size_t lda,
    const MKL_Complex8 beta,
    const MKL_Complex8 * B, size_t ldb,
    MKL_Complex8 * C, size_t ldc);

#define mkl_zomatadd MKL_Zomatadd
void MKL_Zomatadd(
    char ordering, char transa, char transb,
    size_t rows, size_t cols,
    const MKL_Complex16 alpha,
    const MKL_Complex16 * A, size_t lda,
    const MKL_Complex16 beta,
    const MKL_Complex16 * B, size_t ldb,
    MKL_Complex16 * C, size_t ldc);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_TRANS_H */
