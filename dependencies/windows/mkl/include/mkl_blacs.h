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
!      Intel(R) Math Kernel Library (Intel(R) MKL) interface for BLACS routines
!******************************************************************************/

#ifndef _MKL_BLACS_H_
#define _MKL_BLACS_H_

#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* <name>_ declarations */

void    igamx2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    igamx2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    IGAMX2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    IGAMX2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);

void    sgamx2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    sgamx2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    SGAMX2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    SGAMX2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);

void    dgamx2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    dgamx2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    DGAMX2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    DGAMX2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);

void    cgamx2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    cgamx2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    CGAMX2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    CGAMX2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);

void    zgamx2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    zgamx2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZGAMX2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZGAMX2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);


void    igamn2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    igamn2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    IGAMN2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    IGAMN2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);

void    sgamn2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    sgamn2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    SGAMN2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    SGAMN2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);

void    dgamn2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    dgamn2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    DGAMN2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    DGAMN2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);

void    cgamn2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    cgamn2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    CGAMN2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    CGAMN2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);

void    zgamn2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    zgamn2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZGAMN2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZGAMN2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, MKL_INT *rA, MKL_INT *cA, const MKL_INT *ldia, const MKL_INT *rdest, const MKL_INT *cdest);

void    igsum2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    igsum2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    IGSUM2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    IGSUM2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    sgsum2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    sgsum2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    SGSUM2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    SGSUM2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    dgsum2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    dgsum2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    DGSUM2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    DGSUM2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    cgsum2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    cgsum2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    CGSUM2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    CGSUM2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    zgsum2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    zgsum2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZGSUM2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZGSUM2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);


void    igesd2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    igesd2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    IGESD2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    IGESD2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    sgesd2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    sgesd2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    SGESD2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    SGESD2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    dgesd2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    dgesd2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    DGESD2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    DGESD2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    cgesd2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    cgesd2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    CGESD2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    CGESD2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    zgesd2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    zgesd2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZGESD2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZGESD2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    itrsd2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    itrsd2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ITRSD2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ITRSD2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    strsd2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    strsd2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    STRSD2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    STRSD2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    dtrsd2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    dtrsd2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    DTRSD2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    DTRSD2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    ctrsd2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ctrsd2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    CTRSD2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    CTRSD2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    ztrsd2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ztrsd2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZTRSD2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);
void    ZTRSD2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda, const MKL_INT *rdest, const MKL_INT *cdest);

void    igerv2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    igerv2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    IGERV2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    IGERV2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    sgerv2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    sgerv2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    SGERV2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    SGERV2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    dgerv2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    dgerv2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    DGERV2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    DGERV2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    cgerv2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    cgerv2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    CGERV2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    CGERV2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    zgerv2d(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    zgerv2d_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ZGERV2D(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ZGERV2D_(const MKL_INT *ConTxt, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);


void    itrrv2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    itrrv2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ITRRV2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ITRRV2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    strrv2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    strrv2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    STRRV2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    STRRV2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    dtrrv2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    dtrrv2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    DTRRV2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    DTRRV2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    ctrrv2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ctrrv2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    CTRRV2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    CTRRV2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    ztrrv2d(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ztrrv2d_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ZTRRV2D(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ZTRRV2D_(const MKL_INT *ConTxt, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    igebs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda);
void    igebs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda);
void    IGEBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda);
void    IGEBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda);

void    sgebs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    sgebs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    SGEBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    SGEBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);

void    dgebs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    dgebs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    DGEBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    DGEBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);

void    cgebs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    cgebs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    CGEBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    CGEBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);

void    zgebs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    zgebs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    ZGEBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    ZGEBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);


void    itrbs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda);
void    itrbs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda);
void    ITRBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda);
void    ITRBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_INT *A, const MKL_INT *lda);

void    strbs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    strbs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    STRBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    STRBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);

void    dtrbs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    dtrbs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    DTRBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    DTRBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);

void    ctrbs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    ctrbs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    CTRBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);
void    CTRBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *A, const MKL_INT *lda);

void    ztrbs2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    ztrbs2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    ZTRBS2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);
void    ZTRBS2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *A, const MKL_INT *lda);


void    igebr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    igebr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    IGEBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    IGEBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    sgebr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    sgebr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    SGEBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    SGEBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    dgebr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    dgebr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    DGEBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    DGEBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    cgebr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    cgebr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    CGEBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    CGEBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    zgebr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    zgebr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ZGEBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ZGEBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    itrbr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    itrbr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ITRBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ITRBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, MKL_INT *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    strbr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    strbr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    STRBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    STRBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    dtrbr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    dtrbr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    DTRBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    DTRBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    ctrbr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ctrbr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    CTRBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    CTRBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, float *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);

void    ztrbr2d(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ztrbr2d_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ZTRBR2D(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);
void    ZTRBR2D_(const MKL_INT *ConTxt, const char *scope, const char *top, const char *uplo, const char *diag, const MKL_INT *m, const MKL_INT *n, double *A, const MKL_INT *lda, const MKL_INT *rsrc, const MKL_INT *csrc);


void    blacs_pinfo(MKL_INT *mypnum, MKL_INT *nprocs);
void    blacs_pinfo_(MKL_INT *mypnum, MKL_INT *nprocs);
void    BLACS_PINFO(MKL_INT *mypnum, MKL_INT *nprocs);
void    BLACS_PINFO_(MKL_INT *mypnum, MKL_INT *nprocs);

void    blacs_setup(MKL_INT *mypnum, MKL_INT *nprocs);
void    blacs_setup_(MKL_INT *mypnum, MKL_INT *nprocs);
void    BLACS_SETUP(MKL_INT *mypnum, MKL_INT *nprocs);
void    BLACS_SETUP_(MKL_INT *mypnum, MKL_INT *nprocs);

void    blacs_get(const MKL_INT *ConTxt, const MKL_INT *what, MKL_INT *val);
void    blacs_get_(const MKL_INT *ConTxt, const MKL_INT *what, MKL_INT *val);
void    BLACS_GET(const MKL_INT *ConTxt, const MKL_INT *what, MKL_INT *val);
void    BLACS_GET_(const MKL_INT *ConTxt, const MKL_INT *what, MKL_INT *val);

void    blacs_set(const MKL_INT *ConTxt, const MKL_INT *what, const MKL_INT *val);
void    blacs_set_(const MKL_INT *ConTxt, const MKL_INT *what, const MKL_INT *val);
void    BLACS_SET(const MKL_INT *ConTxt, const MKL_INT *what, const MKL_INT *val);
void    BLACS_SET_(const MKL_INT *ConTxt, const MKL_INT *what, const MKL_INT *val);

void    blacs_gridinit(MKL_INT *ConTxt, const char *layout, const MKL_INT *nprow, const MKL_INT *npcol);
void    blacs_gridinit_(MKL_INT *ConTxt, const char *layout, const MKL_INT *nprow, const MKL_INT *npcol);
void    BLACS_GRIDINIT(MKL_INT *ConTxt, const char *layout, const MKL_INT *nprow, const MKL_INT *npcol);
void    BLACS_GRIDINIT_(MKL_INT *ConTxt, const char *layout, const MKL_INT *nprow, const MKL_INT *npcol);

void    blacs_gridmap(MKL_INT *ConTxt, const MKL_INT *usermap, const MKL_INT *ldup, const MKL_INT *nprow0, const MKL_INT *npcol0);
void    blacs_gridmap_(MKL_INT *ConTxt, const MKL_INT *usermap, const MKL_INT *ldup, const MKL_INT *nprow0, const MKL_INT *npcol0);
void    BLACS_GRIDMAP(MKL_INT *ConTxt, const MKL_INT *usermap, const MKL_INT *ldup, const MKL_INT *nprow0, const MKL_INT *npcol0);
void    BLACS_GRIDMAP_(MKL_INT *ConTxt, const MKL_INT *usermap, const MKL_INT *ldup, const MKL_INT *nprow0, const MKL_INT *npcol0);


void    blacs_freebuff(const MKL_INT *ConTxt, const MKL_INT *Wait);
void    blacs_freebuff_(const MKL_INT *ConTxt, const MKL_INT *Wait);
void    BLACS_FREEBUFF(const MKL_INT *ConTxt, const MKL_INT *Wait);
void    BLACS_FREEBUFF_(const MKL_INT *ConTxt, const MKL_INT *Wait);

void    blacs_gridexit(const MKL_INT *ConTxt);
void    blacs_gridexit_(const MKL_INT *ConTxt);
void    BLACS_GRIDEXIT(const MKL_INT *ConTxt);
void    BLACS_GRIDEXIT_(const MKL_INT *ConTxt);

void    blacs_abort(const MKL_INT *ConTxt, const MKL_INT *ErrNo);
void    blacs_abort_(const MKL_INT *ConTxt, const MKL_INT *ErrNo);
void    BLACS_ABORT(const MKL_INT *ConTxt, const MKL_INT *ErrNo);
void    BLACS_ABORT_(const MKL_INT *ConTxt, const MKL_INT *ErrNo);

void    blacs_exit(const MKL_INT *notDone);
void    blacs_exit_(const MKL_INT *notDone);
void    BLACS_EXIT(const MKL_INT *notDone);
void    BLACS_EXIT_(const MKL_INT *notDone);


void    blacs_gridinfo(const MKL_INT *ConTxt, MKL_INT *nprow, MKL_INT *npcol, MKL_INT *myrow, MKL_INT *mycol);
void    blacs_gridinfo_(const MKL_INT *ConTxt, MKL_INT *nprow, MKL_INT *npcol, MKL_INT *myrow, MKL_INT *mycol);
void    BLACS_GRIDINFO(const MKL_INT *ConTxt, MKL_INT *nprow, MKL_INT *npcol, MKL_INT *myrow, MKL_INT *mycol);
void    BLACS_GRIDINFO_(const MKL_INT *ConTxt, MKL_INT *nprow, MKL_INT *npcol, MKL_INT *myrow, MKL_INT *mycol);

MKL_INT blacs_pnum(const MKL_INT *ConTxt, const MKL_INT *prow, const MKL_INT *pcol);
MKL_INT blacs_pnum_(const MKL_INT *ConTxt, const MKL_INT *prow, const MKL_INT *pcol);
MKL_INT BLACS_PNUM(const MKL_INT *ConTxt, const MKL_INT *prow, const MKL_INT *pcol);
MKL_INT BLACS_PNUM_(const MKL_INT *ConTxt, const MKL_INT *prow, const MKL_INT *pcol);

void    blacs_pcoord(const MKL_INT *ConTxt, const MKL_INT *nodenum, MKL_INT *prow, MKL_INT *pcol);
void    blacs_pcoord_(const MKL_INT *ConTxt, const MKL_INT *nodenum, MKL_INT *prow, MKL_INT *pcol);
void    BLACS_PCOORD(const MKL_INT *ConTxt, const MKL_INT *nodenum, MKL_INT *prow, MKL_INT *pcol);
void    BLACS_PCOORD_(const MKL_INT *ConTxt, const MKL_INT *nodenum, MKL_INT *prow, MKL_INT *pcol);


void    blacs_barrier(const MKL_INT *ConTxt, const char *scope);
void    blacs_barrier_(const MKL_INT *ConTxt, const char *scope);
void    BLACS_BARRIER(const MKL_INT *ConTxt, const char *scope);
void    BLACS_BARRIER_(const MKL_INT *ConTxt, const char *scope);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_BLACS_H_ */
