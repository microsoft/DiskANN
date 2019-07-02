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
!      Intel(R) Math Kernel Library (Intel(R) MKL) stdcall interface for BLAS routines
!******************************************************************************/

#ifndef _MKL_BLAS_STDCALL_H_
#define _MKL_BLAS_STDCALL_H_

#include "mkl_types.h"

#ifdef __GNUC__
#define MKL_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define MKL_DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: MKL {S,D}GEMM_ALLOC and {S,D}GEMM_FREE were declared deprecated. Use MKL {S,D}GEMM_PACK_GET_SIZE, MKL_MALLOC and MKL_FREE instead")
#define MKL_DEPRECATED
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if defined(MKL_STDCALL)

/* Upper case declaration */

void __stdcall XERBLA(const char *srname, int srname_len, const int *info);
int __stdcall LSAME(const char *ca, int ca_len, const char *cb, int cb_len);

/* BLAS Level1 */

float   __stdcall SCABS1(const MKL_Complex8 *c);
float   __stdcall SASUM(const MKL_INT *n, const float *x, const MKL_INT *incx);
void    __stdcall SAXPY(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, float *y, const MKL_INT *incy);
void    __stdcall SAXPBY(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void    __stdcall SAXPYI(const MKL_INT *nz, const float *a, const float *x, const MKL_INT *indx,float *y);
float   __stdcall SCASUM(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx);
float   __stdcall SCNRM2(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx);
void    __stdcall SCOPY(const MKL_INT *n, const float *x, const MKL_INT *incx, float *y, const MKL_INT *incy);
float   __stdcall SDOT(const MKL_INT *n, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy);
float   __stdcall SDSDOT(const MKL_INT *n, const float *sb, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy);
float   __stdcall SDOTI(const MKL_INT *nz, const float *x, const MKL_INT *indx, const float *y);
void    __stdcall SGTHR(const MKL_INT *nz, const float *y, float *x, const MKL_INT *indx);
void    __stdcall SGTHRZ(const MKL_INT *nz, float *y, float *x, const MKL_INT *indx);
float   __stdcall SNRM2(const MKL_INT *n, const float *x, const MKL_INT *incx);
void    __stdcall SROT(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy, const float *c, const float *s);
void    __stdcall SROTG(float *a,float *b,float *c,float *s);
void    __stdcall SROTI(const MKL_INT *nz, float *x, const MKL_INT *indx, float *y, const float *c, const float *s);
void    __stdcall SROTM(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy, const float *param);
void    __stdcall SROTMG(float *d1, float *d2, float *x1, const float *y1, float *param);
void    __stdcall SSCAL(const MKL_INT *n, const float *a, float *x, const MKL_INT *incx);
void    __stdcall SSCTR(const MKL_INT *nz, const float *x, const MKL_INT *indx, float *y);
void    __stdcall SSWAP(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy);
MKL_INT __stdcall ISAMAX(const MKL_INT *n, const float *x, const MKL_INT *incx);
MKL_INT __stdcall ISAMIN(const MKL_INT *n, const float *x, const MKL_INT *incx);

void    __stdcall CAXPY(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall CAXPBY(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall CAXPYI(const MKL_INT *nz, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT *indx, MKL_Complex8 *y);
void    __stdcall CCOPY(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall CDOTC(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, const  MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall CDOTCI(MKL_Complex8 *pres, const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, const MKL_Complex8 *y);
void    __stdcall CDOTU(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, const  MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall CDOTUI(MKL_Complex8 *pres, const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, const MKL_Complex8 *y);
void    __stdcall CGTHR(const MKL_INT *nz, const MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT *indx);
void    __stdcall CGTHRZ(const MKL_INT *nz, MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT *indx);
void    __stdcall CROTG(MKL_Complex8 *a, const MKL_Complex8 *b, float *c, MKL_Complex8 *s);
void    __stdcall CSCAL(const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *incx);
void    __stdcall CSCTR(const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, MKL_Complex8 *y);
void    __stdcall CSROT(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy, const float *c, const float *s);
void    __stdcall CSSCAL(const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *incx);
void    __stdcall CSWAP(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy);
MKL_INT __stdcall ICAMAX(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx);
MKL_INT __stdcall ICAMIN(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx);

double  __stdcall DCABS1(const MKL_Complex16 *z);
double  __stdcall DASUM(const MKL_INT *n, const double *x, const MKL_INT *incx);
void    __stdcall DAXPY(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
void    __stdcall DAXPBY(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void    __stdcall DAXPYI(const MKL_INT *nz, const double *a, const double *x, const MKL_INT *indx, double *y);
void    __stdcall DCOPY(const MKL_INT *n, const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
double  __stdcall DDOT(const  MKL_INT *n, const double *x, const MKL_INT *incx, const double *y, const MKL_INT *incy);
double  __stdcall DSDOT(const MKL_INT *n, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy);
double  __stdcall DDOTI(const MKL_INT *nz, const double *x, const MKL_INT *indx, const double *y);
void    __stdcall DGTHR(const MKL_INT *nz, const double *y, double *x, const MKL_INT *indx);
void    __stdcall DGTHRZ(const MKL_INT *nz, double *y, double *x, const MKL_INT *indx);
double  __stdcall DNRM2(const MKL_INT *n, const double *x, const MKL_INT *incx);
void    __stdcall DROT(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy, const double *c, const double *s);
void    __stdcall DROTG(double *a,double *b,double *c,double *s);
void    __stdcall DROTI(const MKL_INT *nz, double *x, const MKL_INT *indx, double *y, const double *c, const double *s);
void    __stdcall DROTM(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy, const double *param);
void    __stdcall DROTMG(double *d1, double *d2, double *x1, const double *y1, double *param);
void    __stdcall DSCAL(const MKL_INT *n, const double *a, double *x, const MKL_INT *incx);
void    __stdcall DSCTR(const MKL_INT *nz, const double *x, const MKL_INT *indx, double *y);
void    __stdcall DSWAP(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
double  __stdcall DZASUM(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx);
double  __stdcall DZNRM2(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx);
MKL_INT __stdcall IDAMAX(const MKL_INT *n, const double *x, const MKL_INT *incx);
MKL_INT __stdcall IDAMIN(const MKL_INT *n, const double *x, const MKL_INT *incx);

void    __stdcall ZAXPY(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall ZAXPBY(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall ZAXPYI(const MKL_INT *nz, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT *indx, MKL_Complex16 *y);
void    __stdcall ZCOPY(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall ZDOTC(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, const  MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall ZDOTCI(MKL_Complex16 *pres,const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, const MKL_Complex16 *y);
void    __stdcall ZDOTU(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall ZDOTUI(MKL_Complex16 *pres, const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, const MKL_Complex16 *y);
void    __stdcall ZDROT(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy, const double *c, const double *s);
void    __stdcall ZDSCAL(const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *incx);
void    __stdcall ZGTHR(const MKL_INT *nz, const MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT *indx);
void    __stdcall ZGTHRZ(const MKL_INT *nz, MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT *indx);
void    __stdcall ZROTG(MKL_Complex16 *a, const MKL_Complex16 *b, double *c, MKL_Complex16 *s);
void    __stdcall ZSCAL(const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *incx);
void    __stdcall ZSCTR(const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, MKL_Complex16 *y);
void    __stdcall ZSWAP(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy);
MKL_INT __stdcall IZAMAX(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx);
MKL_INT __stdcall IZAMIN(const  MKL_INT *n,const  MKL_Complex16 *x, const MKL_INT *incx);

/* BLAS Level2 */

void __stdcall SGBMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
                     const float *alpha, const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
                     const float *beta, float *y, const MKL_INT *incy);
void __stdcall SGEMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                     const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
                     const float *beta, float *y, const MKL_INT *incy);
void __stdcall SGER(const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                    const float *y, const MKL_INT *incy, float *a, const MKL_INT *lda);
void __stdcall SSBMV(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_INT *k,
                     const float *alpha, const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
                     const float *beta, float *y, const MKL_INT *incy);
void __stdcall SSPMV(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *ap,
                     const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void __stdcall SSPR(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, float *ap);
void __stdcall SSPR2(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                     const float *y, const MKL_INT *incy, float *ap);
void __stdcall SSYMV(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
                     const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void __stdcall SSYR(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                    float *a, const MKL_INT *lda);
void __stdcall SSYR2(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                     const float *y, const MKL_INT *incy, float *a, const MKL_INT *lda);
void __stdcall STBMV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx);
void __stdcall STBSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx);
void __stdcall STPMV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const float *ap, float *x, const MKL_INT *incx);
void __stdcall STPSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const float *ap, float *x, const MKL_INT *incx);
void __stdcall STRMV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *n,
                     const float *a, const MKL_INT *lda, float *b, const MKL_INT *incx);
void __stdcall STRSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx);
void __stdcall SGEM2VU(const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
                     const float *x1, const MKL_INT *incx1, const float *x2, const MKL_INT *incx2,
                     const float *beta, float *y1, const MKL_INT *incy1, float *y2, const MKL_INT *incy2);

void __stdcall CGBMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
                     const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta,
                     MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall CGEMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
                     const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall CGERC(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
                     MKL_Complex8 *a, const MKL_INT *lda);
void __stdcall CGERU(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
                     MKL_Complex8 *a, const MKL_INT *lda);
void __stdcall CHBMV(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
                     const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall CHEMV(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
                     const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall CHER(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
                    MKL_Complex8 *a, const MKL_INT *lda);
void __stdcall CHER2(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
                     MKL_Complex8 *a, const MKL_INT *lda);
void __stdcall CHPMV(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *ap,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta,
                     MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall CHPR(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
                    MKL_Complex8 *ap);
void __stdcall CHPR2(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
                     MKL_Complex8 *ap);
void __stdcall CTBMV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall CTBSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall CTPMV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall CTPSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall CTRMV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *b, const MKL_INT *incx);
void __stdcall CTRSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall CGEM2VC(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x1, const MKL_INT *incx1,
                     const MKL_Complex8 *x2, const MKL_INT *incx2, const MKL_Complex8 *beta,
                     MKL_Complex8 *y1, const MKL_INT *incy1, MKL_Complex8 *y2, const MKL_INT *incy2);
void __stdcall SCGEMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const float *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
                     const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);

void __stdcall DGBMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
                     const double *alpha, const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
                     const double *beta, double *y, const MKL_INT *incy);
void __stdcall DGEMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                     const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
                     const double *beta, double *y, const MKL_INT *incy);
void __stdcall DGER(const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                    const double *y, const MKL_INT *incy, double *a, const MKL_INT *lda);
void __stdcall DSBMV(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_INT *k, const double *alpha,
                     const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
                     const double *beta, double *y, const MKL_INT *incy);
void __stdcall DSPMV(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *ap,
                     const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void __stdcall DSPR(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, double *ap);
void __stdcall DSPR2(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                     const double *y, const MKL_INT *incy, double *ap);
void __stdcall DSYMV(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
                     const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void __stdcall DSYR(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                    double *a, const MKL_INT *lda);
void __stdcall DSYR2(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                     const double *y, const MKL_INT *incy, double *a, const MKL_INT *lda);
void __stdcall DTBMV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx);
void __stdcall DTBSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx);
void __stdcall DTPMV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const double *ap, double *x, const MKL_INT *incx);
void __stdcall DTPSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const double *ap, double *x, const MKL_INT *incx);
void __stdcall DTRMV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *n,
                     const double *a, const MKL_INT *lda, double *b, const MKL_INT *incx);
void __stdcall DTRSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx);
void __stdcall DGEM2VU(const MKL_INT *m, const MKL_INT *n, const double *alpha,
                     const double *a, const MKL_INT *lda, const double *x1, const MKL_INT *incx1,
                     const double *x2, const MKL_INT *incx2, const double *beta,
                     double *y1, const MKL_INT *incy1, double *y2, const MKL_INT *incy2);

void __stdcall ZGBMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta,
                     MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall ZGEMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall ZGERC(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
                     MKL_Complex16 *a, const MKL_INT *lda);
void __stdcall ZGERU(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
                     MKL_Complex16 *a, const MKL_INT *lda);
void __stdcall ZHBMV(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall ZHEMV(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall ZHER(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha,
                    const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *a, const MKL_INT *lda);
void __stdcall ZHER2(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
                     MKL_Complex16 *a, const MKL_INT *lda);
void __stdcall ZHPMV(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *ap,
                     const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta,
                     MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall ZHPR(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x,
                    const MKL_INT *incx, MKL_Complex16 *ap);
void __stdcall ZHPR2(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
                     MKL_Complex16 *ap);
void __stdcall ZTBMV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall ZTBSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall ZTPMV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall ZTPSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall ZTRMV(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *incx);
void __stdcall ZTRSV(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall ZGEM2VC(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x1, const MKL_INT *incx1,
                     const MKL_Complex16 *x2, const MKL_INT *incx2, const MKL_Complex16 *beta,
                     MKL_Complex16 *y1, const MKL_INT *incy1, MKL_Complex16 *y2, const MKL_INT *incy2);
void __stdcall DZGEMV(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const double *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);

/* BLAS Level3 */

void __stdcall SGEMM(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
                     const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED float* __stdcall SGEMM_ALLOC(const char *identifier, int identifier_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
size_t __stdcall SGEMM_PACK_GET_SIZE(const char *identifier, int identifier_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void __stdcall SGEMM_PACK(const char *identifier, int identifier_len, const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *src, const MKL_INT *ld, float *dest);
void __stdcall SGEMM_COMPUTE(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall SGEMM_FREE(float *mat);
void __stdcall SGEMM_BATCH(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                     const float *alpha_array, const float **a_array, const MKL_INT *lda_array, const float **b_array, const MKL_INT *ldb_array,
                     const float *beta_array, float **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall SGEMMT(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *n, const MKL_INT *k,
                      const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
                      const float *beta, float *c, const MKL_INT *ldc);
void __stdcall SSYMM(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
                     const float *beta, float *c, const MKL_INT *ldc);
void __stdcall SSYR2K(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
                      const float *beta, float *c, const MKL_INT *ldc);
void __stdcall SSYRK(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const float *alpha, const float *a, const MKL_INT *lda,
                     const float *beta, float *c, const MKL_INT *ldc);
void __stdcall STRMM(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
                     float *b, const MKL_INT *ldb);
void __stdcall STRSM(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
                     float *b, const MKL_INT *ldb);
void __stdcall STRSM_BATCH(const char *side_array, int side_len, const char *uplo_array, int uplo_len, const char *transa_array, int transa_len, const char *diag_array, int diag_len,
                           const MKL_INT *m_array, const MKL_INT *n_array, const float *alpha_array, const float *a_array, const MKL_INT *lda_array,
                           float *b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size);

void __stdcall CGEMM(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                     MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CGEMM_BATCH(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                           const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array, const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                           const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall SCGEMM(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex8 *alpha, const float *a, const MKL_INT *lda,
                     const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                     MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CGEMM3M(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                       const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                       const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                       MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CGEMM3M_BATCH(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                             const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array, const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                             const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall CGEMMT(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                      const MKL_Complex8 *b, const MKL_INT *ldb,
                      const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CHEMM(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                     MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CHER2K(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                      const MKL_Complex8 *b, const MKL_INT *ldb, const float *beta,
                      MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CHERK(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const float *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const float *beta, MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CSYMM(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *b, const MKL_INT *ldb,
                     const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CSYR2K(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                      const MKL_Complex8 *b, const MKL_INT *ldb,
                      const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CSYRK(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall CTRMM(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda,
                     MKL_Complex8 *b, const MKL_INT *ldb);
void __stdcall CTRSM(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda,
                     MKL_Complex8 *b, const MKL_INT *ldb);
void __stdcall CTRSM_BATCH(const char *side_array, int side_len, const char *uplo_array, int uplo_len, const char *transa_array, int transa_len, const char *diag_array, int diag_len,
                           const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex8 *alpha_array, const MKL_Complex8 *a_array, const MKL_INT *lda_array,
                           MKL_Complex8 *b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size);

void __stdcall DGEMM(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
                     const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED double* __stdcall DGEMM_ALLOC(const char *identifier, int identifier_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
size_t __stdcall DGEMM_PACK_GET_SIZE(const char *identifier, int identifier_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void __stdcall DGEMM_PACK(const char *identifier, int identifier_len, const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *src, const MKL_INT *ld, double *dest);
void __stdcall DGEMM_COMPUTE(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall DGEMM_FREE(double *mat);
void __stdcall DGEMM_BATCH(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                           const double *alpha_array, const double **a_array, const MKL_INT *lda_array, const double **b_array, const MKL_INT *ldb_array,
                           const double *beta_array, double **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall DGEMMT(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *n, const MKL_INT *k,
                      const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
                      double *beta, double *c, const MKL_INT *ldc);
void __stdcall DSYMM(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
                     const double *beta, double *c, const MKL_INT *ldc);
void __stdcall DSYR2K(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
                      double *beta, double *c, const MKL_INT *ldc);
void __stdcall DSYRK(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const double *alpha, const double *a, const MKL_INT *lda, const double *beta,
                     double *c, const MKL_INT *ldc);
void __stdcall DTRMM(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
                     double *b, const MKL_INT *ldb);
void __stdcall DTRSM(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
                     double *b, const MKL_INT *ldb);
void __stdcall DTRSM_BATCH(const char *side_array, int side_len, const char *uplo_array, int uplo_len, const char *transa_array, int transa_len, const char *diag_array, int diag_len,
                           const MKL_INT *m_array, const MKL_INT *n_array, const double *alpha_array, const double *a_array, const MKL_INT *lda_array,
                           double *b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size);

void __stdcall ZGEMM(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                     MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZGEMM_BATCH(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                           const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array, const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                           const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall DZGEMM(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *alpha, const double *a, const MKL_INT *lda,
                     const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                     MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZGEMM3M(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                       const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                       const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                       MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZGEMM3M_BATCH(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                             const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array, const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                             const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall ZGEMMT(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                      const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                      MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZHEMM(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                     MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZHER2K(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                      const MKL_Complex16 *b, const MKL_INT *ldb, const double *beta,
                      MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZHERK(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const double *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const double *beta, MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZSYMM(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                     MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZSYR2K(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                      const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                      MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZSYRK(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ZTRMM(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *ldb);
void __stdcall ZTRSM(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *ldb);
void __stdcall ZTRSM_BATCH(const char *side_array, int side_len, const char *uplo_array, int uplo_len, const char *transa_array, int transa_len, const char *diag_array, int diag_len,
                           const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex16 *alpha_array, const MKL_Complex16 *a_array, const MKL_INT *lda_array,
                           MKL_Complex16 *b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size);


void __stdcall GEMM_S8U8S32(const char *transa, int transa_len, const char *transb, int transb_len, const char *offsetc, int offsetc_len,
                    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, 
                    const float *alpha, const MKL_INT8 *a, const MKL_INT *lda, const MKL_INT8 *ao, 
                    const MKL_UINT8 *b, const MKL_INT *ldb, const MKL_INT8 *bo, 
                    const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);
void __stdcall GEMM_S16S16S32(const char *transa, int transa_len, const char *transb, int transb_len, const char *offsetc, int offsetc_len,
                    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, 
                    const float *alpha, const MKL_INT16 *a, const MKL_INT *lda, const MKL_INT16 *ao, 
                    const MKL_INT16 *b, const MKL_INT *ldb, const MKL_INT16 *bo, 
                    const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);


size_t  __stdcall GEMM_S8U8S32_PACK_GET_SIZE   (const char *identifier, int identifier_len, const MKL_INT *m, 
                                                 const MKL_INT *n, const MKL_INT *k);
size_t  __stdcall GEMM_S16S16S32_PACK_GET_SIZE (const char *identifier, int identifier_len, const MKL_INT *m, 
                                                 const MKL_INT *n, const MKL_INT *k);
void  __stdcall GEMM_S8U8S32_PACK     (const char *identifier,  int identifier_len, const char *trans, 
                                       int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                                       const void *src, const MKL_INT *ld, void *dest);
void  __stdcall GEMM_S16S16S32_PACK   (const char *identifier, int identifier_len, const char *trans, 
                                       int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                                       const MKL_INT16 *src, const MKL_INT *ld, MKL_INT16 *dest);
void  __stdcall GEMM_S8U8S32_COMPUTE  (const char *transa, int transa_len, const char *transb, int transb_len, 
                                       const char *offsetc, int offsetc_len, 
                                       const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                                       const float *alpha,
                                       const MKL_INT8  *a, const MKL_INT *lda,  const MKL_INT8 *ao,
                                       const MKL_UINT8 *b, const MKL_INT *ldb,  const MKL_INT8 *bo,
                                       const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);
void  __stdcall GEMM_S16S16S32_COMPUTE(const char *transa, int transa_len, const char *transb, int transb_len, 
                                       const char *offsetc, int offsetc_len,
                                       const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                                       const float *alpha,
                                       const MKL_INT16 *a, const MKL_INT *lda,  const MKL_INT16 *ao,
                                       const MKL_INT16 *b, const MKL_INT *ldb,  const MKL_INT16 *bo,
                                       const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);


/* Lower case declaration */

void __stdcall xerbla(const char *srname, int srname_len, const int *info);
int __stdcall lsame(const char *ca, int ca_len, const char *cb, int cb_len);

/* BLAS Level1 */
float   __stdcall scabs1(const MKL_Complex8 *c);
float   __stdcall sasum(const MKL_INT *n, const float *x, const MKL_INT *incx);
void    __stdcall saxpy(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, float *y, const MKL_INT *incy);
void    __stdcall saxpby(const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void    __stdcall saxpyi(const MKL_INT *nz, const float *a, const float *x, const MKL_INT *indx, float *y);
float   __stdcall scasum(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx);
float   __stdcall scnrm2(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx);
void    __stdcall scopy(const MKL_INT *n, const float *x, const MKL_INT *incx, float *y, const MKL_INT *incy);
float   __stdcall sdot(const MKL_INT *n, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy);
float   __stdcall sdoti(const MKL_INT *nz, const float *x, const MKL_INT *indx, const float *y);
float   __stdcall sdsdot(const MKL_INT *n, const float *sb, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy);
void    __stdcall sgthr(const MKL_INT *nz, const float *y, float *x, const MKL_INT *indx);
void    __stdcall sgthrz(const MKL_INT *nz, float *y, float *x, const MKL_INT *indx);
float   __stdcall snrm2(const MKL_INT *n, const float *x, const MKL_INT *incx);
void    __stdcall srot(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy, const float *c, const float *s);
void    __stdcall srotg(float *a,float *b,float *c,float *s);
void    __stdcall sroti(const MKL_INT *nz, float *x, const MKL_INT *indx, float *y, const float *c, const float *s);
void    __stdcall srotm(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy, const float *param);
void    __stdcall srotmg(float *d1, float *d2, float *x1, const float *y1, float *param);
void    __stdcall sscal(const MKL_INT *n, const float *a, float *x, const MKL_INT *incx);
void    __stdcall ssctr(const MKL_INT *nz, const float *x, const MKL_INT *indx, float *y);
void    __stdcall sswap(const MKL_INT *n, float *x, const MKL_INT *incx, float *y, const MKL_INT *incy);
MKL_INT __stdcall isamax(const MKL_INT *n, const float *x, const MKL_INT *incx);
MKL_INT __stdcall isamin(const MKL_INT *n, const float *x, const MKL_INT *incx);

void    __stdcall caxpy(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall caxpby(const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall caxpyi(const MKL_INT *nz, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT *indx, MKL_Complex8 *y);
void    __stdcall ccopy(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall cdotc(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall cdotci(MKL_Complex8 *pres, const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, const MKL_Complex8 *y);
void    __stdcall cdotu(MKL_Complex8 *pres, const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy);
void    __stdcall cdotui(MKL_Complex8 *pres, const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, const MKL_Complex8 *y);
void    __stdcall cgthr(const MKL_INT *nz, const MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT *indx);
void    __stdcall cgthrz(const MKL_INT *nz, MKL_Complex8 *y, MKL_Complex8 *x, const MKL_INT *indx);
void    __stdcall crotg(MKL_Complex8 *a, const MKL_Complex8 *b, float *c, MKL_Complex8 *s);
void    __stdcall cscal(const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *incx);
void    __stdcall csctr(const MKL_INT *nz, const MKL_Complex8 *x, const MKL_INT *indx, MKL_Complex8 *y);
void    __stdcall csrot(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy, const float *c, const float *s);
void    __stdcall csscal(const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *incx);
void    __stdcall cswap(const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *incy);
MKL_INT __stdcall icamax(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx);
MKL_INT __stdcall icamin(const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *incx);

double  __stdcall dcabs1(const MKL_Complex16 *z);
double  __stdcall dasum(const MKL_INT *n, const double *x, const MKL_INT *incx);
void    __stdcall daxpy(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
void    __stdcall daxpby(const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void    __stdcall daxpyi(const MKL_INT *nz, const double *a, const double *x, const MKL_INT *indx, double *y);
void    __stdcall dcopy(const MKL_INT *n, const double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
double  __stdcall ddot(const MKL_INT *n, const double *x, const MKL_INT *incx, const double *y, const MKL_INT *incy);
double  __stdcall dsdot(const MKL_INT *n, const float *x, const MKL_INT *incx, const float *y, const MKL_INT *incy);
double  __stdcall ddoti(const MKL_INT *nz, const double *x, const MKL_INT *indx, const double *y);
void    __stdcall dgthr(const MKL_INT *nz, const double *y, double *x, const MKL_INT *indx);
void    __stdcall dgthrz(const MKL_INT *nz, double *y, double *x, const MKL_INT *indx);
double  __stdcall dnrm2(const MKL_INT *n, const double *x, const MKL_INT *incx);
void    __stdcall drot(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy, const double *c, const double *s);
void    __stdcall drotg(double *a, double *b, double *c, double *s);
void    __stdcall droti(const MKL_INT *nz, double *x, const MKL_INT *indx, double *y, const double *c, const double *s);
void    __stdcall drotm(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy, const double *param);
void    __stdcall drotmg(double *d1, double *d2, double *x1, const double *y1, double *param);
void    __stdcall dscal(const MKL_INT *n, const double *a, double *x, const MKL_INT *incx);
void    __stdcall dsctr(const MKL_INT *nz, const double *x, const MKL_INT *indx, double *y);
void    __stdcall dswap(const MKL_INT *n, double *x, const MKL_INT *incx, double *y, const MKL_INT *incy);
double  __stdcall dzasum(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx);
double  __stdcall dznrm2(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx);
MKL_INT __stdcall idamax(const MKL_INT *n, const double *x, const MKL_INT *incx);
MKL_INT __stdcall idamin(const MKL_INT *n, const double *x, const MKL_INT *incx);

void    __stdcall zaxpy(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall zaxpby(const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall zaxpyi(const MKL_INT *nz, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT *indx, MKL_Complex16 *y);
void    __stdcall zcopy(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall zdotc(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall zdotci(MKL_Complex16 *pres, const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, const MKL_Complex16 *y);
void    __stdcall zdotu(MKL_Complex16 *pres, const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy);
void    __stdcall zdotui(MKL_Complex16 *pres, const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, const MKL_Complex16 *y);
void    __stdcall zdrot(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy, const double *c, const double *s);
void    __stdcall zdscal(const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *incx);
void    __stdcall zgthr(const MKL_INT *nz, const MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT *indx);
void    __stdcall zgthrz(const MKL_INT *nz, MKL_Complex16 *y, MKL_Complex16 *x, const MKL_INT *indx);
void    __stdcall zrotg(MKL_Complex16 *a, const MKL_Complex16 *b, double *c, MKL_Complex16 *s);
void    __stdcall zscal(const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *incx);
void    __stdcall zsctr(const MKL_INT *nz, const MKL_Complex16 *x, const MKL_INT *indx, MKL_Complex16 *y);
void    __stdcall zswap(const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *incy);
MKL_INT __stdcall izamax(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx);
MKL_INT __stdcall izamin(const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *incx);

/* blas level2 */

void __stdcall sgbmv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
                     const float *alpha, const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
                     const float *beta, float *y, const MKL_INT *incy);
void __stdcall sgemv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const float *alpha,
                     const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
                     const float *beta, float *y, const MKL_INT *incy);
void __stdcall sger(const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                    const float *y, const MKL_INT *incy, float *a, const MKL_INT *lda);
void __stdcall ssbmv(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_INT *k, const float *alpha,
                     const float *a, const MKL_INT *lda, const float *x, const MKL_INT *incx,
                     const float *beta, float *y, const MKL_INT *incy);
void __stdcall sspmv(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *ap,
                     const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void __stdcall sspr(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                    float *ap);
void __stdcall sspr2(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                     const float *y, const MKL_INT *incy, float *ap);
void __stdcall ssymv(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
                     const float *x, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *incy);
void __stdcall ssyr(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                    float *a, const MKL_INT *lda);
void __stdcall ssyr2(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *incx,
                     const float *y, const MKL_INT *incy, float *a, const MKL_INT *lda);
void __stdcall stbmv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx);
void __stdcall stbsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx);
void __stdcall stpmv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const float *ap,
                     float *x, const MKL_INT *incx);
void __stdcall stpsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const float *ap,
                     float *x, const MKL_INT *incx);
void __stdcall strmv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *n, const float *a,
                     const MKL_INT *lda, float *b, const MKL_INT *incx);
void __stdcall strsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const float *a, const MKL_INT *lda, float *x, const MKL_INT *incx);
void __stdcall sgem2vu(const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
                     const float *x1, const MKL_INT *incx1, const float *x2, const MKL_INT *incx2,
                     const float *beta, float *y1, const MKL_INT *incy1, float *y2, const MKL_INT *incy2);

void __stdcall cgbmv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
                     const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta,
                     MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall cgemv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
                     const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall cgerc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
                     MKL_Complex8 *a, const MKL_INT *lda);
void __stdcall cgeru(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
                     MKL_Complex8 *a, const MKL_INT *lda);
void __stdcall chbmv(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
                     const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall chemv(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
                     const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall cher(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
                    MKL_Complex8 *a, const MKL_INT *lda);
void __stdcall cher2(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *incy,
                     MKL_Complex8 *a, const MKL_INT *lda);
void __stdcall chpmv(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *ap,
                     const MKL_Complex8 *x, const MKL_INT *incx, const MKL_Complex8 *beta,
                     MKL_Complex8 *y, const MKL_INT *incy);
void __stdcall chpr(const char *uplo, int uplo_len, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
                    MKL_Complex8 *ap);
void __stdcall chpr2(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *incx,
                     const MKL_Complex8 *y, const MKL_INT *incy, MKL_Complex8 *ap);
void __stdcall ctbmv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall ctbsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall ctpmv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall ctpsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex8 *ap, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall ctrmv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *b, const MKL_INT *incx);
void __stdcall ctrsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *x, const MKL_INT *incx);
void __stdcall cgem2vc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, const MKL_Complex8 *x1, const MKL_INT *incx1,
                     const MKL_Complex8 *x2, const MKL_INT *incx2, const MKL_Complex8 *beta,
                     MKL_Complex8 *y1, const MKL_INT *incy1, MKL_Complex8 *y2, const MKL_INT *incy2);
void __stdcall scgemv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const float *a, const MKL_INT *lda, const MKL_Complex8 *x, const MKL_INT *incx,
                     const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *incy);

void __stdcall dgbmv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
                     const double *alpha, const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
                     const double *beta, double *y, const MKL_INT *incy);
void __stdcall dgemv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const double *alpha,
                     const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
                     const double *beta, double *y, const MKL_INT *incy);
void __stdcall dger(const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                    const double *y, const MKL_INT *incy, double *a, const MKL_INT *lda);
void __stdcall dsbmv(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_INT *k, const double *alpha,
                     const double *a, const MKL_INT *lda, const double *x, const MKL_INT *incx,
                     const double *beta, double *y, const MKL_INT *incy);
void __stdcall dspmv(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *ap,
                     const double *x, const MKL_INT *incx, const double *beta,
                     double *y, const MKL_INT *incy);
void __stdcall dspr(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                    double *ap);
void __stdcall dspr2(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                     const double *y, const MKL_INT *incy, double *ap);
void __stdcall dsymv(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
                     const double *x, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *incy);
void __stdcall dsyr(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                    double *a, const MKL_INT *lda);
void __stdcall dsyr2(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *incx,
                     const double *y, const MKL_INT *incy, double *a, const MKL_INT *lda);
void __stdcall dtbmv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx);
void __stdcall dtbsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx);
void __stdcall dtpmv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const double *ap, double *x, const MKL_INT *incx);
void __stdcall dtpsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const double *ap, double *x, const MKL_INT *incx);
void __stdcall dtrmv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *n,
                     const double *a, const MKL_INT *lda, double *b, const MKL_INT *incx);
void __stdcall dtrsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const double *a, const MKL_INT *lda, double *x, const MKL_INT *incx);
void __stdcall dgem2vu(const MKL_INT *m, const MKL_INT *n, const double *alpha,
                     const double *a, const MKL_INT *lda, const double *x1, const MKL_INT *incx1,
                     const double *x2, const MKL_INT *incx2, const double *beta,
                     double *y1, const MKL_INT *incy1, double *y2, const MKL_INT *incy2);

void __stdcall zgbmv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *kl, const MKL_INT *ku,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta,
                     MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall zgemv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall zgerc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *y, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *lda);
void __stdcall zgeru(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *y, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *lda);
void __stdcall zhbmv(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall zhemv(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall zher(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
                    MKL_Complex16 *a, const MKL_INT *lda);
void __stdcall zher2(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *incy,
                     MKL_Complex16 *a, const MKL_INT *lda);
void __stdcall zhpmv(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *ap,
                     const MKL_Complex16 *x, const MKL_INT *incx, const MKL_Complex16 *beta,
                     MKL_Complex16 *y, const MKL_INT *incy);
void __stdcall zhpr(const char *uplo, int uplo_len, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
                    MKL_Complex16 *ap);
void __stdcall zhpr2(const char *uplo, int uplo_len, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *y, const MKL_INT *incy, MKL_Complex16 *ap);
void __stdcall ztbmv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall ztbsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall ztpmv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall ztpsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex16 *ap, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall ztrmv(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *incx);
void __stdcall ztrsv(const char *uplo, int uplo_len, const char *trans, int trans_len, const char *diag, int diag_len, const MKL_INT *n,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *x, const MKL_INT *incx);
void __stdcall zgem2vc(const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, const MKL_Complex16 *x1, const MKL_INT *incx1,
                     const MKL_Complex16 *x2, const MKL_INT *incx2, const MKL_Complex16 *beta,
                     MKL_Complex16 *y1, const MKL_INT *incy1, MKL_Complex16 *y2, const MKL_INT *incy2);
void __stdcall dzgemv(const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const double *a, const MKL_INT *lda, const MKL_Complex16 *x, const MKL_INT *incx,
                     const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *incy);

/* blas level3 */

void __stdcall sgemm(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
                     const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED float* __stdcall sgemm_alloc(const char *identifier, int identifier_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
size_t __stdcall sgemm_pack_get_size(const char *identifier, int identifier_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void __stdcall sgemm_pack(const char *identifier, int identifier_len, const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const float *alpha, const float *src, const MKL_INT *ld, float *dest);
void __stdcall sgemm_compute(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb, const float *beta, float *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall sgemm_free(float *mat);
void __stdcall sgemm_batch(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                           const float *alpha_array, const float **a_array, const MKL_INT *lda_array, const float **b_array, const MKL_INT *ldb_array,
                           const float *beta_array, float **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall sgemmt(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *n, const MKL_INT *k,
                      const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
                      const float *beta, float *c, const MKL_INT *ldc);
void __stdcall ssymm(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
                     const float *beta, float *c, const MKL_INT *ldc);
void __stdcall ssyr2k(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const float *alpha, const float *a, const MKL_INT *lda, const float *b, const MKL_INT *ldb,
                      const float *beta, float *c, const MKL_INT *ldc);
void __stdcall ssyrk(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const float *alpha, const float *a, const MKL_INT *lda, const float *beta,
                     float *c, const MKL_INT *ldc);
void __stdcall strmm(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
                     float *b, const MKL_INT *ldb);
void __stdcall strsm(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *lda,
                     float *b, const MKL_INT *ldb);
void __stdcall strsm_batch(const char *side_array, int side_len, const char *uplo_array, int uplo_len, const char *transa_array, int transa_len, const char *diag_array, int diag_len,
                           const MKL_INT *m_array, const MKL_INT *n_array, const float *alpha_array, const float *a_array, const MKL_INT *lda_array,
                           float *b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size);

void __stdcall cgemm(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                     MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall cgemm_batch(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                           const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array, const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                           const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall scgemm(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex8 *alpha, const float *a, const MKL_INT *lda,
                      const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                      MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall cgemm3m(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                       const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                       const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                       MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall cgemm3m_batch(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                             const MKL_Complex8 *alpha_array, const MKL_Complex8 **a_array, const MKL_INT *lda_array, const MKL_Complex8 **b_array, const MKL_INT *ldb_array,
                             const MKL_Complex8 *beta_array, MKL_Complex8 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall cgemmt(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                      const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                      MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall chemm(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                     MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall cher2k(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                      const MKL_Complex8 *b, const MKL_INT *ldb, const float *beta,
                      MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall cherk(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const float *alpha, const MKL_Complex8 *a, const MKL_INT *lda, const float *beta,
                     MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall csymm(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                     MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall csyr2k(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                      const MKL_Complex8 *b, const MKL_INT *ldb, const MKL_Complex8 *beta,
                      MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall csyrk(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *lda,
                     const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ldc);
void __stdcall ctrmm(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *b, const MKL_INT *ldb);
void __stdcall ctrsm(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha,
                     const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *b, const MKL_INT *ldb);
void __stdcall ctrsm_batch(const char *side_array, int side_len, const char *uplo_array, int uplo_len, const char *transa_array, int transa_len, const char *diag_array, int diag_len,
                           const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex8 *alpha_array, const MKL_Complex8 *a_array, const MKL_INT *lda_array,
                           MKL_Complex8 *b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size);

void __stdcall dgemm(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
                     const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED double* __stdcall dgemm_alloc(const char *identifier, int identifier_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
size_t __stdcall dgemm_pack_get_size(const char *identifier, int identifier_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k);
void __stdcall dgemm_pack(const char *identifier, int identifier_len, const char *trans, int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const double *alpha, const double *src, const MKL_INT *ld, double *dest);
void __stdcall dgemm_compute(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
           const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb, const double *beta, double *c, const MKL_INT *ldc);
MKL_DEPRECATED void __stdcall dgemm_free(double *mat);
void __stdcall dgemm_batch(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                           const double *alpha_array, const double **a_array, const MKL_INT *lda_array, const double **b_array, const MKL_INT *ldb_array,
                           const double *beta_array, double **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall dgemmt(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *n, const MKL_INT *k,
                      const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
                      const double *beta, double *c, const MKL_INT *ldc);
void __stdcall dsymm(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
                     const double *beta, double *c, const MKL_INT *ldc);
void __stdcall dsyr2k(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const double *alpha, const double *a, const MKL_INT *lda, const double *b, const MKL_INT *ldb,
                      const double *beta, double *c, const MKL_INT *ldc);
void __stdcall dsyrk(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const double *alpha, const double *a, const MKL_INT *lda, const double *beta,
                     double *c, const MKL_INT *ldc);
void __stdcall dtrmm(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
                     double *b, const MKL_INT *ldb);
void __stdcall dtrsm(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *lda,
                     double *b, const MKL_INT *ldb);
void __stdcall dtrsm_batch(const char *side_array, int side_len, const char *uplo_array, int uplo_len, const char *transa_array, int transa_len, const char *diag_array, int diag_len,
                           const MKL_INT *m_array, const MKL_INT *n_array, const double *alpha_array, const double *a_array, const MKL_INT *lda_array,
                           double *b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size);

void __stdcall zgemm(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                     MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall zgemm_batch(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                           const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array, const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                           const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall dzgemm(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *alpha, const double *a, const MKL_INT *lda,
                     const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                     MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall zgemm3m(const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                       const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                       const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                       MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall zgemm3m_batch(const char *transa_array, int transa_len, const char *transb_array, int transb_len, const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
                             const MKL_Complex16 *alpha_array, const MKL_Complex16 **a_array, const MKL_INT *lda_array, const MKL_Complex16 **b_array, const MKL_INT *ldb_array,
                             const MKL_Complex16 *beta_array, MKL_Complex16 **c_array, const MKL_INT *ldc_array, const MKL_INT *group_count, const MKL_INT *group_size);
void __stdcall zgemmt(const char *uplo, int uplo_len, const char *transa, int transa_len, const char *transb, int transb_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                      const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                      MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall zhemm(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                     MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall zher2k(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                      const MKL_Complex16 *b, const MKL_INT *ldb, const double *beta,
                      MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall zherk(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const double *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const double *beta, MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall zsymm(const char *side, int side_len, const char *uplo, int uplo_len, const MKL_INT *m, const MKL_INT *n,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                     MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall zsyr2k(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                      const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                      const MKL_Complex16 *b, const MKL_INT *ldb, const MKL_Complex16 *beta,
                      MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall zsyrk(const char *uplo, int uplo_len, const char *trans, int trans_len, const MKL_INT *n, const MKL_INT *k,
                     const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *lda,
                     const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ldc);
void __stdcall ztrmm(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *ldb);
void __stdcall ztrsm(const char *side, int side_len, const char *uplo, int uplo_len, const char *transa, int transa_len, const char *diag, int diag_len,
                     const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha,
                     const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *ldb);
void __stdcall ztrsm_batch(const char *side_array, int side_len, const char *uplo_array, int uplo_len, const char *transa_array, int transa_len, const char *diag_array, int diag_len,
                           const MKL_INT *m_array, const MKL_INT *n_array, const MKL_Complex16 *alpha_array, const MKL_Complex16 *a_array, const MKL_INT *lda_array,
                           MKL_Complex16 *b_array, const MKL_INT *ldb_array, const MKL_INT *group_count, const MKL_INT *group_size);

void __stdcall gemm_s8u8s32 (const char *transa, int transa_len, const char *transb, int transb_len, const char *offsetc, int offsetc_len,
                    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                    const float *alpha, const MKL_INT8 *a, const MKL_INT *lda, const MKL_INT8 *ao,
                    const MKL_UINT8 *b, const MKL_INT *ldb, const MKL_INT8 *bo,
                    const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);
void __stdcall gemm_s16s16s32(const char *transa, int transa_len, const char *transb, int transb_len, const char *offsetc, int offsetc_len,
                    const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, 
                    const float *alpha, const MKL_INT16 *a, const MKL_INT *lda, const MKL_INT16 *ao, 
                    const MKL_INT16 *b, const MKL_INT *ldb, const MKL_INT16 *bo, 
                    const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);



size_t  __stdcall gemm_s8u8s32_pack_get_size   (const char *identifier, int identifier_len, const MKL_INT *m, 
                                                 const MKL_INT *n, const MKL_INT *k);
size_t  __stdcall gemm_s16s16s32_pack_get_size (const char *identifier, int identifier_len, const MKL_INT *m, 
                                                 const MKL_INT *n, const MKL_INT *k);
void  __stdcall gemm_s8u8s32_pack     (const char *identifier,  int identifier_len, const char *trans, 
                                       int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                                       const void *src, const MKL_INT *ld, void *dest);
void  __stdcall gemm_s16s16s32_pack   (const char *identifier, int identifier_len, const char *trans, 
                                       int trans_len, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                                       const MKL_INT16 *src, const MKL_INT *ld, MKL_INT16 *dest);
void  __stdcall gemm_s8u8s32_compute  (const char *transa, int transa_len, const char *transb, int transb_len, 
                                       const char *offsetc, int offsetc_len, 
                                       const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                                       const float *alpha,
                                       const MKL_INT8  *a, const MKL_INT *lda,  const MKL_INT8 *ao,
                                       const MKL_UINT8 *b, const MKL_INT *ldb,  const MKL_INT8 *bo,
                                       const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);
void  __stdcall gemm_s16s16s32_compute(const char *transa, int transa_len, const char *transb, int transb_len, 
                                       const char *offsetc, int offsetc_len,
                                       const MKL_INT *m, const MKL_INT *n, const MKL_INT *k,
                                       const float *alpha,
                                       const MKL_INT16 *a, const MKL_INT *lda,  const MKL_INT16 *ao,
                                       const MKL_INT16 *b, const MKL_INT *ldb,  const MKL_INT16 *bo,
                                       const float *beta, MKL_INT32 *c, const MKL_INT *ldc, const MKL_INT32 *co);



#endif /* MKL_STDCALL */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_BLAS_STDCALL_H_ */
