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
!      Intel(R) Math Kernel Library (Intel(R) MKL) interface for PBLAS routines
!******************************************************************************/

#ifndef _MKL_PBLAS_H_
#define _MKL_PBLAS_H_

#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef _NETLIB_PBLAS_TYPES

/* PBLAS Level 1 Routines */

void psamax( const MKL_INT *n, float *amax, MKL_INT *indx, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdamax( const MKL_INT *n, double *amax, MKL_INT *indx, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pcamax( const MKL_INT *n, MKL_Complex8 *amax, MKL_INT *indx, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pzamax( const MKL_INT *n, MKL_Complex16 *amax, MKL_INT *indx, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void psasum( const MKL_INT *n, float *asum, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdasum( const MKL_INT *n, double *asum, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void pscasum( const MKL_INT *n, float *asum, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdzasum( const MKL_INT *n, double *asum, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void psaxpy( const MKL_INT *n, const float *a, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdaxpy( const MKL_INT *n, const double *a, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcaxpy( const MKL_INT *n, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzaxpy( const MKL_INT *n, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void picopy( const MKL_INT *n, const MKL_INT *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_INT *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pscopy( const MKL_INT *n, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdcopy( const MKL_INT *n, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pccopy( const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzcopy( const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void psdot( const MKL_INT *n, float *dot, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pddot( const MKL_INT *n, double *dot, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void pcdotc( const MKL_INT *n, MKL_Complex8 *dotc, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzdotc( const MKL_INT *n, MKL_Complex16 *dotc, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void pcdotu( const MKL_INT *n, MKL_Complex8 *dotu, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzdotu( const MKL_INT *n, MKL_Complex16 *dotu, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void psnrm2( const MKL_INT *n, float *norm2, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdnrm2( const MKL_INT *n, double *norm2, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pscnrm2( const MKL_INT *n, float *norm2, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdznrm2( const MKL_INT *n, double *norm2, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void psscal( const MKL_INT *n, const float *a, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdscal( const MKL_INT *n, const double *a, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pcscal( const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pzscal( const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pcsscal( const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pzdscal( const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void psswap( const MKL_INT *n, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdswap( const MKL_INT *n, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcswap( const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzswap( const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

/* PBLAS Level 2 Routines */

void psgemv( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdgemv( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcgemv( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzgemv( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void psagemv( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdagemv( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcagemv( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzagemv( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void psger( const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pdger( const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void pcgerc( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pzgerc( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void pcgeru( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pzgeru( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void pchemv( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzhemv( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcahemv( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzahemv( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void pcher( const char *uplo, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pzher( const char *uplo, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void pcher2( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pzher2( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void pssymv( const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdsymv( const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void psasymv( const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdasymv( const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void pssyr( const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pdsyr( const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void pssyr2( const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pdsyr2( const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void pstrmv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdtrmv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pctrmv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pztrmv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void psatrmv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdatrmv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcatrmv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzatrmv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void pstrsv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdtrsv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pctrsv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pztrsv( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

/* PBLAS Level 3 Routines */

void psgemm( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdgemm( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcgemm( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzgemm( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pchemm( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzhemm( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pcherk( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzherk( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pcher2k( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzher2k( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pssymm( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdsymm( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcsymm( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzsymm( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pssyrk( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdsyrk( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcsyrk( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzsyrk( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pssyr2k( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdsyr2k( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcsyr2k( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzsyr2k( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pstran( const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdtran( const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pctranu( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pztranu( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pctranc( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pztranc( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pstrmm( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pdtrmm( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pctrmm( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pztrmm( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );

void pstrsm( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pdtrsm( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pctrsm( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pztrsm( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );

void psgeadd( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdgeadd( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcgeadd( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzgeadd( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void pstradd( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdtradd( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pctradd( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pztradd( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

/* PBLAS Auxiliary Routines */

MKL_INT	pilaenv( const MKL_INT *ictxt, const char *prec );

/* OTHER NAMING CONVENSIONS FOLLOW */

/* PBLAS Level 1 Routines */

void PSAMAX( const MKL_INT *n, float *amax, MKL_INT *indx, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDAMAX( const MKL_INT *n, double *amax, MKL_INT *indx, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PCAMAX( const MKL_INT *n, MKL_Complex8 *amax, MKL_INT *indx, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZAMAX( const MKL_INT *n, MKL_Complex16 *amax, MKL_INT *indx, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PSAMAX_( const MKL_INT *n, float *amax, MKL_INT *indx, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDAMAX_( const MKL_INT *n, double *amax, MKL_INT *indx, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PCAMAX_( const MKL_INT *n, MKL_Complex8 *amax, MKL_INT *indx, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZAMAX_( const MKL_INT *n, MKL_Complex16 *amax, MKL_INT *indx, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void psamax_( const MKL_INT *n, float *amax, MKL_INT *indx, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdamax_( const MKL_INT *n, double *amax, MKL_INT *indx, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pcamax_( const MKL_INT *n, MKL_Complex8 *amax, MKL_INT *indx, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pzamax_( const MKL_INT *n, MKL_Complex16 *amax, MKL_INT *indx, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void PSASUM( const MKL_INT *n, float *asum, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDASUM( const MKL_INT *n, double *asum, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PSASUM_( const MKL_INT *n, float *asum, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDASUM_( const MKL_INT *n, double *asum, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void psasum_( const MKL_INT *n, float *asum, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdasum_( const MKL_INT *n, double *asum, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void PSCASUM( const MKL_INT *n, float *asum, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDZASUM( const MKL_INT *n, double *asum, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PSCASUM_( const MKL_INT *n, float *asum, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDZASUM_( const MKL_INT *n, double *asum, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pscasum_( const MKL_INT *n, float *asum, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdzasum_( const MKL_INT *n, double *asum, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void PSAXPY( const MKL_INT *n, const float *a, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDAXPY( const MKL_INT *n, const double *a, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCAXPY( const MKL_INT *n, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZAXPY( const MKL_INT *n, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSAXPY_( const MKL_INT *n, const float *a, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDAXPY_( const MKL_INT *n, const double *a, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCAXPY_( const MKL_INT *n, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZAXPY_( const MKL_INT *n, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void psaxpy_( const MKL_INT *n, const float *a, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdaxpy_( const MKL_INT *n, const double *a, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcaxpy_( const MKL_INT *n, const MKL_Complex8 *a, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzaxpy_( const MKL_INT *n, const MKL_Complex16 *a, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PICOPY( const MKL_INT *n, const MKL_INT *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_INT *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSCOPY( const MKL_INT *n, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDCOPY( const MKL_INT *n, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCCOPY( const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZCOPY( const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PICOPY_( const MKL_INT *n, const MKL_INT *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_INT *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSCOPY_( const MKL_INT *n, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDCOPY_( const MKL_INT *n, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCCOPY_( const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZCOPY_( const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void picopy_( const MKL_INT *n, const MKL_INT *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_INT *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pscopy_( const MKL_INT *n, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdcopy_( const MKL_INT *n, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pccopy_( const MKL_INT *n, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzcopy_( const MKL_INT *n, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PSDOT( const MKL_INT *n, float *dot, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDDOT( const MKL_INT *n, double *dot, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSDOT_( const MKL_INT *n, float *dot, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDDOT_( const MKL_INT *n, double *dot, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void psdot_( const MKL_INT *n, float *dot, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pddot_( const MKL_INT *n, double *dot, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PCDOTC( const MKL_INT *n, MKL_Complex8 *dotc, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZDOTC( const MKL_INT *n, MKL_Complex16 *dotc, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCDOTC_( const MKL_INT *n, MKL_Complex8 *dotc, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZDOTC_( const MKL_INT *n, MKL_Complex16 *dotc, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcdotc_( const MKL_INT *n, MKL_Complex8 *dotc, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzdotc_( const MKL_INT *n, MKL_Complex16 *dotc, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PCDOTU( const MKL_INT *n, MKL_Complex8 *dotu, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZDOTU( const MKL_INT *n, MKL_Complex16 *dotu, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCDOTU_( const MKL_INT *n, MKL_Complex8 *dotu, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZDOTU_( const MKL_INT *n, MKL_Complex16 *dotu, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcdotu_( const MKL_INT *n, MKL_Complex8 *dotu, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzdotu_( const MKL_INT *n, MKL_Complex16 *dotu, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PSNRM2( const MKL_INT *n, float *norm2, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDNRM2( const MKL_INT *n, double *norm2, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PSCNRM2( const MKL_INT *n, float *norm2, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDZNRM2( const MKL_INT *n, double *norm2, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PSNRM2_( const MKL_INT *n, float *norm2, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDNRM2_( const MKL_INT *n, double *norm2, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PSCNRM2_( const MKL_INT *n, float *norm2, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDZNRM2_( const MKL_INT *n, double *norm2, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void psnrm2_( const MKL_INT *n, float *norm2, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdnrm2_( const MKL_INT *n, double *norm2, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pscnrm2_( const MKL_INT *n, float *norm2, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdznrm2_( const MKL_INT *n, double *norm2, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void PSSCAL( const MKL_INT *n, const float *a, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDSCAL( const MKL_INT *n, const double *a, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PCSCAL( const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZSCAL( const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PCSSCAL( const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZDSCAL( const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PSSCAL_( const MKL_INT *n, const float *a, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDSCAL_( const MKL_INT *n, const double *a, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PCSCAL_( const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZSCAL_( const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void PCSSCAL_( const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZDSCAL_( const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void psscal_( const MKL_INT *n, const float *a, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdscal_( const MKL_INT *n, const double *a, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pcscal_( const MKL_INT *n, const MKL_Complex8 *a, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pzscal_( const MKL_INT *n, const MKL_Complex16 *a, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pcsscal_( const MKL_INT *n, const float *a, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pzdscal_( const MKL_INT *n, const double *a, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void PSSWAP( const MKL_INT *n, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDSWAP( const MKL_INT *n, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCSWAP( const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZSWAP( const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSSWAP_( const MKL_INT *n, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDSWAP_( const MKL_INT *n, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCSWAP_( const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZSWAP_( const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void psswap_( const MKL_INT *n, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdswap_( const MKL_INT *n, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcswap_( const MKL_INT *n, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzswap_( const MKL_INT *n, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

/* PBLAS Level 2 Routines */

void PSGEMV( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDGEMV( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCGEMV( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZGEMV( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSGEMV_( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDGEMV_( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCGEMV_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZGEMV_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void psgemv_( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdgemv_( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcgemv_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzgemv_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PSAGEMV( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDAGEMV( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCAGEMV( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZAGEMV( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSAGEMV_( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDAGEMV_( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCAGEMV_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZAGEMV_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void psagemv_( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdagemv_( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcagemv_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzagemv_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PSGER( const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PDGER( const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PSGER_( const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PDGER_( const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void psger_( const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pdger_( const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void PCGERC( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PZGERC( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PCGERC_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PZGERC_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pcgerc_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pzgerc_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void PCGERU( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PZGERU( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PCGERU_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PZGERU_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pcgeru_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pzgeru_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void PCHEMV( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZHEMV( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCHEMV_( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZHEMV_( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pchemv_( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzhemv_( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PCAHEMV( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZAHEMV( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCAHEMV_( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZAHEMV_( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcahemv_( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzahemv_( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PCHER( const char *uplo, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PZHER( const char *uplo, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PCHER_( const char *uplo, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PZHER_( const char *uplo, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pcher_( const char *uplo, const MKL_INT *n, const float *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pzher_( const char *uplo, const MKL_INT *n, const double *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void PCHER2( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PZHER2( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PCHER2_( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PZHER2_( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pcher2_( const char *uplo, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pzher2_( const char *uplo, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void PSSYMV( const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDSYMV( const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSSYMV_( const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDSYMV_( const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pssymv_( const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdsymv_( const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PSASYMV( const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDASYMV( const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSASYMV_( const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDASYMV_( const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void psasymv_( const char *uplo, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdasymv_( const char *uplo, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PSSYR( const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PDSYR( const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PSSYR_( const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PDSYR_( const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pssyr_( const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pdsyr_( const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void PSSYR2( const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PDSYR2( const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PSSYR2_( const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void PDSYR2_( const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pssyr2_( const char *uplo, const MKL_INT *n, const float *alpha, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );
void pdsyr2_( const char *uplo, const MKL_INT *n, const double *alpha, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy, double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca );

void PSTRMV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDTRMV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PCTRMV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZTRMV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PSTRMV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDTRMV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PCTRMV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZTRMV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pstrmv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdtrmv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pctrmv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pztrmv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

void PSATRMV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDATRMV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCATRMV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZATRMV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PSATRMV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PDATRMV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PCATRMV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void PZATRMV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void psatrmv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const float *beta, float *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pdatrmv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const double *beta, double *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pcatrmv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex8 *beta, MKL_Complex8 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );
void pzatrmv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx, const MKL_Complex16 *beta, MKL_Complex16 *y, const MKL_INT *iy, const MKL_INT *jy, const MKL_INT *descy, const MKL_INT *incy );

void PSTRSV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDTRSV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PCTRSV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZTRSV( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PSTRSV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PDTRSV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PCTRSV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void PZTRSV_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pstrsv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pdtrsv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pctrsv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );
void pztrsv_( const char *uplo, const char *trans, const char *diag, const MKL_INT *n, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *x, const MKL_INT *ix, const MKL_INT *jx, const MKL_INT *descx, const MKL_INT *incx );

/* PBLAS Level 3 Routines  */

void PSGEMM( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDGEMM( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCGEMM( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZGEMM( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PSGEMM_( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDGEMM_( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCGEMM_( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZGEMM_( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void psgemm_( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdgemm_( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcgemm_( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzgemm_( const char *transa, const char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PCHEMM( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZHEMM( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCHEMM_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZHEMM_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pchemm_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzhemm_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PCHERK( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZHERK( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCHERK_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZHERK_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcherk_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzherk_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PCHER2K( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZHER2K( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCHER2K_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZHER2K_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcher2k_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzher2k_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PSSYMM( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDSYMM( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCSYMM( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZSYMM( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PSSYMM_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDSYMM_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCSYMM_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZSYMM_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pssymm_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdsymm_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcsymm_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzsymm_( const char *side, const char *uplo, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PSSYRK( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDSYRK( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCSYRK( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZSYRK( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PSSYRK_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDSYRK_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCSYRK_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZSYRK_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pssyrk_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdsyrk_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcsyrk_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzsyrk_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PSSYR2K( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDSYR2K( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PSSYR2K_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDSYR2K_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcsyr2k_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzsyr2k_( const char *uplo, const char *trans, const MKL_INT *n, const MKL_INT *k, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PSTRAN( const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDTRAN( const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PSTRAN_( const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDTRAN_( const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pstran_( const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdtran_( const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PCTRANU( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZTRANU( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCTRANU_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZTRANU_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pctranu_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pztranu_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PCTRANC( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZTRANC( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCTRANC_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZTRANC_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pctranc_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pztranc_( const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PSTRMM( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PDTRMM( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PCTRMM( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PZTRMM( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PSTRMM_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PDTRMM_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PCTRMM_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PZTRMM_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pstrmm_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pdtrmm_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pctrmm_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pztrmm_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );

void PSTRSM( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PDTRSM( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PCTRSM( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PZTRSM( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PSTRSM_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PDTRSM_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PCTRSM_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void PZTRSM_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pstrsm_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, float *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pdtrsm_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, double *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pctrsm_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex8 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );
void pztrsm_( const char *side, const char *uplo, const char *transa, const char *diag, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, MKL_Complex16 *b, const MKL_INT *ib, const MKL_INT *jb, const MKL_INT *descb );

void PSGEADD( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDGEADD( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCGEADD( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZGEADD( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PSGEADD_( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDGEADD_( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCGEADD_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZGEADD_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void psgeadd_( const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdgeadd_( const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pcgeadd_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pzgeadd_( const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

void PSTRADD( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDTRADD( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCTRADD( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZTRADD( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PSTRADD_( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PDTRADD_( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PCTRADD_( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void PZTRADD_( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pstradd_( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const float *alpha, const float *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const float *beta, float *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pdtradd_( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const double *alpha, const double *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const double *beta, double *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pctradd_( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex8 *alpha, const MKL_Complex8 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex8 *beta, MKL_Complex8 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );
void pztradd_( const char *uplo, const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_Complex16 *alpha, const MKL_Complex16 *a, const MKL_INT *ia, const MKL_INT *ja, const MKL_INT *desca, const MKL_Complex16 *beta, MKL_Complex16 *c, const MKL_INT *ic, const MKL_INT *jc, const MKL_INT *descc );

/* PBLAS Auxiliary Routines */

MKL_INT	PILAENV( const MKL_INT *ictxt, const char *prec );
MKL_INT	PILAENV_( const MKL_INT *ictxt, const char *prec );
MKL_INT	pilaenv_( const MKL_INT *ictxt, const char *prec );

#else
/* if defined _NETLIB_PBLAS_TYPES */

/* PBLAS Level 1 Routines */

void    psamax( MKL_INT *n, float *amax, MKL_INT *indx, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdamax( MKL_INT *n, double *amax, MKL_INT *indx, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pcamax( MKL_INT *n, float *amax, MKL_INT *indx, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pzamax( MKL_INT *n, double *amax, MKL_INT *indx, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    psasum( MKL_INT *n, float *asum, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdasum( MKL_INT *n, double *asum, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    pscasum( MKL_INT *n, float *asum, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdzasum( MKL_INT *n, double *asum, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    psaxpy( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdaxpy( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcaxpy( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzaxpy( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    picopy( MKL_INT *n, MKL_INT *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, MKL_INT *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pscopy( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdcopy( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pccopy( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzcopy( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    psdot( MKL_INT *n, float *dot, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pddot( MKL_INT *n, double *dot, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    pcdotc( MKL_INT *n, float *dotc, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzdotc( MKL_INT *n, double *dotc, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    pcdotu( MKL_INT *n, float *dotu, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzdotu( MKL_INT *n, double *dotu, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    psnrm2( MKL_INT *n, float *norm2, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdnrm2( MKL_INT *n, double *norm2, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pscnrm2( MKL_INT *n, float *norm2, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdznrm2( MKL_INT *n, double *norm2, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    psscal( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdscal( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pcscal( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pzscal( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pcsscal( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pzdscal( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    psswap( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdswap( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcswap( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzswap( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );


/* PBLAS Level 2 Routines */

void    psgemv( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdgemv( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcgemv( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzgemv( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    psagemv( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdagemv( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcagemv( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzagemv( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    psger( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pdger( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    pcgerc( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pzgerc( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    pcgeru( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pzgeru( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    pchemv( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzhemv( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    pcahemv( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzahemv( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    pcher( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pzher( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    pcher2( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pzher2( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    pssymv( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdsymv( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    psasymv( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdasymv( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    pssyr( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pdsyr( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    pssyr2( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pdsyr2( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    pstrmv( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdtrmv( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pctrmv( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pztrmv( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    psatrmv( char *uplo, char *trans, char *diag, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdatrmv( char *uplo, char *trans, char *diag, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcatrmv( char *uplo, char *trans, char *diag, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzatrmv( char *uplo, char *trans, char *diag, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    pstrsv( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdtrsv( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pctrsv( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pztrsv( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );


/* PBLAS Level 3 Routines */

void    psgemm( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdgemm( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcgemm( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzgemm( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pchemm( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzhemm( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pcherk( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzherk( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pcher2k( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzher2k( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pssymm( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdsymm( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcsymm( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzsymm( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pssyrk( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdsyrk( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcsyrk( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzsyrk( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pssyr2k( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdsyr2k( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcsyr2k( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzsyr2k( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pstran( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdtran( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pctranu( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pztranu( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pctranc( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pztranc( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pstrmm( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pdtrmm( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pctrmm( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pztrmm( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );

void    pstrsm( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pdtrsm( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pctrsm( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pztrsm( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );

void    psgeadd( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdgeadd( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcgeadd( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzgeadd( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    pstradd( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdtradd( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pctradd( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pztradd( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

/* PBLAS Auxiliary Routines */

MKL_INT	pilaenv( MKL_INT *ictxt, char *prec );

/* OTHER NAMING CONVENSIONS FOLLOW */

/* PBLAS Level 1 Routines */

void    PSAMAX( MKL_INT *n, float *amax, MKL_INT *indx, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDAMAX( MKL_INT *n, double *amax, MKL_INT *indx, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCAMAX( MKL_INT *n, float *amax, MKL_INT *indx, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZAMAX( MKL_INT *n, double *amax, MKL_INT *indx, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PSAMAX_( MKL_INT *n, float *amax, MKL_INT *indx, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDAMAX_( MKL_INT *n, double *amax, MKL_INT *indx, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCAMAX_( MKL_INT *n, float *amax, MKL_INT *indx, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZAMAX_( MKL_INT *n, double *amax, MKL_INT *indx, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    psamax_( MKL_INT *n, float *amax, MKL_INT *indx, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdamax_( MKL_INT *n, double *amax, MKL_INT *indx, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pcamax_( MKL_INT *n, float *amax, MKL_INT *indx, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pzamax_( MKL_INT *n, double *amax, MKL_INT *indx, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    PSASUM( MKL_INT *n, float *asum, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDASUM( MKL_INT *n, double *asum, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PSASUM_( MKL_INT *n, float *asum, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDASUM_( MKL_INT *n, double *asum, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    psasum_( MKL_INT *n, float *asum, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdasum_( MKL_INT *n, double *asum, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    PSCASUM( MKL_INT *n, float *asum, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDZASUM( MKL_INT *n, double *asum, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PSCASUM_( MKL_INT *n, float *asum, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDZASUM_( MKL_INT *n, double *asum, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pscasum_( MKL_INT *n, float *asum, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdzasum_( MKL_INT *n, double *asum, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    PSAXPY( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDAXPY( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCAXPY( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZAXPY( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSAXPY_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDAXPY_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCAXPY_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZAXPY_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    psaxpy_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdaxpy_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcaxpy_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzaxpy_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PICOPY( MKL_INT *n, MKL_INT *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, MKL_INT *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSCOPY( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDCOPY( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCCOPY( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZCOPY( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PICOPY_( MKL_INT *n, MKL_INT *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, MKL_INT *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSCOPY_( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDCOPY_( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCCOPY_( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZCOPY_( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    picopy_( MKL_INT *n, MKL_INT *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, MKL_INT *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pscopy_( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdcopy_( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pccopy_( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzcopy_( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PSDOT( MKL_INT *n, float *dot, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDDOT( MKL_INT *n, double *dot, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSDOT_( MKL_INT *n, float *dot, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDDOT_( MKL_INT *n, double *dot, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    psdot_( MKL_INT *n, float *dot, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pddot_( MKL_INT *n, double *dot, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PCDOTC( MKL_INT *n, float *dotc, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZDOTC( MKL_INT *n, double *dotc, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCDOTC_( MKL_INT *n, float *dotc, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZDOTC_( MKL_INT *n, double *dotc, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcdotc_( MKL_INT *n, float *dotc, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzdotc_( MKL_INT *n, double *dotc, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PCDOTU( MKL_INT *n, float *dotu, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZDOTU( MKL_INT *n, double *dotu, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCDOTU_( MKL_INT *n, float *dotu, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZDOTU_( MKL_INT *n, double *dotu, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcdotu_( MKL_INT *n, float *dotu, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzdotu_( MKL_INT *n, double *dotu, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PSNRM2( MKL_INT *n, float *norm2, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDNRM2( MKL_INT *n, double *norm2, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PSCNRM2( MKL_INT *n, float *norm2, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDZNRM2( MKL_INT *n, double *norm2, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PSNRM2_( MKL_INT *n, float *norm2, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDNRM2_( MKL_INT *n, double *norm2, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PSCNRM2_( MKL_INT *n, float *norm2, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDZNRM2_( MKL_INT *n, double *norm2, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    psnrm2_( MKL_INT *n, float *norm2, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdnrm2_( MKL_INT *n, double *norm2, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pscnrm2_( MKL_INT *n, float *norm2, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdznrm2_( MKL_INT *n, double *norm2, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    PSSCAL( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDSCAL( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCSCAL( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZSCAL( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCSSCAL( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZDSCAL( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PSSCAL_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDSCAL_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCSCAL_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZSCAL_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCSSCAL_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZDSCAL_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    psscal_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdscal_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pcscal_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pzscal_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pcsscal_( MKL_INT *n, float *a, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pzdscal_( MKL_INT *n, double *a, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    PSSWAP( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDSWAP( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCSWAP( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZSWAP( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSSWAP_( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDSWAP_( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCSWAP_( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZSWAP_( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    psswap_( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdswap_( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcswap_( MKL_INT *n, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzswap_( MKL_INT *n, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );


/* PBLAS Level 2 Routines */

void    PSGEMV( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDGEMV( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCGEMV( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZGEMV( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSGEMV_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDGEMV_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCGEMV_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZGEMV_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    psgemv_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdgemv_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcgemv_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzgemv_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PSAGEMV( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDAGEMV( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCAGEMV( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZAGEMV( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSAGEMV_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDAGEMV_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCAGEMV_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZAGEMV_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    psagemv_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdagemv_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcagemv_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzagemv_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PSGER( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PDGER( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PSGER_( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PDGER_( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    psger_( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pdger_( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    PCGERC( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PZGERC( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PCGERC_( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PZGERC_( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pcgerc_( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pzgerc_( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    PCGERU( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PZGERU( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PCGERU_( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PZGERU_( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pcgeru_( MKL_INT *m, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pzgeru_( MKL_INT *m, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    PCHEMV( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZHEMV( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCHEMV_( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZHEMV_( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pchemv_( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzhemv_( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PCAHEMV( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZAHEMV( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCAHEMV_( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZAHEMV_( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcahemv_( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzahemv_( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PCHER( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PZHER( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PCHER_( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PZHER_( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pcher_( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pzher_( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    PCHER2( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PZHER2( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PCHER2_( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PZHER2_( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pcher2_( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pzher2_( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    PSSYMV( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDSYMV( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSSYMV_( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDSYMV_( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pssymv_( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdsymv_( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PSASYMV( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDASYMV( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSASYMV_( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDASYMV_( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    psasymv_( char *uplo, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdasymv_( char *uplo, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PSSYR( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PDSYR( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PSSYR_( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PDSYR_( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pssyr_( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pdsyr_( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    PSSYR2( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PDSYR2( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PSSYR2_( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    PDSYR2_( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pssyr2_( char *uplo, MKL_INT *n, float *alpha, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );
void    pdsyr2_( char *uplo, MKL_INT *n, double *alpha, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca );

void    PSTRMV( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDTRMV( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCTRMV( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZTRMV( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PSTRMV_( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDTRMV_( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCTRMV_( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZTRMV_( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pstrmv_( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdtrmv_( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pctrmv_( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pztrmv_( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );

void    PSATRMV( char *uplo, char *trans, char *diag, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDATRMV( char *uplo, char *trans, char *diag, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCATRMV( char *uplo, char *trans, char *diag, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZATRMV( char *uplo, char *trans, char *diag, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PSATRMV_( char *uplo, char *trans, char *diag, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PDATRMV_( char *uplo, char *trans, char *diag, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PCATRMV_( char *uplo, char *trans, char *diag, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    PZATRMV_( char *uplo, char *trans, char *diag, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    psatrmv_( char *uplo, char *trans, char *diag, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pdatrmv_( char *uplo, char *trans, char *diag, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pcatrmv_( char *uplo, char *trans, char *diag, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, float *beta, float *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );
void    pzatrmv_( char *uplo, char *trans, char *diag, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx, double *beta, double *y, MKL_INT *iy, MKL_INT *jy, MKL_INT *descy, MKL_INT *incy );

void    PSTRSV( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDTRSV( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCTRSV( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZTRSV( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PSTRSV_( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PDTRSV_( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PCTRSV_( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    PZTRSV_( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pstrsv_( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pdtrsv_( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pctrsv_( char *uplo, char *trans, char *diag, MKL_INT *n, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );
void    pztrsv_( char *uplo, char *trans, char *diag, MKL_INT *n, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *x, MKL_INT *ix, MKL_INT *jx, MKL_INT *descx, MKL_INT *incx );


/* PBLAS Level 3 Routines */

void    PSGEMM( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDGEMM( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCGEMM( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZGEMM( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PSGEMM_( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDGEMM_( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCGEMM_( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZGEMM_( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    psgemm_( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdgemm_( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcgemm_( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzgemm_( char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PCHEMM( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZHEMM( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCHEMM_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZHEMM_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pchemm_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzhemm_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PCHERK( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZHERK( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCHERK_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZHERK_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcherk_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzherk_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PCHER2K( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZHER2K( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCHER2K_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZHER2K_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcher2k_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzher2k_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PSSYMM( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDSYMM( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCSYMM( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZSYMM( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PSSYMM_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDSYMM_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCSYMM_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZSYMM_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pssymm_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdsymm_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcsymm_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzsymm_( char *side, char *uplo, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PSSYRK( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDSYRK( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCSYRK( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZSYRK( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PSSYRK_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDSYRK_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCSYRK_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZSYRK_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pssyrk_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdsyrk_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcsyrk_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzsyrk_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PSSYR2K( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDSYR2K( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PSSYR2K_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDSYR2K_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcsyr2k_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzsyr2k_( char *uplo, char *trans, MKL_INT *n, MKL_INT *k, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PSTRAN( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDTRAN( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PSTRAN_( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDTRAN_( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pstran_( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdtran_( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PCTRANU( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZTRANU( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCTRANU_( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZTRANU_( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pctranu_( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pztranu_( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PCTRANC( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZTRANC( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCTRANC_( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZTRANC_( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pctranc_( MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pztranc_( MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PSTRMM( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PDTRMM( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PCTRMM( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PZTRMM( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PSTRMM_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PDTRMM_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PCTRMM_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PZTRMM_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pstrmm_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pdtrmm_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pctrmm_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pztrmm_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );

void    PSTRSM( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PDTRSM( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PCTRSM( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PZTRSM( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PSTRSM_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PDTRSM_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PCTRSM_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    PZTRSM_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pstrsm_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pdtrsm_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pctrsm_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );
void    pztrsm_( char *side, char *uplo, char *transa, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *b, MKL_INT *ib, MKL_INT *jb, MKL_INT *descb );

void    PSGEADD( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDGEADD( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCGEADD( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZGEADD( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PSGEADD_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDGEADD_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCGEADD_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZGEADD_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    psgeadd_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdgeadd_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pcgeadd_( char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pzgeadd_( char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

void    PSTRADD( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDTRADD( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCTRADD( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZTRADD( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PSTRADD_( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PDTRADD_( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PCTRADD_( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    PZTRADD_( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pstradd_( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pdtradd_( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pctradd_( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, float *alpha, float *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, float *beta, float *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );
void    pztradd_( char *uplo, char *trans, MKL_INT *m, MKL_INT *n, double *alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc );

/* PBLAS Auxiliary Routines */

MKL_INT	PILAENV( MKL_INT *ictxt, char *prec );
MKL_INT	PILAENV_( MKL_INT *ictxt, char *prec );
MKL_INT	pilaenv_( MKL_INT *ictxt, char *prec );

#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_PBLAS_H_ */
