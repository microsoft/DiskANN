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
!      Intel(R) Math Kernel Library (Intel(R) MKL) interface
!******************************************************************************/

#ifndef _MKL_H_
#define _MKL_H_

#if defined(__MIC__) || defined(__TARGET_ARCH_MIC)
#   define MKL_CALL_CONV
#else
#   if defined(MKL_STDCALL)
#       define MKL_CALL_CONV __stdcall
#   else
#       define MKL_CALL_CONV __cdecl
#   endif
#endif

#define _Mkl_Api(rtype,name,arg)   extern rtype MKL_CALL_CONV   name    arg;
#define _mkl_api(rtype,name,arg)   extern rtype MKL_CALL_CONV   name    arg;
#define _MKL_API(rtype,name,arg)   extern rtype MKL_CALL_CONV   name    arg;

#include "mkl_version.h"
#include "mkl_types.h"
#include "mkl_blas.h"
#include "mkl_trans.h"
#include "mkl_cblas.h"
#include "mkl_spblas.h"
#include "mkl_lapack.h"
#include "mkl_lapacke.h"
#include "mkl_pardiso.h"
#include "mkl_sparse_handle.h"
#include "mkl_dss.h"
#include "mkl_rci.h"
#include "mkl_vml.h"
#include "mkl_vsl.h"
#include "mkl_df.h"
#include "mkl_service.h"
#include "mkl_dfti.h"
#include "mkl_trig_transforms.h"
#include "mkl_poisson.h"
#include "mkl_solvers_ee.h"
#include "mkl_direct_call.h"
#include "mkl_dnn.h"
#include "mkl_compact.h"

#endif /* _MKL_H_ */
