#pragma once

#if defined(__MIC__) || defined(__TARGET_ARCH_MIC)
#define MKL_CALL_CONV
#else
#if defined(MKL_STDCALL)
#define MKL_CALL_CONV __stdcall
#else
#define MKL_CALL_CONV __cdecl
#endif
#endif

#define _Mkl_Api(rtype, name, arg) extern rtype MKL_CALL_CONV name arg;
#define _mkl_api(rtype, name, arg) extern rtype MKL_CALL_CONV name arg;
#define _MKL_API(rtype, name, arg) extern rtype MKL_CALL_CONV name arg;

#include "mkl_cblas.h"
