// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#ifdef USE_MKL_BLAS

#include "mkl.h"
typedef MKL_INT BLAS_INT;
#define blas_set_num_threads(num_threads) mkl_set_num_threads(num_threads)

#endif


#ifdef USE_OPENBLAS_BLAS

#include "cblas.h"
typedef MKL_INT int;
#define blas_set_num_threads(num_threads) openblas_set_num_threads(num_threads);

#endif
