// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cblas.h>
#include <stdio.h>
#include <vector>
#include <chrono>

// A temporary test just to play with OpenBLAS
int main(int argc, char **argv)
{
    int m, n, k;
    int lda, ldb, ldc;
    double alpha, beta;
    int loops;

    std::chrono::high_resolution_clock::time_point start{};
    std::chrono::high_resolution_clock::time_point stop{};
    std::chrono::duration<double, std::milli> elapsed_cpu_ms{};

    printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
           " OpenBLAS dgemm, where A, B, and  C are matrices and \n"
           " alpha and beta are double precision scalars\n\n");

    int size = 4092;
    m = size, k = size, n = size;
    lda = size, ldb = size, ldc = size;
    alpha = 1.0, beta = 0.0;
    loops = 1;

    printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
           " A(%ix%i) and matrix B(%ix%i)\n\n",
           m, k, k, n);

    start = std::chrono::high_resolution_clock::now();
    printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
           " performance \n\n");
    std::vector<double> A(m * k);
    std::vector<double> B(k * n);
    std::vector<double> C(m * n);

    for (int l = 0; l < loops; l++)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta,
                    C.data(), ldc);

    stop = std::chrono::high_resolution_clock::now();
    elapsed_cpu_ms = stop - start;

    printf(" Time Elapsed: %0.2f ms \n\n", elapsed_cpu_ms.count() / loops);
    printf(" Example completed. \n\n");

    return 0;
}
