#include <cblas.h>
#include <stdio.h>
#include <vector>
#include <chrono>

// A temporary test just to play with OpenBLAS
// Source of the sample 
int main(int argc, char **argv)
{
    printf("Compute alpha*A*B+beta*C using OpenBLAS cblas_dgemm \n");

    int size = 100;
    int m = size, k = size, n = size;
    int lda = size, ldb = size, ldc = size;
    double alpha = 1.0, beta = 2.0;

    std::vector<double> A(m * k);
    std::vector<double> B(k * n);
    std::vector<double> C(m * n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);

    printf("Completed Successfully. \n");

    return 0;
}
