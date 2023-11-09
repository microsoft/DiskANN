#ifdef USE_OPENBLAS
#include <cblas.h>
#include <openblas_config.h>
// #include<lapacke.h> //TODO: commented out for now, until we fix the error in header.
#else
#include <mkl.h>
#endif

#include <stdio.h>
#include <vector>
#include <cmath>

#ifdef USE_OPENBLAS
using Flex_INT = blasint;
#else
using Flex_INT = MKL_INT;
#endif

#ifdef USE_OPENBLAS
using Flex_CBLAS_ORDER = CBLAS_ORDER;
#else
using Flex_CBLAS_ORDER = CBLAS_LAYOUT;
#endif

#ifdef USE_OPENBLAS
using Flex_CBLAS_TRANSPOSE = CBLAS_TRANSPOSE;
#else
using Flex_CBLAS_TRANSPOSE = CBLAS_TRANSPOSE;
#endif

#ifdef USE_OPENBLAS
using Flex_LAPACK_INT = int;
#else
using Flex_LAPACK_INT = MKL_INT;
#endif

int test_cblas_snrm2();
int test_cblas_sdot();
int test_cblas_sgemm();
int test_LAPACKE_sgesdd();

// A temporary test just to play with OpenBLAS
int main(int argc, char **argv)
{
#ifdef USE_OPENBLAS
    printf("Using Open BLAS.... \n\n");
#else
    printf("Using Intel MKL.... \n\n");
#endif

    auto errorCode = test_cblas_snrm2();
    errorCode += test_cblas_sdot();
    errorCode += test_cblas_sgemm();
    errorCode += test_LAPACKE_sgesdd();

    if (errorCode == 0)
    {
        printf("\n Completed Successfully. \n");
    }
    else
    {
        printf("\n Completed With ERRORs. \n");
    }

    return errorCode;
}

int test_cblas_snrm2()
{
    printf("Testing test_cblas_snrm2... \n");
    std::vector<float> vectorA{1.4, 2.6, 3.7, 0.45, 12, 100.3};

    float result = cblas_snrm2((Flex_INT)vectorA.size(), vectorA.data(), (Flex_INT)1);

#ifdef USE_OPENBLAS
    // Expected result from intelMKL: 101.127167
    if (std::fabs(result - 101.127167) > 1.0e-4f)
    {
        printf("OPEN BLAS value (%f) is not matching with Intel MKL value (101.127167)... \n\n", result);
        printf("Validation FAILED :( \n-------------------------\n");
        return 1;
    }
#else
    printf("cblas_snrm2 result: %f \n\n", result);
#endif

    printf("Completed\n-------------------------\n");
    return 0;
}

// NOTE: it seems that cblas_sdot of an exactly identical vectors throws an Exception with openBLAS but not with MKL...
// NOTE:  OpenBLAS value (9682.850586) is not very close to Intel MKL value (9682.849609).
int test_cblas_sdot()
{
    printf("Testing test_cblas_sdot... \n");

    std::vector<float> vectorA{1.4, 2.6, 3.7, 0.45, 12, 100.3};
    std::vector<float> vectorB{201.5, 83, 56.0, 2, 0, 89.5};

    float result = cblas_sdot((Flex_INT)vectorA.size(), vectorA.data(), (Flex_INT)1, vectorB.data(), (Flex_INT)1);

#ifdef USE_OPENBLAS
    // Expected result from intelMKL: 9682.849609
    if (std::fabs(result - 9682.849609) > 1.0e-1f)
    {
        printf("OPEN BLAS value (%f) is not matching with Intel MKL value (9682.849609)... \n\n", result);
        printf("Validation FAILED :( \n-------------------------\n");
        return 1;
    }
#else
    printf("cblas_sdot result: %f \n\n", result);
#endif

    printf("Completed\n-------------------------\n");
    return 0;
}

int test_cblas_sgemm()
{
    printf("Testing test_cblas_sgemm... \n");

    Flex_INT size = 3;
    Flex_INT m = size, k = size, n = size;
    Flex_INT lda = size, ldb = size, ldc = size;
    float alpha = 1.0, beta = 2.0;

    const std::vector<float> A(m * k, 1.0);
    const std::vector<float> B(k * n, 2.0);
    std::vector<float> C(m * n);

    cblas_sgemm(Flex_CBLAS_ORDER::CblasRowMajor, Flex_CBLAS_TRANSPOSE::CblasNoTrans, Flex_CBLAS_TRANSPOSE::CblasNoTrans,
                (Flex_INT)m, (Flex_INT)n, (Flex_INT)k, alpha, A.data(), (Flex_INT)lda, B.data(), (Flex_INT)ldb, beta,
                C.data(), (Flex_INT)ldc);

#ifdef USE_OPENBLAS
    // Expected result from intelMKL: all the values should be 6.0
    for (auto val : C)
    {
        if (std::fabs(val - 6.0) > 1.0e-4f)
        {
            printf("OPEN BLAS value (%f) is not matching with Intel MKL value (6.0)... \n\n", val);
            printf("Validation FAILED :( \n-------------------------\n");
            return 1;
        }
    }

#else
    printf("test_cblas_sgemm result:\n");
    for (auto val : C)
    {
        printf("%f, ", val);
    }
    printf("\n\n");
#endif

    printf("Completed\n-------------------------\n");
    return 0;
}

int test_LAPACKE_sgesdd()
{
    printf("Testing test_LAPACKE_sgesdd... \n");

    Flex_LAPACK_INT size = 3;
    Flex_LAPACK_INT m = size, k = size, n = size;
    Flex_LAPACK_INT lda = size, ldu = size, ldvt = size;

    std::vector<float> A(m * k, 1.0);
    std::vector<float> S(k * n, 2.0);
    std::vector<float> U(m * n);
    std::vector<float> VT(m * n);

#ifndef USE_OPENBLAS // TODO: this won't work for OpenBLas until we fix the lapack header
    uint32_t errcode = (uint32_t)LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', (Flex_LAPACK_INT)m, (Flex_LAPACK_INT)n, A.data(),
                                                (Flex_LAPACK_INT)lda, S.data(), U.data(), (Flex_LAPACK_INT)ldu,
                                                VT.data(), (Flex_LAPACK_INT)ldvt);

#ifdef USE_OPENBLAS
    // Expected result from intelMKL: VT: -0.577350, -0.577350, -0.577350, 0.816497, -0.408248, -0.408248, 0.000000,
    // -0.707107, 0.707107,
    std::vector<float> expectedVT{-0.577350, -0.577350, -0.577350, 0.816497, -0.408248,
                                  -0.408248, 0.000000,  -0.707107, 0.707107};
    for (size_t i = 0; i < VT.size(); i++)
    {
        if (std::fabs(VT[i] - expectedVT[i]) > 1.0e-4f)
        {
            printf("OPEN BLAS value (%f) is not matching with Intel MKL value (%f)... \n\n", VT[i], expectedVT[i]);
            printf("Validation FAILED :( \n-------------------------\n");
            return 1;
        }
    }
#else
    printf("test_cblas_sgemm result:\n");
    for (auto val : VT)
    {
        printf("%f, ", val);
    }
    printf("\n\n");
#endif
#endif
    printf("Completed\n-------------------------\n");
    return 0;
}
