#ifdef USE_OPENBLAS
#include <cblas.h>
#include <openblas_config.h>
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

int test_cblas_snrm2();

int test_cblas_sdot();

// A temporary test just to play with OpenBLAS
int main(int argc, char **argv)
{
#ifdef USE_OPENBLAS
    printf("Using Open BLAS.... \n\n");
#else
    printf("Using Intel MKL.... \n\n");
#endif

    auto errorCode = test_cblas_snrm2();
    if (errorCode == 0)
    {
        errorCode = test_cblas_sdot();
    }

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
        printf("Validation FAILED :( \n");
        return 1;
    }
#else
    printf("cblas_snrm2 result: %f \n\n", result);
#endif

    printf("Completed\n-------------------------\n");
    return 0;
}

// NOTE: it seems that cblas_sdot of an exactly identical vectors throws an Exception with openBLAS but not with MKL...
int test_cblas_sdot()
{
    printf("Testing test_cblas_sdot... \n");

    std::vector<float> vectorA{1.4, 2.6, 3.7, 0.45, 12, 100.3};
    std::vector<float> vectorB{201.5, 83, 56.0, 2, 0, 89.5};

    float result = cblas_sdot((Flex_INT)vectorA.size(), vectorA.data(), (Flex_INT)1, vectorB.data(), (Flex_INT)1);

#ifdef USE_OPENBLAS
    // Expected result from intelMKL: 9682.849609
    if (std::fabs(result - 9682.849609) > 1.0e-4f)
    {
        printf("OPEN BLAS value (%f) is not matching with Intel MKL value (9682.849609)... \n\n", result);
        printf("Validation FAILED :( \n");
        return 1;
    }
#else
    printf("cblas_sdot result: %f \n\n", result);
#endif

    printf("Completed\n-------------------------\n");
    return 0;
}
