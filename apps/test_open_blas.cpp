#ifdef USE_OPENBLAS
#include <cblas.h>
#include <openblas_config.h>
#else
#include <mkl.h>
#endif

#include <stdio.h>
#include <vector>
#include <cmath>

int test_cblas_snrm2();

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
    std::vector<float> vectorA{1.4, 2.6, 3.7, 0.45, 12, 100.3};

#ifdef USE_OPENBLAS
    float result = cblas_snrm2((blasint)vectorA.size(), vectorA.data(), (blasint)1);
#else
    float result = cblas_snrm2((MKL_INT)vectorA.size(), vectorA.data(), (MKL_INT)1);
#endif

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

    return 0;
}
