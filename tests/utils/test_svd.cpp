#include <stdlib.h>
#include <stdio.h>
#include "mkl_lapacke.h"

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, MKL_INT m, MKL_INT n, float* a, MKL_INT lda );

/* Parameters */
#define M 6
#define N 4
#define LDA N
#define LDU M
#define LDVT N

/* Main program */
int main() {
        /* Locals */
        MKL_INT m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info;
        /* Local arrays */
        float s[N], u[LDU*M], vt[LDVT*N];
        float a[LDA*M] = {
            7.52f, -1.10f, -7.95f, 1.08f,
           -0.76f,  0.62f,  9.34f, -7.10f,
            5.13f,  6.62f, -5.66f, 0.87f,
           -4.75f,  8.52f,  5.75f, 5.30f,
            1.33f,  4.91f, -5.49f, -3.52f,
           -2.40f, -6.77f,  2.34f, 3.95f
        };
        /* Executable statements */
        printf( "LAPACKE_sgesdd (row-major, high-level) Example Program Results\n" );
        /* Compute SVD */
        info = LAPACKE_sgesdd( LAPACK_ROW_MAJOR, 'S', m, n, a, lda, s,
                        u, ldu, vt, ldvt );
        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm computing SVD failed to converge.\n" );
                exit( 1 );
        }
        /* Print singular values */
        print_matrix( "Singular values", 1, n, s, 1 );
        /* Print left singular vectors */
        print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
        /* Print right singular vectors */
        print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
        exit( 0 );
} /* End of LAPACKE_sgesdd Example */

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, MKL_INT m, MKL_INT n, float* a, MKL_INT lda ) {
        MKL_INT i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i*lda+j] );
                printf( "\n" );
        }
}