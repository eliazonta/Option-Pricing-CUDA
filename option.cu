#include <stdio.h>

#include <cufft.h>

#include "parameters.h"

const int N = 16;
const int blocksize = 16;

#define NX 64
#define NY 64
#define NZ 128

__global__
void hello(char *a, int *b)
{
    a[threadIdx.x] += b[threadIdx.x];
}

/*

computeGPU()
{

}

*/

int main()
{
    Parameters params;

    char a[N] = "Hello \0\0\0\0\0\0";
    int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    char *ad;
    int *bd;
    const int csize = N*sizeof(char);
    const int isize = N*sizeof(int);

    printf("%s", a);

    cudaMalloc( (void**)&ad, csize );
    cudaMalloc( (void**)&bd, isize );
    cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
    cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( 1, 1 );
    hello<<<dimGrid, dimBlock>>>(ad, bd);
    cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
    cudaFree( ad );
    cudaFree( bd );

    printf("%s\n", a);

    cufftHandle plan;
    cufftComplex *data1, *data2;
    cudaMalloc((void**)&data1, sizeof(cufftComplex)*NX*NY*NZ);
    cudaMalloc((void**)&data2, sizeof(cufftComplex)*NX*NY*NZ);
    // Create a 3D FFT plan.
    cufftPlan3d(&plan, NX, NY, NZ, CUFFT_C2C);

    // Transform the first signal in place.
    cufftExecC2C(plan, data1, data1, CUFFT_FORWARD);

    // Transform the second signal using the same plan.
    cufftExecC2C(plan, data2, data2, CUFFT_FORWARD);

    // Destroy the cuFFT plan.
    cufftDestroy(plan);
    cudaFree(data1); cudaFree(data2);

    return EXIT_SUCCESS;
}

