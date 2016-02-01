#include <assert.h>
#include <stdio.h>
#include <vector>

#include <cufft.h>

#include "parameters.h"

using namespace std;

// Copied from docs
// CUFFT_SUCCESS = 0, // The cuFFT operation was successful
// CUFFT_INVALID_PLAN = 1, // cuFFT was passed an invalid plan handle
// CUFFT_ALLOC_FAILED = 2, // cuFFT failed to allocate GPU or CPU memory
// CUFFT_INVALID_TYPE = 3, // No longer used
// CUFFT_INVALID_VALUE = 4, // User specified an invalid pointer or parameter
// CUFFT_INTERNAL_ERROR = 5, // Driver or internal cuFFT library error
// CUFFT_EXEC_FAILED = 6, // Failed to execute an FFT on the GPU
// CUFFT_SETUP_FAILED = 7, // The cuFFT library failed to initialize
// CUFFT_INVALID_SIZE = 8, // User specified an invalid transform size
// CUFFT_UNALIGNED_DATA = 9, // No longer used
// CUFFT_INCOMPLETE_PARAMETER_LIST = 10, // Missing parameters in call
// CUFFT_INVALID_DEVICE = 11, // Execution of a plan was on different GPU than plan creation
// CUFFT_PARSE_ERROR = 12, // Internal plan database error
// CUFFT_NO_WORKSPACE = 13 // No workspace has been provided prior to plan execution
#define checkCufft(result) do {           \
    if (result != CUFFT_SUCCESS) {                      \
        fprintf(stderr, "CUFFT at %d error: %d\n", __LINE__, result);   \
        exit(-1);                                       \
    }                                                   \
} while(0)

#define checkCuda(result) do {            \
    if (result != cudaSuccess) {                        \
        fprintf(stderr, "CUDA at %d error: %d\n", __LINE__, result);   \
        exit(-1);                                       \
    }                                                   \
} while(0)

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

vector<double> pricesAtPayoff(Parameters& prms)
{
    vector<double> out(prms.resolution);

    // Tree parameters (see p.53 of notes).
    double u = exp(prms.volatility * sqrt(prms.timeIncrement));
    double d = 1.0 / u;
    double a = exp(prms.riskFreeRate * prms.timeIncrement);
    // double p = (a - d) / (u - d);

    double N = prms.resolution;
    for (int i = 0; i < N; i++) {
        double asset = prms.startPrice * pow(u, i) * pow(d, N - i);
        if (prms.optionType == Call) {
            out[i] = max(asset - prms.strikePrice, 0.0);
        } else {
            out[i] = max(prms.strikePrice - asset, 0.0);
        }
    }

    return out;
}

// Prints Hello, World if the GPU code is working right.
void helloWorld()
{
    const int N = 16;
    const int blocksize = 16;

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
}

void printPrices(vector<double>& prices) {
    for (int i = 0; i < prices.size(); i++) {
        printf("%f ", prices[i]);
    }
    printf("\n");
}

int main()
{
    assert(sizeof(cufftDoubleReal) == sizeof(double));
    assert(sizeof(cufftDoubleComplex) == 2 * sizeof(double));

    helloWorld();

    Parameters params;
    vector<double> prices = pricesAtPayoff(params);

    printPrices(prices);

    double N = params.resolution;

    cufftDoubleReal* d_prices;
    checkCuda(cudaMalloc((void**)&d_prices, sizeof(cufftDoubleReal) * N));
    checkCuda(cudaMemcpy(d_prices, &prices[0], sizeof(cufftDoubleReal) * N,
                         cudaMemcpyHostToDevice));

    cufftDoubleComplex* d_ft;
    checkCuda(cudaMalloc((void**)&d_ft, sizeof(cufftDoubleComplex) * N));

    cufftHandle plan;
    // Double to double-complex interleaved
    checkCufft(cufftPlan1d(&plan, N, CUFFT_D2Z, /* deprecated? */ 1));
    //checkCufft(cufftPlan3d(&plan, 5, 5, 5, CUFFT_C2C /* deprecated? */));

    // Forward transform
    checkCufft(cufftExecD2Z(plan, d_prices, d_ft));

    // Reverse transform
    checkCufft(cufftExecZ2D(plan, d_ft, d_prices));

    checkCuda(cudaMemcpy(d_prices, &prices[0], sizeof(cufftDoubleReal) * N,
                         cudaMemcpyDeviceToHost));
    printPrices(prices);

    // Destroy the cuFFT plan.
    cufftDestroy(plan);
    cudaFree(d_prices);
    cudaFree(d_ft);

    return EXIT_SUCCESS;
}

