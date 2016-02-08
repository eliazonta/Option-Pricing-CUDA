#include <assert.h>
#include <math_constants.h>
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

__host__ __device__ static __inline__
cufftComplex cuComplexExponential(cufftComplex x)
{
    float a = cuCrealf(x);
    float b = cuCrealf(x);
    float ea = exp(a);
    return make_cuComplex(ea * cos(b), ea * sin(b));
}

__global__
void hello(char *a, int *b)
{
    a[threadIdx.x] += b[threadIdx.x];
}

__global__
void solveODE(cufftComplex* ft,
              float from_time,         // τ_l (T - t_l)
              float to_time,           // τ_u (T - t_u)
              float riskFreeRate, float volatility,
              float jumpMean, float kappa)
{
    int idx = threadIdx.x;

    cufftComplex old_value = ft[idx];

    // Frequency.
    float k = 0.0;

    // Calculate Ψ (psi) (2.14)
    // Equation slightly simplified to save a few operations.
    float fst_term = volatility * M_PI * k;
    float psi_real = (-2.0 * fst_term * fst_term) - (riskFreeRate + jumpMean);
    float psi_imag = (riskFreeRate - jumpMean * kappa - volatility * volatility / 2.0) *
                      (2 * M_PI * k);

    // TODO: jump component.

    // Solution to ODE (2.27)
    float delta_tau = to_time - from_time;
    cufftComplex exponent =
        make_cuComplex(psi_real * delta_tau, psi_imag * delta_tau);
    cufftComplex exponential = cuComplexExponential(exponent);

    cufftComplex new_value = cuCmulf(old_value, exponential);

    ft[idx] = new_value;
}

/*

computeGPU()
{

}

*/

vector<float> pricesAtPayoff(Parameters& prms)
{
    vector<float> out(prms.resolution);

    // Tree parameters (see p.53 of notes).
    float u = exp(prms.volatility * sqrt(prms.timeIncrement));
    float d = 1.0 / u;
    float a = exp(prms.riskFreeRate * prms.timeIncrement);
    // float p = (a - d) / (u - d);

    float N = prms.resolution;
    for (int i = 0; i < N; i++) {
        float asset = prms.startPrice * pow(u, i) * pow(d, N - i);
        if (prms.optionType == Call) {
            out[i] = max(asset - prms.strikePrice, 0.0);
        } else {
            out[i] = max(prms.strikePrice - asset, 0.0);
        }
    }

    return out;
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %zu\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %zu\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %zu\n", devProp.totalConstMem);
    printf("Texture alignment:             %zu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

void printAllDevices()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
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

    checkCuda(cudaMalloc( (void**)&ad, csize ));
    checkCuda(cudaMalloc( (void**)&bd, isize ));
    checkCuda(cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ));
    checkCuda(cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ));

    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( 1, 1 );
    hello<<<dimGrid, dimBlock>>>(ad, bd);
    checkCuda(cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ));
    checkCuda(cudaFree( ad ));
    checkCuda(cudaFree( bd ));

    printf("%s\n", a);
}

// Run a couple tests to see that CUDA works properly.
void cudaCheck()
{
    printf("Calling cudaFree(0) no-op...\n");
    cudaFree(0);
    printf("Calling cudaFree(0) succeeded!\n");

    printAllDevices();
    helloWorld();
}

void printPrices(vector<float>& prices) {
    for (int i = 0; i < prices.size(); i++) {
        printf("%f ", prices[i]);
    }
    printf("\n");
}

int main()
{
    assert(sizeof(cufftReal) == sizeof(float));
    assert(sizeof(cufftComplex) == 2 * sizeof(float));

    cudaCheck();

    printf("\nChecks finished. Starting option calculation...\n\n");

    Parameters params;
    vector<float> prices = pricesAtPayoff(params);

    printPrices(prices);

    float N = params.resolution;

    cufftReal* d_prices;
    checkCuda(cudaMalloc((void**)&d_prices, sizeof(cufftReal) * N));
    checkCuda(cudaMemcpy(d_prices, &prices[0], sizeof(cufftReal) * N,
                         cudaMemcpyHostToDevice));

    cufftComplex* d_ft;
    checkCuda(cudaMalloc((void**)&d_ft, sizeof(cufftComplex) * N));

    cufftHandle plan;
    cufftHandle planr;

    // Float to complex interleaved
    checkCufft(cufftPlan1d(&plan, N, CUFFT_R2C, /* deprecated? */ 1));
    checkCufft(cufftPlan1d(&planr, N, CUFFT_C2R, /* deprecated? */ 1));

    // Forward transform
    checkCufft(cufftExecR2C(plan, d_prices, d_ft));

    // Solve ODE
    solveODE<<<dim3(N, 1), dim3(1, 1)>>>(d_ft, 0.0, params.expiryTime,
            params.riskFreeRate,
            params.volatility, params.jumpMean, params.kappa);

    // Reverse transform
    checkCufft(cufftExecC2R(planr, d_ft, d_prices));

    checkCuda(cudaMemcpy(d_prices, &prices[0], sizeof(cufftReal) * N,
                         cudaMemcpyDeviceToHost));
    printPrices(prices);

    // Destroy the cuFFT plan.
    cufftDestroy(plan);
    cudaFree(d_prices);
    cudaFree(d_ft);

    return EXIT_SUCCESS;
}

