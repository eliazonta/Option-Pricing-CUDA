#include <assert.h>
#include <getopt.h>
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
    float b = cuCimagf(x);
    float ea = exp(a);
    return make_cuComplex(ea * cos(b), ea * sin(b));
}

__host__ __device__ static __inline__
cufftComplex cuComplexScalarMult(float scalar, cufftComplex x)
{
    float a = cuCrealf(x);
    float b = cuCimagf(x);
    return make_cuComplex(scalar * a, scalar * b);
}

__global__
void hello(char *a, int *b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] += b[idx];
}

__global__
void normalize(cufftReal* ft, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    ft[idx] /= length;
}

__global__
void solveODE(cufftComplex* ft,
              float from_time,         // τ_l (T - t_l)
              float to_time,           // τ_u (T - t_u)
              float riskFreeRate, float volatility,
              float jumpMean, float kappa,
              int N, float delta_frequency)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    cufftComplex old_value = ft[idx];

    // Frequency (see p.11 for discretization).
    float m;
    if (idx <= N / 2) {
        m = idx;
    } else {
        m = idx - N;
    }
    float k = delta_frequency * m;

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

vector<float> assetPricesAtPayoff(Parameters& prms)
{
    float N = prms.resolution;
    vector<float> out(N);

    float x_max = prms.logBoundary;
    float x_min = -prms.logBoundary;
    float delta_x = (x_max - x_min) / (N - 1);

    /*
    // Tree parameters (see p.53 of notes).
    float u = exp(prms.volatility * sqrt(prms.timeIncrement));
    float d = 1.0 / u;
    float a = exp(prms.riskFreeRate * prms.timeIncrement);
    // float p = (a - d) / (u - d);

    for (int i = 0; i < N; i++) {
        out[i] = prms.startPrice * pow(u, i) * pow(d, N - i);
    }
    */

    for (int i = 0; i < N; i++) {
        out[i] = prms.startPrice * exp(x_min + i * delta_x);
    }

    return out;
}

vector<float> optionValuesAtPayoff(Parameters& prms, vector<float>& assetPrices)
{
    vector<float> out(prms.resolution);

    float N = prms.resolution;
    for (int i = 0; i < N; i++) {
        if (prms.optionPayoffType == Call) {
            out[i] = max(assetPrices[i] - prms.strikePrice, 0.0);
        } else {
            out[i] = max(prms.strikePrice - assetPrices[i], 0.0);
        }
    }

    return out;
}

void printComplex(cufftComplex x) {
    float a = cuCrealf(x);
    float b = cuCimagf(x);
    printf("%f + %fi", a, b);
}

void printComplexArray(vector<cufftComplex> xs)
{
    for (int i = 0; i < xs.size(); i++) {
        printComplex(xs[i]);
        if (i < xs.size() - 1)
            printf(", ");
        if (i % 5 == 0 && i > 0)
            printf("\n");
    }
    printf("\n");
}

vector<cufftComplex> dft(vector<float>& in)
{
    vector<cufftComplex> out(in.size());

    for (int k = 0; k < out.size(); k++) {
        out[k] = make_cuComplex(0, 0);

        for (int n = 0; n < in.size(); n++) {
            cufftComplex exponent = make_cuComplex(0, -2.0f * M_PI * k * n / in.size());
            out[k] = cuCaddf(out[k], cuComplexScalarMult(in[n], cuComplexExponential(exponent)));
        }
    }

    return out;
}

vector<cufftComplex> idft_complex(vector<cufftComplex>& in)
{
    vector<cufftComplex> out(in.size());

    for (int k = 0; k < out.size(); k++) {
        out[k] = make_cuComplex(0, 0);

        for (int n = 0; n < in.size(); n++) {
            cufftComplex exponent = make_cuComplex(0, 2.0f * M_PI * k * n / in.size());
            out[k] = cuCaddf(out[k], cuCmulf(in[n], cuComplexExponential(exponent)));
        }

        out[k] = cuComplexScalarMult(1.0 / out.size(), out[k]);
    }

    /*
    printComplexArray(out);
    printf("\n");
    */

    return out;
}

vector<float> idft(vector<cufftComplex>& in)
{
    vector<cufftComplex> ift = idft_complex(in);
    vector<float> out(ift.size());
    for (int i = 0; i < ift.size(); i++) {
        out[i] = cuCrealf(ift[i]);
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

    // For some reason, systems that don't have CUDA devices might
    // print infinitely many of them. If we run the program accidently,
    // the program might hang while printing. We don't want that.
    if (devCount > 5)
        printf("Printing first 5 devices.\n");

    // Iterate through devices
    for (int i = 0; i < min(5, devCount); ++i)
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

void computeCPU(Parameters& params, vector<float>& assetPrices, vector<float>& optionValues)
{
    int N = params.resolution;

    // Discretization parameters (see p.11)
    float x_max = params.logBoundary;
    float x_min = -params.logBoundary;
    float delta_frequency = (float)(N - 1) / (x_max - x_min) / N;

    float from_time = 0.0f;
    float to_time = params.expiryTime;
    float riskFreeRate = params.riskFreeRate;
    float volatility = params.volatility;
    float jumpMean = params.jumpMean;
    float kappa = params.kappa;

    // Forward transform
    vector<cufftComplex> ft = dft(optionValues);
    vector<cufftComplex> ft2(N);

    for (int idx = 0; idx < ft.size(); idx++) {
        cufftComplex old_value = ft[idx];

        // Frequency (see p.11 for discretization).
        float m;
        if (idx <= N / 2) {
            m = idx;
        } else {
            m = idx - N;
        }
        float k = delta_frequency * m;

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

        ft2[idx] = new_value;
    }

    // Inverse transform
    vector<float> ift = idft(ft2);

    // printPrices(ift);

    float answer_index = -x_min * (N - 1) / (x_max - x_min);
    float price_lower = ift[(int)floor(answer_index)];
    float price_upper = ift[(int)ceil(answer_index)];
    float interpolated = price_lower * (ceil(answer_index) - answer_index) +
                         price_upper * (answer_index - floor(answer_index));
    printf("Price is at index %f. Price at %d: %f. Price at %d: %f.\n",
            answer_index, (int)floor(answer_index), price_lower,
            (int)ceil(answer_index), price_upper);
    printf("Interpolated price: %f\n", interpolated);
}

void computeGPU(Parameters& params, vector<float>& assetPrices, vector<float>& optionValues)
{
    // Option values at time t = 0
    vector<float> initialValues(optionValues.size());

    float N = params.resolution;

    cufftReal* d_prices;
    checkCuda(cudaMalloc((void**)&d_prices, sizeof(cufftReal) * N));
    checkCuda(cudaMemcpy(d_prices, &optionValues[0], sizeof(cufftReal) * N,
                         cudaMemcpyHostToDevice));

    cufftComplex* d_ft;
    checkCuda(cudaMalloc((void**)&d_ft, sizeof(cufftComplex) * N));

    cufftHandle plan;
    cufftHandle planr;

    // Float to complex interleaved
    checkCufft(cufftPlan1d(&plan, N, CUFFT_R2C, /* deprecated? */ 1));
    checkCufft(cufftPlan1d(&planr, N, CUFFT_C2R, /* deprecated? */ 1));

    // Discretization parameters (see p.11)
    float x_max = params.logBoundary;
    float x_min = -params.logBoundary;
    float delta_frequency = (float)(N - 1) / (x_max - x_min) / N;

    for (int i = 0; i < params.timesteps; i++) {
        float from_time = (float)i / params.timesteps * params.expiryTime;
        float to_time = (float)(i + 1) / params.timesteps * params.expiryTime;

        // Forward transform
        checkCufft(cufftExecR2C(plan, d_prices, d_ft));

        // Solve ODE
        // Note that we solve the ODE only on the first half of the frequency
        // data. Why? A fourier transform on real (non-complex) data will give
        // hermetian symmetry, where the second half of the array is just the
        // complex conjugate of the first half. So cufft & fftw doesn't store
        // any values in the second half at all! They don't use the second half
        // of the array either to compute the inverse fourier transform.
        // See http://www.fftw.org/doc/The-1d-Real_002ddata-DFT.html
        int ode_size = N / 2;
        solveODE<<<dim3(ode_size / 512, 1), dim3(min(ode_size, 512), 1)>>>(
                d_ft, from_time, to_time, params.riskFreeRate,
                params.volatility, params.jumpMean, params.kappa,
                N, delta_frequency);

        // Reverse transform
        checkCufft(cufftExecC2R(planr, d_ft, d_prices));
        normalize<<<dim3(N / 512, 1), dim3(min((int)N, 512), 1)>>>(d_prices, N);
    }

    checkCuda(cudaMemcpy(&initialValues[0], d_prices, sizeof(cufftReal) * N,
                         cudaMemcpyDeviceToHost));

    // Destroy the cuFFT plan.
    cufftDestroy(plan);
    cufftDestroy(planr);
    cudaFree(d_prices);
    cudaFree(d_ft);

    float answer_index = -x_min * (N - 1) / (x_max - x_min);
    float price_lower = initialValues[(int)floor(answer_index)];
    float price_upper = initialValues[(int)ceil(answer_index)];
    float interpolated = price_lower * (ceil(answer_index) - answer_index) +
                         price_upper * (answer_index - floor(answer_index));
    printf("Price is at index %f. Price at %d: %f. Price at %d: %f.\n",
            answer_index, (int)floor(answer_index), price_lower,
            (int)ceil(answer_index), price_upper);
    printf("Interpolated price: %f\n", interpolated);
}

int main(int argc, char** argv)
{
    assert(sizeof(cufftReal) == sizeof(float));
    assert(sizeof(cufftComplex) == 2 * sizeof(float));

    Parameters params;

    // Parse arguments
    while (true) {
        static struct option long_options[] = {
            {"payoff",  required_argument, 0, 'p'},
            {"exercise",  required_argument, 0, 'e'},
            {"timesteps",  required_argument, 0, 't'},
            {0, 0, 0, 0}
        };

        int option_index = 0;
        char c = getopt_long(argc, argv, "abc:d:f:", long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch (c) {
            case 'e':
                if (!strcmp(optarg, "european")) {
                    params.optionExerciseType = European;
                } else if (!strcmp(optarg, "american")) {
                    params.optionExerciseType = American;
                } else {
                    fprintf(stderr, "Option exercise type %s invalid.\n", optarg);
                    abort();
                }
            case 'p':
                if (!strcmp(optarg, "put")) {
                    params.optionPayoffType = Put;
                } else if (!strcmp(optarg, "call")) {
                    params.optionPayoffType = Call;
                } else {
                    fprintf(stderr, "Option payoff type %s invalid.\n", optarg);
                    abort();
                }
            case 't':
                params.timesteps = atoi(optarg);
            case '?':
                break;
            default:
                abort();
        }
    }

    cudaCheck();

    printf("\nChecks finished. Starting option calculation...\n\n");

    vector<float> assetPrices = assetPricesAtPayoff(params);
    vector<float> optionValues = optionValuesAtPayoff(params, assetPrices);

    printPrices(optionValues);

    printf("\nComputing CPU results...\n");
    //computeCPU(params, assetPrices, optionValues);
    printf("\nComputing GPU results...\n");
    computeGPU(params, assetPrices, optionValues);

    /*
    float X[] = {1, 0, 1, 0, 1, 0, 1, 0};
    vector<float> a(X, X + 8);
    vector<cufftComplex> x = dft(a);
    printComplexArray(x);
    vector<cufftComplex> y = idft(x);
    printComplexArray(y);
    */

    return EXIT_SUCCESS;
}

