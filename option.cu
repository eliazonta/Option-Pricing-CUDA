#include <assert.h>
#include <chrono>
#include <getopt.h>
#include <math_constants.h>
#include <stdio.h>
#include <vector>

#include <cufft.h>

#include "parameters.h"
#include "fftwproxy.h"
#include "utils.h"

// For quick testing of floats only, otherwise this is obviously a terrible idea.
#ifdef USE_FLOAT

#define double float
#define gendouble2 genfloat2
#define complex cufftComplex

// See definitions in usr/local/cuda/include/cuComplex.h
#define cuCreal cuCrealf
#define cuCimag cuCimagf
#define cuCadd cuCaddf
#define cuCmul cuCmulf
#define cuCdiv cuCdivf
#define cuConj cuConjf
#define cuCabs cuCabsf
#define CUFFT_D2Z CUFFT_R2C
#define CUFFT_Z2D CUFFT_C2R
#define cufftExecD2Z cufftExecR2C
#define cufftExecZ2D cufftExecC2R
#define makeComplex make_cuComplex

// If we're using floats, assume that we're using CUDA Compute Capability < 1.3,
// which means the max block size is 512.
#define MAX_BLOCK_SIZE 512

#else

#define complex cufftDoubleComplex
#define makeComplex make_cuDoubleComplex

// If we're using doubles, assume that we're using CUDA Compute Capability >= 2.x,
// which means the max block size is 1024.
// (We're ignoring Compute Capability 1.3 which supports doubles but not block
// sizes of 1024 since we don't have any devices of that particular generation)
#define MAX_BLOCK_SIZE 1024

#endif

using namespace std;

typedef std::chrono::high_resolution_clock Clock;

// ----------------------------------------------------------------------------
// Complex math helper functions
// ----------------------------------------------------------------------------

__host__ __device__ static __inline__
complex cuComplexExponential(complex x)
{
    double a = cuCreal(x);
    double b = cuCimag(x);
    double ea = exp(a);
    return makeComplex(ea * cos(b), ea * sin(b));
}

__host__ __device__ static __inline__
complex cuComplexLog(complex c)
{
    double x = cuCreal(c);
    double y = cuCimag(c);
    return makeComplex(log(sqrt(x * x + y * y)), atan2(y, x));
}

__host__ __device__ static __inline__
complex cuComplexPower(complex base, complex exponent)
{
    double a = cuCreal(base);
    double b = cuCimag(base);
    double c = cuCreal(exponent);
    double d = cuCimag(exponent);
    double r = cuCabs(base);
    double theta = atan2(b, a);

    double scalar = pow(r, c) * exp(-theta * d);
    double angle = d * log(r) + c * theta;
    return makeComplex(scalar * cos(angle), scalar * sin(angle));
}

__host__ __device__ static __inline__
complex cuComplexScalarMult(double scalar, complex x)
{
    double a = cuCreal(x);
    double b = cuCimag(x);
    return makeComplex(scalar * a, scalar * b);
}

// ----------------------------------------------------------------------------
// Option pricing helper functions and kernels
// ----------------------------------------------------------------------------

__global__
void normalize(double* ft, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    ft[idx] /= length;
}

__host__ __device__ static __inline__
double optionValueAtPayoff(int i, double x_min, double delta_x,
                           double startPrice, double strikePrice,
                           OptionPayoffType type)
{
    double assetPrice = startPrice * exp(x_min + i * delta_x);
    if (type == Call) {
        return max(assetPrice - strikePrice, 0.0);
    } else {
        return max(strikePrice - assetPrice, 0.0);
    }
}

__global__
void optionValuesAtPayoff(double* optionValues,
                          double x_min, double x_max,
                          double startPrice, double strikePrice,
                          OptionPayoffType type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = blockDim.x * gridDim.x;

    double delta_x = (x_max - x_min) / (N - 1);

    optionValues[idx] = optionValueAtPayoff(idx, x_min, delta_x,
                                            startPrice, strikePrice, type);
}

__host__ __device__ static __inline__
double earlyExercise(double optionValue, double startPrice, double strikePrice,
        double x, OptionPayoffType type)
{
    double assetPrice = startPrice * exp(x);
    if (type == Call) {
        return max(optionValue, max(assetPrice - strikePrice, 0.0));
    } else {
        return max(optionValue, max(strikePrice - assetPrice, 0.0));
    }
}

__global__
// TODO: Need better argument names for the last two...
void earlyExercise(double* optionValues, double startPrice, double strikePrice,
                   double x_min, double delta_x,
                   OptionPayoffType type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    optionValues[idx] = earlyExercise(optionValues[idx], startPrice, strikePrice,
            x_min + idx * delta_x, type);
}

// ----------------------------------------------------------------------------
// Calculation of jump diffusion and pure jump characteristic functions
// ----------------------------------------------------------------------------

// Fourier transform of the Merton jump function.
__host__ __device__ static __inline__
complex mertonJumpFT(double k, double mertonNormalStdev, double mertonMean)
{
    // See Lippa (2013) p.13
    double real = M_PI * k * mertonNormalStdev;
    real = -2 * real * real;
    double imag = -2 * M_PI * k * mertonMean;
    complex exponent = makeComplex(real, imag);

    return cuComplexExponential(exponent);
}

// Fourier transform of the Kou jump function (double exponential).
__host__ __device__ static __inline__
complex kouJumpFT(double k, double kouUpJumpProbability,
        double kouUpRate, double kouDownRate)
{
    double p = kouUpJumpProbability;

    // See Lippa (2013) p.54
    complex up = cuCdiv(makeComplex(p, 0),
            makeComplex(1, 2 * M_PI * k / kouUpRate));
    complex down = cuCdiv(makeComplex(1 - p, 0),
            makeComplex(1, -2 * M_PI * k / kouDownRate));

    return cuCadd(up, down);
}

__host__ __device__ static __inline__
complex jumpDiffusionCharacteristic(double k, complex jump_ft,
        double riskFreeRate, double dividend,
        double volatility, double jumpMean,
        double kappa)
{
    // Calculate Ψ (psi) (Lippa (2013) 2.14)
    // The dividend is shown on p.13
    // Equation slightly simplified to save a few operations.
    // TODO: Continuous dividend is too specific, there's more interpretations (see thesis).
    double fst_term = volatility * M_PI * k;
    complex psi = makeComplex(
            (-2.0 * fst_term * fst_term) - (riskFreeRate + jumpMean),
            (riskFreeRate - dividend - jumpMean * kappa - volatility * volatility / 2.0) *
                      (2 * M_PI * k));

    // Jump component.
    psi = cuCadd(psi, cuComplexScalarMult(jumpMean, cuConj(jump_ft)));

    return psi;
}

__host__ __device__ static __inline__
complex varianceGammaCharacteristic(double k,
        double mu,      // jumpMeanInv
        double gamma,   // vg_drift
        double sigma    // volatility
        )
{
    // See Surkov (2009) p.26 or Lippa (2013) p.16
    double w = 2 * M_PI * k;
    complex c = makeComplex(1 + sigma * sigma * mu * w * w / 2, -gamma * mu * w);
    complex vg = cuComplexScalarMult(-1.0 / mu, cuComplexLog(c));
    return vg;
}

__host__ __device__ static __inline__
complex CGMYCharacteristic(double k,
        double C, double G, double M, double Y,
        double gamma /* Γ(-Y), do it on the CPU */)
{
    // Note that the equation in those papers use the symbol ω
    // instead of k for the frequency.
    double w = 2 * M_PI * k;

    // Variance Gamma and CGMY equations see Lippa (2013) p.17
    // and Surkov (2009) p.26
    // Originally from Carr (2002) p.313

    if (Y == 0) {
        // When Y == 0, CGMY == Variance Gamma (Wang, Wan, Forsyth (2007) p. 18)

        // Note that we need to compute the parameters μ (mu), γ (gamma), σ (sigma).
        // We solve for them by using C1 and C2.

        // We obtain C1 and C2 by making the Levy Density of Variance Gamma and CGMY
        // equal when Y == 0.
        double C1 = (G - M) / 2.0;
        double C2 = (G + M) / 2.0;
        double mu = 1.0 / C;
        double temp = C2 / C1;
        double gamma = 2 * C / (C1 * (temp * temp - 1));
        double sigma = sqrt(gamma / C1);

        return varianceGammaCharacteristic(k, mu, gamma, sigma);
    } else {
        complex MwY = cuComplexPower(makeComplex(M, -w), makeComplex(Y, 0));
        complex GwY = cuComplexPower(makeComplex(G, w), makeComplex(Y, 0));
        return cuComplexScalarMult(C * gamma,
                cuCadd(makeComplex(-pow(M, Y) - pow(G, Y), 0), cuCadd(MwY, GwY)));
    }
}

// ----------------------------------------------------------------------------
// Characteristic function kernels
// ----------------------------------------------------------------------------

__global__
void prepareMertonJumpFT(complex* jump_ft, double delta_frequency,
                         int N, double mertonNormalStdev, double mertonMean)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Frequency (see Lippa (2013) p.11 for discretization).
    double m;
    if (idx <= N / 2) {
        m = idx;
    } else {
        m = idx - N;
    }
    double k = delta_frequency * m;

    jump_ft[idx] = mertonJumpFT(k, mertonNormalStdev, mertonMean);
}

__global__
void prepareKouJumpFT(complex* jump_ft, double delta_frequency,
                      int N, double kouUpJumpProbability,
                      double kouUpRate, double kouDownRate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Frequency (see Lippa (2013) p.11 for discretization).
    double m;
    if (idx <= N / 2) {
        m = idx;
    } else {
        m = idx - N;
    }
    double k = delta_frequency * m;

    jump_ft[idx] = kouJumpFT(k, kouUpJumpProbability, kouUpRate, kouDownRate);
}

__global__
void prepareJumpDiffusionCharacteristic(
        complex* characteristic, complex* jump_ft,
        double riskFreeRate, double dividend,
        double volatility, double jumpMean,
        double kappa, double delta_frequency,
        int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Frequency (see Lippa (2013) p.11 for discretization).
    double m;
    if (idx <= N / 2) {
        m = idx;
    } else {
        m = idx - N;
    }
    double k = delta_frequency * m;

    complex jumpTerm;
    if (jump_ft)  {
        jumpTerm = jump_ft[idx];
    } else {
        jumpTerm = makeComplex(0, 0);
    }

    characteristic[idx] = jumpDiffusionCharacteristic(k, jumpTerm,
            riskFreeRate, dividend, volatility, jumpMean, kappa);
}

__global__
void prepareVarianceGammaCharacteristic(
        complex* characteristic,
        int N, double delta_frequency,
        double jumpMeanInv, double vg_drift, double volatility)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Frequency (see Lippa (2013) p.11 for discretization).
    double m;
    if (idx <= N / 2) {
        m = idx;
    } else {
        m = idx - N;
    }
    double k = delta_frequency * m;

    characteristic[idx] = varianceGammaCharacteristic(k, jumpMeanInv, vg_drift, volatility);
}

__global__
void prepareCGMYCharacteristic(
        complex* characteristic,
        int N, double delta_frequency,
        double C, double G, double M, double Y,
        double gamma /* Γ(-Y), do it on the CPU */)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Frequency (see Lippa (2013) p.11 for discretization).
    double m;
    if (idx <= N / 2) {
        m = idx;
    } else {
        m = idx - N;
    }
    double k = delta_frequency * m;

    characteristic[idx] = CGMYCharacteristic(k, C, G, M, Y, gamma);
}

// ----------------------------------------------------------------------------
// Solving the ODE to compute option prices
// ----------------------------------------------------------------------------

__host__ __device__ static __inline__
complex solveODE(complex ft, complex psi, double from_time, double to_time)
{
    // Solution to ODE (Lippa (2013) 2.27)
    double delta_tau = to_time - from_time;
    complex exponent = cuComplexScalarMult(delta_tau, psi);
    complex exponential = cuComplexExponential(exponent);

    return cuCmul(ft, exponential);
}

__global__
void solveODE(complex* ft,
              complex* characteristic,  // psi
              double from_time,         // τ_l (T - t_l)
              double to_time            // τ_u (T - t_u)
             )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    complex old_value = ft[idx];

    complex psi = characteristic[idx];

    ft[idx] = solveODE(old_value, psi, from_time, to_time);
}

// ----------------------------------------------------------------------------
// CPU-side helper functions
// ----------------------------------------------------------------------------

vector<double> optionValuesAtPayoff(Parameters& prms)
{
    vector<double> out(prms.resolution);

    double N = prms.resolution;

    // Discretization parameters (see Lippa (2013) p.11)
    double x_max = prms.x_max();
    double x_min = prms.x_min();
    double delta_x = prms.delta_x();

    for (int i = 0; i < N; i++) {
        out[i] = optionValueAtPayoff(i, x_min, delta_x, prms.startPrice, prms.strikePrice,
                                     prms.optionPayoffType);
    }

    return out;
}

void printComplex(complex x) {
    double a = cuCreal(x);
    double b = cuCimag(x);
    if (b >= 0) {
        printf("%f + %fi", a, b);
    } else {
        printf("%f - %fi", a, -b);
    }
}

void printComplexArray(vector<complex> xs)
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

void printPrices(vector<double>& prices) {
    int first_negative = -1;
    for (int i = 0; i < prices.size(); i++) {
        printf("%f ", prices[i]);
        if (first_negative == -1 && prices[i] < 0) {
            first_negative = i;
        }
    }
    printf("\n");
}


// ----------------------------------------------------------------------------
// CPU-side option price calculation
// ----------------------------------------------------------------------------

void computeCPU_characteristic(Parameters& params, vector<complex>& characteristic)
{
    int N = params.resolution;
    double delta_frequency = params.delta_frequency();

    // Note: technically it would be fine to compute just N/2 + 1 of these,
    // but compute them all for safety..
    for (int i = 0; i < N; i++) {
        // Frequency (see Lippa (2013) p.11 for discretization).
        double m;
        if (i <= N / 2) {
            m = i;
        } else {
            m = i - N;
        }
        double k = delta_frequency * m;

        if (params.jumpType == VarianceGamma) {
            characteristic[i] = varianceGammaCharacteristic(k,
                    params.jumpMeanInverse(), params.VG_driftRate, params.volatility);
        } else if (params.jumpType == CGMY) {
            characteristic[i] = CGMYCharacteristic(k,
                    params.CGMY_C, params.CGMY_G, params.CGMY_M, params.CGMY_Y,
                    tgamma(-params.CGMY_Y));
        } else {
            complex jumpFT = makeComplex(0, 0);
            if (params.jumpType != None) {
                if (params.jumpType == Merton) {
                    jumpFT = mertonJumpFT(k, params.mertonNormalStdev, params.mertonMean);
                } else if (params.jumpType == Kou) {
                    jumpFT = kouJumpFT(k, params.kouUpJumpProbability,
                            params.kouUpRate, params.kouDownRate);
                }
            }

            characteristic[i] = jumpDiffusionCharacteristic(k, jumpFT,
                    params.riskFreeRate, params.dividendRate,
                    params.volatility, params.jumpMean, params.kappa());
        }
    }
}

void computeCPU_timesteps(Parameters& params,
                          vector<double>& optionValues,
                          vector<complex>& characteristic)
{
    int N = params.resolution;
    double x_min = params.x_min();
    double delta_x = params.delta_x();

    // Forward transform of option price
    vector<complex> ft(N);

    // FFTW execution
    FFTWProxy proxy(N, &optionValues[0], (gendouble2 *)(&ft[0]));

    for (int i = 0; i < params.timesteps; i++) {
        double from_time = (double)i / params.timesteps * params.expiryTime;
        double to_time = (double)(i + 1) / params.timesteps * params.expiryTime;

        // Forward transform
        proxy.execForward();

        // Solve ODE
        int fourier_size = N / 2 + 1;
        for (int j = 0; j < fourier_size; j++) {
            ft[j] = solveODE(ft[j], characteristic[j], from_time, to_time);
        }

        // Reverse transform
        proxy.execInverse();

        for (int idx = 0; idx < N; idx++) {
            // Scale, fftw doesn't do it for you.
            optionValues[idx] /= N;
        }

        // Early exercise
        if (params.optionExerciseType == American) {
            for (int j = 0; j < N; j++) {
                optionValues[j] = earlyExercise(optionValues[j],
                        params.startPrice, params.strikePrice,
                        x_min + j * delta_x, params.optionPayoffType);
            }
        }
    }
}

void computeCPU(Parameters& params)
{
    int N = params.resolution;

    // Option values at time t = 0
    vector<double> optionValues = optionValuesAtPayoff(params);

    // Characteristic Ψ (psi)
    vector<complex> characteristic(N);
    computeCPU_characteristic(params, characteristic);

    computeCPU_timesteps(params, optionValues, characteristic);

    int answer_index = params.answer_index();

    if (params.verbose) {
        printf("Price at index %i: %f\n", answer_index, optionValues[answer_index]);
    } else {
        printf("%f\n", optionValues[answer_index]);
    }
}

// ----------------------------------------------------------------------------
// GPU-side option price calculation
// ----------------------------------------------------------------------------

void computeGPU_initialize(Parameters& params,
                           double*& d_prices, complex*& d_ft, complex*& d_characteristic,
                           cufftHandle& plan, cufftHandle& planr)
{
    int N = params.resolution;
    double x_min = params.x_min();
    double x_max = params.x_max();

    checkCuda(cudaMalloc((void**)&d_prices, sizeof(double) * N));

    // Option values at time t = 0
    optionValuesAtPayoff<<<max(N / MAX_BLOCK_SIZE, 1), min(N, MAX_BLOCK_SIZE)>>>(
            d_prices, x_min, x_max, params.startPrice, params.strikePrice, params.optionPayoffType);
    checkCuda(cudaPeekAtLastError());

    checkCuda(cudaMalloc((void**)&d_ft, sizeof(complex) * N));

    // Float to complex interleaved transform preparation.
    // NOTE: This is the most expensive part of this function if the number of timesteps
    //       is small, because cufftPlan calculates O(N) transcendental functions. On a GTX 980,
    //       it takes on the order of 500 ms. If this code is eventually extended to calculate
    //       multiple different options of the same size, this (and other initialization code)
    //       should be called only once.
    checkCufft(cufftPlan1d(&plan, N, CUFFT_D2Z, /* deprecated? */ 1));
    checkCufft(cufftPlan1d(&planr, N, CUFFT_Z2D, /* deprecated? */ 1));

    checkCuda(cudaMalloc((void**)&d_characteristic, sizeof(complex) * N));
}

void computeGPU_characteristic(Parameters& params,
                               complex* d_characteristic, complex*& d_jump_ft)
{
    int N = params.resolution;

    double delta_frequency = params.delta_frequency();

    // Note: technically it would be fine to compute just N/2 + 1 of these,
    // but compute them all for safety..
    if (params.jumpType == VarianceGamma) {
        prepareVarianceGammaCharacteristic<<<max(N / MAX_BLOCK_SIZE, 1), min(N, MAX_BLOCK_SIZE)>>>(
                d_characteristic,
                N, delta_frequency,
                params.jumpMeanInverse(), params.VG_driftRate, params.volatility);
    } else if (params.jumpType == CGMY) {
        prepareCGMYCharacteristic<<<max(N / MAX_BLOCK_SIZE, 1), min(N, MAX_BLOCK_SIZE)>>>(
                d_characteristic,
                N, delta_frequency,
                params.CGMY_C, params.CGMY_G, params.CGMY_M, params.CGMY_Y,
                tgamma(-params.CGMY_Y));
    } else {
        if (params.jumpType != None) {
            checkCuda(cudaMalloc((void**)&d_jump_ft, sizeof(complex) * N));

            if (params.jumpType == Merton) {
                prepareMertonJumpFT<<<max(N / MAX_BLOCK_SIZE, 1), min(N, MAX_BLOCK_SIZE)>>>(
                        d_jump_ft, delta_frequency, N,
                        params.mertonNormalStdev, params.mertonMean);
                checkCuda(cudaPeekAtLastError());
            } else if (params.jumpType == Kou) {
                prepareKouJumpFT<<<max(N / MAX_BLOCK_SIZE, 1), min(N, MAX_BLOCK_SIZE)>>>(
                        d_jump_ft, delta_frequency, N,
                        params.kouUpJumpProbability, params.kouUpRate, params.kouDownRate);
                checkCuda(cudaPeekAtLastError());
            }
        }

        prepareJumpDiffusionCharacteristic<<<max(N / MAX_BLOCK_SIZE, 1), min(N, MAX_BLOCK_SIZE)>>>(
                d_characteristic, d_jump_ft,
                params.riskFreeRate, params.dividendRate,
                params.volatility, params.jumpMean, params.kappa(),
                delta_frequency, N);
    }
    checkCuda(cudaPeekAtLastError());
}

void computeGPU_timesteps(Parameters& params, double* d_prices, complex* d_ft,
                          complex* d_characteristic,
                          cufftHandle& plan, cufftHandle& planr)
{
    int N = params.resolution;
    double x_min = params.x_min();
    double delta_x = params.delta_x();

    for (int i = 0; i < params.timesteps; i++) {
        double from_time = (double)i / params.timesteps * params.expiryTime;
        double to_time = (double)(i + 1) / params.timesteps * params.expiryTime;

        // Forward transform
        checkCufft(cufftExecD2Z(plan, d_prices, d_ft));

        // Solve ODE
        // Note that we solve the ODE only on the first half of the frequency
        // data. Why? A fourier transform on real (non-complex) data will give
        // hermetian symmetry, where the second half of the array is just the
        // complex conjugate of the first half. So cufft & fftw doesn't store
        // any values in the second half at all! They don't use the second half
        // of the array either to compute the inverse fourier transform.
        // See http://www.fftw.org/doc/The-1d-Real_002ddata-DFT.html
        int fourier_size = N / 2 + 1;
        int fourier_block_count = (int)ceil((double)fourier_size / MAX_BLOCK_SIZE);
        int fourier_block_size = min(fourier_size, MAX_BLOCK_SIZE);
        solveODE<<<fourier_block_count, fourier_block_size>>>(
                d_ft, d_characteristic, from_time, to_time);
        checkCuda(cudaPeekAtLastError());

        // Reverse transform
        checkCufft(cufftExecZ2D(planr, d_ft, d_prices));
        normalize<<<max(N / MAX_BLOCK_SIZE, 1), min(N, MAX_BLOCK_SIZE)>>>(d_prices, N);
        checkCuda(cudaPeekAtLastError());

        // Consider early exercise for American options. This is the same technique
        // as option pricing using dynamic programming: at each timestep, set the
        // option value to the payoff if is higher than the current option value.
        if (params.optionExerciseType == American) {
            earlyExercise<<<max(N / MAX_BLOCK_SIZE, 1), min(N, MAX_BLOCK_SIZE)>>>(
                    d_prices, params.startPrice, params.strikePrice,
                    x_min, delta_x, params.optionPayoffType);
            checkCuda(cudaPeekAtLastError());
        }
    }
}

void computeGPU(Parameters& params)
{
    int N = params.resolution;

    // Convention: pointers to GPU memory (to "device" memory) should be prefixed with d_
    // This is common in CUDA code to avoid dereferencing pointers in the wrong place, and
    // also one of the few proper ways to use Hungarian notation.
    double* d_prices;                   // Option values
    complex* d_ft;                      // Option values in fourier space
    complex* d_characteristic = NULL;   // Characteristic Ψ (psi)
    cufftHandle plan;
    cufftHandle planr;

    computeGPU_initialize(params, d_prices, d_ft, d_characteristic, plan, planr);

    // For jump diffusion, store the fourier transform of the jump function.
    complex *d_jump_ft = NULL;
    computeGPU_characteristic(params, d_characteristic, d_jump_ft);

    computeGPU_timesteps(params, d_prices, d_ft, d_characteristic, plan, planr);

    vector<double> optionValues(N);
    checkCuda(cudaMemcpy(&optionValues[0], d_prices, sizeof(double) * N,
                         cudaMemcpyDeviceToHost));

    // Cleanup.
    cufftDestroy(plan);
    cufftDestroy(planr);
    cudaFree(d_prices);
    cudaFree(d_ft);
    cudaFree(d_jump_ft);

    int answer_index = params.answer_index();

    if (params.verbose) {
        printf("Price at index %i: %f\n", answer_index, optionValues[answer_index]);
    } else {
        printf("%f\n", optionValues[answer_index]);
    }
}

int main(int argc, char** argv)
{
    assert(sizeof(complex) == 2 * sizeof(double));

    Parameters params;
    bool useCPU = false;

    // Parse arguments
    while (true) {
        static struct option long_options[] = {
            {"payoff",  required_argument, 0, 'p'},
            {"exercise",  required_argument, 0, 'e'},
            {"dividend",  required_argument, 0, 'q'},
            {"debug",  no_argument, 0, 'd'},
            {"verbose",  no_argument, 0, 'v'},
            {"cpu",  no_argument, 0, 'c'},
            // General parameters
            {"S",  required_argument, 0, 'S'},
            {"K",  required_argument, 0, 'K'},
            {"r",  required_argument, 0, 'r'},
            {"T",  required_argument, 0, 'T'},
            {"sigma",  required_argument, 0, 'o'},
            {"resolution",  required_argument, 0, 'n'},
            {"timesteps",  required_argument, 0, 't'},
            // Jump args
            {"lambda",  required_argument, 0, 'l'},
            // Merton Jump args
            {"mertonjumps",  no_argument, 0, 'm'},
            {"mertonmu",  required_argument, 0, 'u'},
            {"mertongamma",  required_argument, 0, 'y'},
            // Kou Jump args
            {"koujumps",  no_argument, 0, 'k'},
            {"p",  required_argument, 0, '0'},
            {"etaUp",  required_argument, 0, '1'},
            {"etaDown",  required_argument, 0, '2'},
            // Variance Gamma model
            {"vg",  no_argument, 0, '5'},
            {"vgdrift",  required_argument, 0, '6'},
            {"vgmu",  required_argument, 0, '7'},
            // CGMY model
            {"CGMY",  no_argument, 0, '4'},
            {"C",  required_argument, 0, 'C'},
            {"G",  required_argument, 0, 'G'},
            {"M",  required_argument, 0, 'M'},
            {"Y",  required_argument, 0, 'Y'},
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
                break;
            case 'p':
                if (!strcmp(optarg, "put")) {
                    params.optionPayoffType = Put;
                } else if (!strcmp(optarg, "call")) {
                    params.optionPayoffType = Call;
                } else {
                    fprintf(stderr, "Option payoff type %s invalid.\n", optarg);
                    abort();
                }
                break;
            case 'q':
                params.dividendRate = atof(optarg);
                break;
            case 'l':
                params.jumpMean = atof(optarg);
                break;
            case 'u':
                params.mertonMean = atof(optarg);
                break;
            case '0':
                params.kouUpJumpProbability = atof(optarg);
                break;
            case '1':
                params.kouUpRate = atof(optarg);
                break;
            case '2':
                params.kouDownRate = atof(optarg);
                break;
            case '5':
                params.jumpType = VarianceGamma;
                break;
            case '6':
                params.VG_driftRate = atof(optarg);
                break;
            case '7':
                params.jumpMean = 1.0 / atof(optarg);
                break;
            case '4':
                params.jumpType = CGMY;
                break;
            case 'C':
                params.CGMY_C = atof(optarg);
                break;
            case 'G':
                params.CGMY_G = atof(optarg);
                break;
            case 'M':
                params.CGMY_M = atof(optarg);
                break;
            case 'Y':
                params.CGMY_Y = atof(optarg);
                break;
            case 'y':
                params.mertonNormalStdev = atof(optarg);
                break;
            case 'S':
                params.startPrice = atof(optarg);
                break;
            case 'K':
                params.strikePrice = atof(optarg);
                break;
            case 'r':
                params.riskFreeRate = atof(optarg);
                break;
            case 'T':
                params.expiryTime = atof(optarg);
                break;
            case 'o':
                params.volatility = atof(optarg);
                break;
            case 'd':
                params.debug = true;
                break;
            case 'm':
                params.jumpType = Merton;
                break;
            case 'k':
                params.jumpType = Kou;
                break;
            case 'n':
                params.resolution = atoi(optarg);
                assert(isPowerOfTwo(params.resolution));
                break;
            case 't':
                params.timesteps = atoi(optarg);
                break;
            case 'v':
                params.verbose = true;
                break;
            case 'c':
                useCPU = true;
                break;
            case '?':
                break;
            default:
                abort();
        }
    }

    cudaCheck(params.debug);

    if (params.verbose) {
        printf("\nChecks finished. Starting option calculation...\n\n");
    }

    // http://stackoverflow.com/questions/5521146/what-is-the-best-most-accurate-timer-in-c
    auto start = Clock::now();
    if (useCPU) {
        if (params.verbose) {
            printf("\nComputing CPU results...\n");
        }
        computeCPU(params);
    } else {
        if (params.verbose) {
            printf("\nComputing GPU results...\n");
        }
        computeGPU(params);
    }
    auto end = Clock::now();

    chrono::duration<double, milli> fp_ms = end - start;
    printf("Computed with %s in %.2f ms (timesteps %d, resolution %d).\n",
            useCPU ? "CPU" : "GPU",
            fp_ms.count(), params.timesteps, params.resolution);

    return EXIT_SUCCESS;
}

