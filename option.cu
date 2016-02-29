#include <assert.h>
#include <getopt.h>
#include <math_constants.h>
#include <stdio.h>
#include <vector>

#include <cufft.h>

#include "parameters.h"
#include "utils.h"

using namespace std;

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
void normalize(cufftReal* ft, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    ft[idx] /= length;
}

__global__
// TODO: Need better argument names for the last two...
void earlyExercise(cufftReal* ft, float startPrice, float strikePrice,
                   float x_min, float delta_x,
                   OptionPayoffType type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float assetPrice = startPrice * exp(x_min + idx * delta_x);
    if (type == Call) {
        ft[idx] = max(ft[idx], max(assetPrice - strikePrice, 0.0));
    } else {
        ft[idx] = max(ft[idx], max(strikePrice - assetPrice, 0.0));
    }
}

__global__
void solveODE(cufftComplex* ft,
              cufftComplex* jump_ft,   // Fourier transform of the jump function
              float from_time,         // τ_l (T - t_l)
              float to_time,           // τ_u (T - t_u)
              float riskFreeRate,
              float dividend,
              float volatility,
              float jumpMean,
              float kappa,
              float delta_frequency,
              int N)
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
    // The dividend is shown on p.13
    // Equation slightly simplified to save a few operations.
    // TODO: Continuous dividends?
    float fst_term = volatility * M_PI * k;
    cufftComplex psi = make_cuComplex(
            (-2.0 * fst_term * fst_term) - (riskFreeRate + jumpMean),
            (riskFreeRate - dividend - jumpMean * kappa - volatility * volatility / 2.0) *
                      (2 * M_PI * k));

    // Jump component.
    if (jump_ft) {
        psi = cuCaddf(psi, cuComplexScalarMult(jumpMean, cuConjf(jump_ft[idx])));
    }

    // Solution to ODE (2.27)
    float delta_tau = to_time - from_time;
    cufftComplex exponent = cuComplexScalarMult(delta_tau, psi);
    cufftComplex exponential = cuComplexExponential(exponent);

    cufftComplex new_value = cuCmulf(old_value, exponential);

    ft[idx] = new_value;
}

vector<float> assetPricesAtPayoff(Parameters& prms)
{
    float N = prms.resolution;
    vector<float> out(N);

    // Discretization parameters (see p.11)
    // TODO: Factor out into params?
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

// Fourier transform of the Merton jump function.
vector<cufftComplex> mertonJumpFT(Parameters& prms, float delta_frequency)
{
    int N = prms.resolution;

    // See Lippa (2013) p.13
    vector<cufftComplex> ft(N);
    for (int i = 0; i < N; i++) {
        // Frequency (see p.11 for discretization).
        float m;
        if (i <= N / 2) {
            m = i;
        } else {
            m = i - N;
        }
        float k = delta_frequency * m;

        float real = M_PI * k * prms.mertonNormalStdev;
        real = -2 * real * real;
        float imag = -2 * M_PI * k * prms.driftRate;
        cufftComplex exponent = make_cuComplex(real, imag);
        ft[i] = cuComplexExponential(exponent);
    }

    return ft;
}

// Fourier transform of the Kou jump function
vector<cufftComplex> kouJumpFT(Parameters& prms, float delta_frequency)
{
    int N = prms.resolution;
    float p = prms.kouUpJumpProbability;

    // See Lippa (2013) p.54
    vector<cufftComplex> ft(N);
    for (int i = 0; i < N; i++) {
        // Frequency (see p.11 for discretization).
        float m;
        if (i <= N / 2) {
            m = i;
        } else {
            m = i - N;
        }
        float k = delta_frequency * m;

        cufftComplex up = cuCdivf(make_cuComplex(p, 0),
                make_cuComplex(1, 2 * M_PI * k / prms.kouUpRate));
        cufftComplex down = cuCdivf(make_cuComplex(1 - p, 0),
                make_cuComplex(1, -2 * M_PI * k / prms.kouDownRate));

        ft[i] = cuCaddf(up, down);
    }

    return ft;
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

void printPrices(vector<float>& prices) {
    int first_negative = -1;
    for (int i = 0; i < prices.size(); i++) {
        printf("%f ", prices[i]);
        if (first_negative == -1 && prices[i] < 0) {
            first_negative = i;
        }
    }
    printf("\n");
    printf("First negative number at %d.\n", first_negative);
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
    float kappa = params.kappa();

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

    if (params.verbose) {
        printf("Price is at index %f. Price at %d: %f. Price at %d: %f.\n",
                answer_index, (int)floor(answer_index), price_lower,
                (int)ceil(answer_index), price_upper);
        printf("Interpolated price: %f\n", interpolated);
    } else {
        printf("%f\n", interpolated);
    }
}

void computeGPU(Parameters& params, vector<float>& assetPrices, vector<float>& optionValues)
{
    // Option values at time t = 0
    vector<float> initialValues(optionValues.size());

    int N = params.resolution;

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
    float delta_x = (x_max - x_min) / (N - 1);
    float delta_frequency = (float)(N - 1) / (x_max - x_min) / N;

    // Jump function
    vector<cufftComplex> jump_ft;
    cufftComplex *d_jump_ft = NULL;

    if (params.jumpType == Merton) {
        jump_ft = mertonJumpFT(params, delta_frequency);
    } else if (params.jumpType == Kou) {
        jump_ft = kouJumpFT(params, delta_frequency);
    }

    if (params.jumpType != None) {
        checkCuda(cudaMalloc((void**)&d_jump_ft, sizeof(cufftComplex) * N));
        checkCuda(cudaMemcpy(d_jump_ft, &jump_ft[0], sizeof(cufftComplex) * N,
                             cudaMemcpyHostToDevice));
    }

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
        int ode_size = N / 2 + 1;
        solveODE<<<dim3(max(ode_size / 512, 1), 1), dim3(min(ode_size, 512), 1)>>>(
                d_ft, d_jump_ft, from_time, to_time,
                params.riskFreeRate, params.dividendRate,
                params.volatility, params.jumpMean, params.kappa(),
                delta_frequency, N);

        // Reverse transform
        checkCufft(cufftExecC2R(planr, d_ft, d_prices));
        normalize<<<dim3(max(N / 512, 1), 1), dim3(min((int)N, 512), 1)>>>(d_prices, N);

        // Consider early exercise for American options. This is the same technique
        // as option pricing using dynamic programming: at each timestep, set the
        // option value to the payoff if is higher than the current option value.
        if (params.optionExerciseType == American) {
            earlyExercise<<<dim3(max(N / 512, 1), 1), dim3(min((int)N, 512), 1)>>>(
                    d_prices, params.startPrice, params.strikePrice,
                    x_min, delta_x, params.optionPayoffType);
        }
    }

    checkCuda(cudaMemcpy(&initialValues[0], d_prices, sizeof(cufftReal) * N,
                         cudaMemcpyDeviceToHost));

    // Destroy the cuFFT plan.
    cufftDestroy(plan);
    cufftDestroy(planr);
    cudaFree(d_prices);
    cudaFree(d_ft);
    cudaFree(d_jump_ft);

    float answer_index = -x_min * (N - 1) / (x_max - x_min);
    float price_lower = initialValues[(int)floor(answer_index)];
    float price_upper = initialValues[(int)ceil(answer_index)];
    float interpolated = price_lower * (ceil(answer_index) - answer_index) +
                         price_upper * (answer_index - floor(answer_index));

    if (params.verbose) {
        printf("Price is at index %f. Price at %d: %f. Price at %d: %f.\n",
                answer_index, (int)floor(answer_index), price_lower,
                (int)ceil(answer_index), price_upper);
        printf("Interpolated price: %f\n", interpolated);
    } else {
        printf("%f\n", interpolated);
    }
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
            {"dividend",  required_argument, 0, 'q'},
            {"debug",  no_argument, 0, 'd'},
            {"mertonjumps",  no_argument, 0, 'm'},
            {"koujumps",  no_argument, 0, 'k'},
            {"lambda",  required_argument, 0, 'l'},
            {"p",  required_argument, 0, '0'},
            {"eta1",  required_argument, 0, '1'},
            {"eta2",  required_argument, 0, '2'},
            {"gamma",  required_argument, 0, 'y'},
            {"S",  required_argument, 0, 'S'},
            {"K",  required_argument, 0, 'K'},
            {"r",  required_argument, 0, 'r'},
            {"T",  required_argument, 0, 'T'},
            {"sigma",  required_argument, 0, 'o'},
            {"resolution",  required_argument, 0, 'n'},
            {"timesteps",  required_argument, 0, 't'},
            {"verbose",  no_argument, 0, 'v'},
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
            case '0':
                params.kouUpJumpProbability = atof(optarg);
                break;
            case '1':
                params.kouUpRate = atof(optarg);
                break;
            case '2':
                params.kouDownRate = atof(optarg);
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

    vector<float> assetPrices = assetPricesAtPayoff(params);
    vector<float> optionValues = optionValuesAtPayoff(params, assetPrices);

    if (params.verbose) {
        printf("\nComputing GPU results...\n");
    }
    computeGPU(params, assetPrices, optionValues);

    return EXIT_SUCCESS;
}

