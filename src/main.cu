#include <assert.h>
#include <chrono>
#include <getopt.h>
#include <math_constants.h>
#include <stdio.h>
#include <vector>

#include <cufft.h>

#include "utils/params.h"
#include "fftwproxy.h"
#include "utils/utils.h"
#include "option.cuh"

using namespace std;

typedef std::chrono::high_resolution_clock Clock;

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

    if (params.verbose) 
    {
        printf("\nChecks finished. Starting option calculation...\n\n");
    }

    // http://stackoverflow.com/questions/5521146/what-is-the-best-most-accurate-timer-in-c
    auto start = Clock::now();
    if (useCPU) {
        if (params.verbose) 
        {
            printf("\nComputing CPU results...\n");
        }
        computeCPU(params);
    } else 
    {
        if (params.verbose) 
        {
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
