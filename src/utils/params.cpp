#include "parameters.h"

#include <assert.h>
#include <cmath>

#include "utils.h"

Parameters::Parameters()
: startPrice(100.0)
, strikePrice(100.0)
, riskFreeRate(0.05)
, dividendRate(0.0)
, expiryTime(10.0)
, volatility(0.15)
, logBoundary(7.5)
, resolution(2048)
, timesteps(1)
, optionPayoffType(Put)
, optionExerciseType(European)
, debug(false)
, verbose(false)
{
    assert(isPowerOfTwo(resolution));

    // No jumps.
    jumpType = None;
    jumpMean = 0.0;
}

double Parameters::timeIncrement()
{
    return expiryTime / resolution;
}

double Parameters::x_min()
{
    return -logBoundary;
}

// Adjust x_max so that the option price is located
// at index N/2 (an integer), so that we don't need to interpolate.
double Parameters::x_max()
{
    return -x_min() * 2 * (resolution - 1) / resolution + x_min();
}

double Parameters::delta_x()
{
    return (x_max() - x_min()) / (resolution - 1);
}

double Parameters::delta_frequency()
{
    return (double)(resolution - 1) / (x_max() - x_min()) / resolution;
}

int Parameters::answer_index()
{
    double answer_index = -x_min() * (resolution - 1) / (x_max() - x_min());
    assert(answer_index == (int)answer_index);
    return (int)answer_index;
}

double Parameters::kappa()
{
    if (jumpType == Merton) {
        // Lippa (2013) p.13
        return exp(mertonMean + mertonNormalStdev * mertonNormalStdev / 2.0) - 1.0;
    } else if (jumpType == Kou) {
        // Lippa (2013) p.54
        return kouUpJumpProbability * kouUpRate / (kouUpRate - 1)
            + (1 - kouUpJumpProbability) * kouDownRate / (kouDownRate + 1) - 1.0;
    } else {
        return 0.0;
    }
}
