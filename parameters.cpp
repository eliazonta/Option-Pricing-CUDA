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
, driftRate(-1.08)
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

double Parameters::kappa()
{
    if (jumpType == Merton) {
        // Lippa (2013) p.13
        return exp(driftRate + mertonNormalStdev * mertonNormalStdev / 2.0) - 1.0;
    } else if (jumpType == Kou) {
        // Lippa (2013) p.54
        return kouUpJumpProbability * kouUpRate / (kouUpRate - 1)
            + (1 - kouUpJumpProbability) * kouDownRate / (kouDownRate + 1) - 1.0;
    } else {
        return 0.0;
    }
}
