#include "parameters.h"

#include <assert.h>
#include <cmath>

Parameters::Parameters()
: startPrice(100.0)
, strikePrice(100.0)
, riskFreeRate(0.05)
, dividendRate(0.02)
, expiryTime(10.0)
, volatility(0.15)
//, jumpMean(0.1)
, jumpMean(0.0)
, driftRate(-1.08)
, normalStdev(0.4)
, logBoundary(7.5)
, resolution(2048)
, optionPayoffType(Put)
, optionExerciseType(European)
{
    assert(resolution % 512 == 0);

    timeIncrement = expiryTime / resolution;

    // Calculation of kappa, see p.13 of paper
    kappa = exp(driftRate + normalStdev * normalStdev / 2.0) - 1.0;

    kappa = 0.0;
}
