#include "parameters.h"

#include <cmath>

Parameters::Parameters()
: startPrice(100.0)
, strikePrice(100.0)
, riskFreeRate(0.05)
, dividendRate(0.02)
, expiryTime(10.0)
, volatility(0.15)
, jumpMean(0.1)
, driftRate(-1.08)
, gamma(0.4)
, resolution(100)
, optionType(Put)
{
    timeIncrement = expiryTime / resolution;

    // Calculation of kappa, see p.13 of paper
    kappa = exp(driftRate + gamma * gamma / 2.0) - 1.0;
}
