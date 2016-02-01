#include "parameters.h"

Parameters::Parameters()
: startPrice(100)
, strikePrice(100)
, riskFreeRate(0.05)
, dividendRate(0.02)
, expiryTime(10)
, volatility(0.15)
, jumpMean(0.1)
, driftRate(-1.08)
, gamma(0.4)
{
}
