enum OptionType
{
    Call,
    Put,
};

struct Parameters
{
    // Symbol: S
    double startPrice;

    // Symbol: K
    double strikePrice;

    // Symbol: r
    double riskFreeRate;

    // Symbol: q
    double dividendRate;

    // Symbol: T
    double expiryTime;

    // Symbol: σ (sigma)
    double volatility;

    // Symbol: λ (lambda)
    // Mean of the poisson distribution for jumps.
    double jumpMean;

    // Symbol: μ (mu)
    double driftRate;

    // Symbol: γ (gamma)
    // Normally, this is σ (standard deviation) in the normal,
    // but here we already use σ for the volatility so we need
    // another variable name.
    double normalStdev;

    // Symbol: x_min and x_max
    double logBoundary;

    // Symbol: N
    int resolution;

    // Symbol: delta t
    double timeIncrement;

    // Symbol: κ (kappa)
    double kappa;

    OptionType optionType;

    Parameters();
};
