enum OptionPayoffType
{
    Call,
    Put,
};

enum OptionExerciseType
{
    European,
    American,
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

    // Number of timesteps to perform (number of times to go to fourier
    // space and back). For European options, this does not need to be
    // greater than one.
    int timesteps;

    // Symbol: delta t
    double timeIncrement;

    // Symbol: κ (kappa)
    double kappa;

    // Put or Call
    OptionPayoffType optionPayoffType;

    // European or American
    OptionExerciseType optionExerciseType;

    Parameters();
};
