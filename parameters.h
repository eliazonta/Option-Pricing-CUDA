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
    Parameters();

    void enableJumps();

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

    // Symbol: q
    // Continuous dividend rate.
    double dividend;

    // Symbol: x_min and x_max
    double logBoundary;

    // Symbol: N
    unsigned int resolution;

    // Number of timesteps to perform (number of times to go to fourier
    // space and back). For European options, this does not need to be
    // greater than one.
    unsigned int timesteps;

    // Symbol: delta t
    double timeIncrement;

    // Symbol: κ (kappa)
    double kappa;

    // Simulate jumps with jump diffusion
    bool useJumps;

    // Print more debug info.
    bool debug;
    bool verbose;

    // Put or Call
    OptionPayoffType optionPayoffType;

    // European or American
    OptionExerciseType optionExerciseType;
};

class JumpProbabilityDensity
{
public:
    virtual double evaluate(double y) = 0;
};

// This is a 'double' exponential in the sense that it combines two
// exponential distributions (one for up (positive) jumps, and one
// for down (negative) jumps) using an indicator variable.
//
// The probability density function is:
// f(y) = p_up * λ1 * exp(-λ1 * y) * [indicator y>=0] +
//        (1 - p_up) * λ2 * exp(λ2 * y) * [indicator y < 0]
//
// Where p_up is the probability of an up jump, λ1, λ2 are the rate parameters
// of the exponential distrubtion (= 1 / mean).
class DoubleExponential : public JumpProbabilityDensity
{
public:
    DoubleExponential();

    double evaluate(double y);

    double rateUp;
    double rateDown;
    double probabilityUpJump;
};
