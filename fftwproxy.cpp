#include "fftwproxy.h"

#include <fftw3.h>

struct FFTWProxyImpl
{
    FFTWProxyImpl(int n, double* timespace, genfloat2* freqspace)
    {
        forwardPlan = fftw_plan_dft_r2c_1d(
                n, timespace, (fftw_complex*)freqspace, FFTW_ESTIMATE);
        inversePlan = fftw_plan_dft_c2r_1d(
                n, (fftw_complex*)freqspace, timespace, FFTW_ESTIMATE);
    }

    ~FFTWProxyImpl()
    {
        fftw_destroy_plan(forwardPlan);
        fftw_destroy_plan(inversePlan);
    }

    fftw_plan forwardPlan;
    fftw_plan inversePlan;
};

FFTWProxy::FFTWProxy(int n, double* timespace, genfloat2* freqspace)
{
    size = n;
    impl = new FFTWProxyImpl(n, timespace, freqspace);
}

FFTWProxy::~FFTWProxy()
{
    delete impl;
}

void FFTWProxy::execForward()
{
    fftw_execute(impl->forwardPlan);
}

void FFTWProxy::execInverse()
{
    fftw_execute(impl->inversePlan);
}
