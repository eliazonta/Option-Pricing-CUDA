#include "fftwproxy.h"

#include <fftw3.h>

struct FFTWProxyImpl
{
    virtual ~FFTWProxyImpl() {};
    virtual void executeForward() = 0;
    virtual void executeInverse() = 0;
};

struct FFTWProxyImplDouble : public FFTWProxyImpl
{
    FFTWProxyImplDouble(int n, double* timespace, gendouble2* freqspace)
    {
        forwardPlan = fftw_plan_dft_r2c_1d(
                n, timespace, (fftw_complex*)freqspace, FFTW_ESTIMATE);
        inversePlan = fftw_plan_dft_c2r_1d(
                n, (fftw_complex*)freqspace, timespace, FFTW_ESTIMATE);
    }

    virtual ~FFTWProxyImplDouble()
    {
        fftw_destroy_plan(forwardPlan);
        fftw_destroy_plan(inversePlan);
    }

    void executeForward()
    {
        fftw_execute(forwardPlan);
    }

    void executeInverse()
    {
        fftw_execute(inversePlan);
    }

    fftw_plan forwardPlan;
    fftw_plan inversePlan;
};

struct FFTWProxyImplFloat : public FFTWProxyImpl
{
    FFTWProxyImplFloat(int n, float* timespace, genfloat2* freqspace)
    {
        forwardPlan = fftwf_plan_dft_r2c_1d(
                n, timespace, (fftwf_complex*)freqspace, FFTW_ESTIMATE);
        inversePlan = fftwf_plan_dft_c2r_1d(
                n, (fftwf_complex*)freqspace, timespace, FFTW_ESTIMATE);
    }

    virtual ~FFTWProxyImplFloat()
    {
        fftwf_destroy_plan(forwardPlan);
        fftwf_destroy_plan(inversePlan);
    }

    void executeForward()
    {
        fftwf_execute(forwardPlan);
    }

    void executeInverse()
    {
        fftwf_execute(inversePlan);
    }

    fftwf_plan forwardPlan;
    fftwf_plan inversePlan;
};

FFTWProxy::FFTWProxy(int n, float* timespace, genfloat2* freqspace)
{
    size = n;
    impl = new FFTWProxyImplFloat(n, timespace, freqspace);
}


FFTWProxy::FFTWProxy(int n, double* timespace, gendouble2* freqspace)
{
    size = n;
    impl = new FFTWProxyImplDouble(n, timespace, freqspace);
}

FFTWProxy::~FFTWProxy()
{
    delete impl;
}

void FFTWProxy::execForward()
{
    impl->executeForward();
}

void FFTWProxy::execInverse()
{
    impl->executeInverse();
}
