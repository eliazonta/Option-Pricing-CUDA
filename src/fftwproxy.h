#pragma once

#include "utils/utils.h"

// Generic way of declaring a structure of two doubles.
typedef float genfloat2[2];

// Generic way of declaring a structure of two doubles.
typedef double gendouble2[2];

// An instance of this class just forwards calls to FFTW.
//
// Why do we do this in such a contrived way, instead of just
// calling the appropriate FFTW directly?
//
// Because right now, NVCC can't compile a file that includes FFTW
// because of some issues of __float128 definition.
//
// We also can't put all the CPU code in a separate file because
// we would still like to share some code between the CPU and GPU
// implementations. And we can't just factor out that shared code
// in a separate file because we'd really like them to be inlined.
struct FFTWProxyImpl;
class FFTWProxy
{
public:
    FFTWProxy(int n, float* timespace, genfloat2* freqspace);
    FFTWProxy(int n, double* timespace, gendouble2* freqspace);
    ~FFTWProxy();

    void execForward();
    void execInverse();

private:
    int size;

    // And we need to have an impl that isn't part of the header
    // because we need to store the FFTW plan, and we can't put
    // the type of that plan in the header (even if it's private)
    // since we'd need to include FFTW in this header.
    FFTWProxyImpl* impl;
};
