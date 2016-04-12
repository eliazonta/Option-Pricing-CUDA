# About

This program compute the price of European and American options in the GPU, under jump diffusion. It uses the fourier time-stepping method to doing so.

It is mainly an implementation of the theses of Lippa (2013) and Surkov (2009), with the heavy lifting done in CUDA.

# Installation

This program requires CUDA and FFTW (for the CPU version), as well as a NVidia graphics card that supports recent versions of CUDA.

To download CUDA, see NVidia website.

```
sudo apt-get install fftw3 fftw3-dev pkg-config
```

To build the program, simply use `make`. By default, this will compile the program such that all computations are done in double-precision. However, some GPUs may not support double-precision. In which case, `make float` compiles a single-precision version. Note that the single-precision version of the code may not be robust - it is supported only for testing purposes and uses some rather questionable ifdef hacks.

# Usage

The program can be run using `./option` and `./option --cpu` for the CPU version. A number of command-line parameters can be used to precify the parameters of the option (e.g. strike price). For the full list, see the parsing code in `main()` and refer to the test file for examples.

There are two test files: `test_diffusion.sh` and `test_pure.sh`. The former runs the option calculation code for European and American options with or without jump diffusion (Merton or Kou). The latter runs the option calculation for European and American options for pure jump models (Variance Gamma and CGMY).

Note that the pure jump model code currently is not able to produce the correct results, this is a work-in-progress.
