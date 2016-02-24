#include "utils.h"

#include <stdio.h>

// Complement & Compare technique, see
// http://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
int isPowerOfTwo (unsigned int x)
{
    return ((x != 0) && ((x & (~x + 1)) == x));
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %zu\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %zu\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %zu\n", devProp.totalConstMem);
    printf("Texture alignment:             %zu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

void printAllDevices(bool debug)
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);

    if (devCount < 1) {
        fprintf(stderr, "Error: no CUDA devices?\n");
    }

    if (debug) {
        printf("CUDA Device Query...\n");
        printf("There are %d CUDA devices.\n", devCount);
    }

    // For some reason, systems that don't have CUDA devices might
    // print infinitely many of them. If we run the program accidently,
    // the program might hang while printing. We don't want that.
    if (devCount > 5)
        printf("Printing first 5 devices.\n");

    // Iterate through devices
    for (int i = 0; i < min(5, devCount); ++i)
    {
        // Get device properties
        if (debug)
            printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);

        if (debug)
            printDevProp(devProp);
    }
}

__global__
void hello(char *a, int *b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] += b[idx];
}

// Prints Hello, World if the GPU code is working right.
void helloWorld()
{
    const int N = 16;
    const int blocksize = 16;

    char a[N] = "Hello \0\0\0\0\0\0";
    int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    char *ad;
    int *bd;
    const int csize = N*sizeof(char);
    const int isize = N*sizeof(int);

    checkCuda(cudaMalloc( (void**)&ad, csize ));
    checkCuda(cudaMalloc( (void**)&bd, isize ));
    checkCuda(cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ));
    checkCuda(cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ));

    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( 1, 1 );
    hello<<<dimGrid, dimBlock>>>(ad, bd);
    checkCuda(cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ));
    checkCuda(cudaFree( ad ));
    checkCuda(cudaFree( bd ));

    if (strcmp(a, "World!")) {
        fprintf(stderr, "Error: Expected \"World!\", got \"%s\"\n", a);
        exit(-1);
    }
}

// Run a couple tests to see that CUDA works properly.
void cudaCheck(bool debug)
{
    if (debug) {
        printf("Calling cudaFree(0) no-op...\n");
    }
    checkCuda(cudaFree(0));
    if (debug) {
        printf("Calling cudaFree(0) succeeded!\n");
    }

    printAllDevices(debug);

    helloWorld();
}
