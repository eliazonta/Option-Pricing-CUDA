// Copied from docs
// CUFFT_SUCCESS = 0, // The cuFFT operation was successful
// CUFFT_INVALID_PLAN = 1, // cuFFT was passed an invalid plan handle
// CUFFT_ALLOC_FAILED = 2, // cuFFT failed to allocate GPU or CPU memory
// CUFFT_INVALID_TYPE = 3, // No longer used
// CUFFT_INVALID_VALUE = 4, // User specified an invalid pointer or parameter
// CUFFT_INTERNAL_ERROR = 5, // Driver or internal cuFFT library error
// CUFFT_EXEC_FAILED = 6, // Failed to execute an FFT on the GPU
// CUFFT_SETUP_FAILED = 7, // The cuFFT library failed to initialize
// CUFFT_INVALID_SIZE = 8, // User specified an invalid transform size
// CUFFT_UNALIGNED_DATA = 9, // No longer used
// CUFFT_INCOMPLETE_PARAMETER_LIST = 10, // Missing parameters in call
// CUFFT_INVALID_DEVICE = 11, // Execution of a plan was on different GPU than plan creation
// CUFFT_PARSE_ERROR = 12, // Internal plan database error
// CUFFT_NO_WORKSPACE = 13 // No workspace has been provided prior to plan execution
#define checkCufft(result) do {           \
    if (result != CUFFT_SUCCESS) {                      \
        fprintf(stderr, "CUFFT at %d error %d: %s\n", __LINE__, result, cudaGetErrorString(cudaGetLastError()));   \
        exit(-1);                                       \
    }                                                   \
} while(0)

#define checkCuda(result) do {            \
    if (result != cudaSuccess) {                        \
        fprintf(stderr, "CUDA at %d error %d: %s\n", __LINE__, result, cudaGetErrorString(cudaGetLastError()));   \
        exit(-1);                                       \
    }                                                   \
} while(0)

int isPowerOfTwo (unsigned int x);
void cudaCheck(bool debug);

