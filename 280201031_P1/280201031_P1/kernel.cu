#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void saxpy_kernel(int n, float a, float* x, float* y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    int N;
    float A;

    // Getting input from user for the "N" and "A"
    printf("Please enter the size of the arrays(N): ");
    scanf("%d", &N);
    printf("Plese enter the scalar value(A): ");
    scanf("%f", &A);

    int size = N * sizeof(float);

    // Allocating memory for arrays on CPU
    float* h_x = (float*)malloc(size);
    float* h_y = (float*)malloc(size);

    // Initializing arrays with random values
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)rand() / RAND_MAX;
        h_y[i] = (float)rand() / RAND_MAX;
    }

    // Allocating memory for arrays on GPU
    float* d_x;
    float* d_y;

    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);

    // Copying data from CPU(Host) to GPU(Device)
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // Getting GPU device properties with using cudaGetDeviceProperties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU Device Name: %s\n", prop.name);
    printf("Maximum Threads Per Block: %d\n", prop.maxThreadsPerBlock);

    // Running SAXPY kernel function with different configurations assigned by myself
    int blockDimensions[] = {128, 256, 512};
    int numberofConfigs = sizeof(blockDimensions) / sizeof(int);

    for (int i = 0; i < numberofConfigs; ++i) {
        int blockDimension = blockDimensions[i];
        int threadNumbers = (N + (blockDimension - 1)) / blockDimension;

        saxpy_kernel <<<threadNumbers, blockDimension >>>(N, A, d_x, d_y);

        // Copying the result back from GPU(Device) to CPU(Host)
        cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
        
        printf("(Block Dimension: %d, Thread Numbers per each block: %d) ", blockDimension, threadNumbers);
        printf("Result: ");
        for (int j = 0; j < N; ++j) {
            printf("%.6f | ", h_y[j]);
        }
        printf("\n");
    }

    // Freeing allocated memory
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
