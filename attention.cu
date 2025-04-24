#include <iostream>
#include <cuda_runtime.h>

#define CACHE_SIZE 1024 // Define the cache size

// Structure for Key-Value pairs
struct KVPair {
    int key;
    float value;
};

// CUDA Kernel for Attention Mechanism
__global__ void attentionKernel(const float *query, const float *key, const float *value, 
                                 float *output, int N, KVPair *cache) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Compute attention scores
        float score = 0.0f;
        for (int j = 0; j < N; ++j) {
            score += query[idx] * key[j]; // Simplified dot product
        }

        // Check if the result is cached
        for (int j = 0; j < CACHE_SIZE; j++) {
            if (cache[j].key == idx) {
                output[idx] = cache[j].value; // Use cached value
                return;
            }
        }

        // If not cached, compute the output using the value matrix
        output[idx] = score; // For simplicity, using score directly as output

        // Store result in cache
        cache[idx % CACHE_SIZE] = {idx, output[idx]};
    }
}

// Function to launch the attention kernel
extern "C" void launchAttention(int N) {
    float *query, *key, *value, *output;
    size_t size = N * sizeof(float);
    
    query = (float*)malloc(size);
    key = (float*)malloc(size);
    value = (float*)malloc(size);
    output = (float*)malloc(size);

    // Initialize query, key, and value
    for (int i = 0; i < N; ++i) {
        query[i] = 1.0f; // Fill query with 1s
        key[i] = 2.0f;   // Fill key with 2s
        value[i] = 3.0f; // Fill value with 3s
    }

    // Allocate device memory for query, key, value, output, and cache
    float *d_query, *d_key, *d_value, *d_output;
    KVPair *d_cache;
    cudaMalloc(&d_query, size);
    cudaMalloc(&d_key, size);
    cudaMalloc(&d_value, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_cache, CACHE_SIZE * sizeof(KVPair));

    // Copy data to device
    cudaMemcpy(d_query, query, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, size, cudaMemcpyHostToDevice);
    cudaMemset(d_cache, -1, CACHE_SIZE * sizeof(KVPair)); // Initialize cache with invalid keys

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the attention kernel
    attentionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_query, d_key, d_value, d_output, N, d_cache);

    // Copy result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the result for the first few elements
    for (int i = 0; i < 10; ++i) {
        std::cout << "Output[" << i << "] = " << output[i] << std::endl;
    }

    // Clean up
    cudaFree(d_query);
    cudaFree(d_key);	
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_cache);
    free(query);
    free(key);
    free(value);
    free(output);
}

int main() {
    int N = 1024; // Example size
    launchAttention(N); // Launch the attention mechanism
    return 0; // Indicate successful execution
}

