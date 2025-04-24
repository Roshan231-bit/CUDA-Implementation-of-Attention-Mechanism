#include <iostream>
#include <cuda_runtime.h>

#define CACHE_SIZE 9 // 3x3 matrix will have 9 possible cached entries

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
extern "C" void launchAttention(const float* query, const float* key, const float* value) {
    int N = 9; // 3x3 matrix (flattened to 9 elements)
    size_t size = N * sizeof(float);
    
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

    int threadsPerBlock = 3; // Each thread computes one output for 3x3 matrix
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the attention kernel
    attentionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_query, d_key, d_value, d_output, N, d_cache);

    // Copy result back to host
    float output[N];
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Output:\n";
    for (int i = 0; i < 9; ++i) {
        std::cout << "Output[" << i << "] = " << output[i] << std::endl;
    }

    // Clean up
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_cache);
}

int main() {
    // Define a 3x3 matrix for query, key, and value
    float query[9] = {1.0f, 1.0f, 1.0f,
                      2.0f, 2.0f, 2.0f,
                      3.0f, 3.0f, 3.0f}; // Example query matrix

    float key[9] = {1.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 1.0f}; // Example key matrix

    float value[9] = {0.0f, 1.0f, 2.0f,
                      3.0f, 4.0f, 5.0f,
                      6.0f, 7.0f, 8.0f}; // Example value matrix

    int iterations = 10; // Define the number of iterations

    for (int i = 0; i < iterations; ++i) {
        std::cout << "Iteration " << i + 1 << ":\n";
        launchAttention(query, key, value); // Launch the attention mechanism
    }

    return 0; // Indicate successful execution
}

