
#include <cassert>
#include <ctime>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

typedef __nv_bfloat16 bf16;
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Error checking helper
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      printf("CUBLAS Error at %s:%d: %d\n", __FILE__, __LINE__, status);       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#include "examples/matmul/group_gemm.cuh" // Our new implementation

// Random number generator
std::default_random_engine generator(42);
std::normal_distribution<float> distribution(0.0f, 1.0f);

cublasHandle_t cublas_handle;

// Helper function to initialize matrix with random values
void initialize_matrix(std::vector<bf16> &matrix, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    matrix[i] = __float2bfloat16(distribution(generator));
  }
}

// Helper function to verify results
bool verify_results(const std::vector<bf16> &ref, const std::vector<bf16> &test,
                    size_t size, float tolerance = 0.1f) {
  bool passed = true;
  float max_diff = 0.0f;
  int max_diff_idx = 0;

  for (size_t i = 0; i < size; ++i) {
    float ref_val = __bfloat162float(ref[i]);
    float test_val = __bfloat162float(test[i]);
    float diff = std::abs(ref_val - test_val);

    if (diff > max_diff) {
      max_diff = diff;
      max_diff_idx = i;
    }

    if (diff > tolerance) {
      passed = false;
      printf("Mismatch at index %zu: ref = %f, test = %f (diff = %f)\n", i,
             ref_val, test_val, diff);
      break;
    }
  }

  printf("Max difference: %f at index %d\n", max_diff, max_diff_idx);
  return passed;
}

// cuBLAS reference implementation
void cublas_gemm(int M, int N, int K, const bf16 *A, const bf16 *B, bf16 *C) {
  float alpha = 1.0f;
  float beta = 0.0f;

  CHECK_CUBLAS(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                            &alpha, B, CUDA_R_16BF, N, A, CUDA_R_16BF, K, &beta,
                            C, CUDA_R_16BF, N, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT));
}

struct TestCase {
  int M, N, K;
  std::vector<bf16> A, B, C, C_ref;
  bf16 *d_A, *d_B, *d_C, *d_C_ref;

  TestCase(int m, int n, int k) : M(m), N(n), K(k) {
    // Allocate host memory
    size_t a_size = M * K;
    size_t b_size = K * N;
    size_t c_size = M * N;

    A.resize(a_size);
    B.resize(b_size);
    C.resize(c_size);
    C_ref.resize(c_size);

    // Initialize matrices
    initialize_matrix(A, a_size);
    initialize_matrix(B, b_size);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A, a_size * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_B, b_size * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_C, c_size * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_C_ref, c_size * sizeof(bf16)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, A.data(), a_size * sizeof(bf16),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), b_size * sizeof(bf16),
                          cudaMemcpyHostToDevice));
  }

  ~TestCase() {
    if (d_A)
      cudaFree(d_A);
    if (d_B)
      cudaFree(d_B);
    if (d_C)
      cudaFree(d_C);
    if (d_C_ref)
      cudaFree(d_C_ref);
  }
};

void run_benchmark(const std::vector<std::tuple<int, int, int>> &sizes) {
  std::vector<TestCase> test_cases;
  for (const auto &[m, n, k] : sizes) {
    test_cases.emplace_back(m, n, k);
  }

  // Prepare GroupGEMM parameters
  std::vector<groupgemm::GemmParams> batch_params;
  for (const auto &test : test_cases) {
    batch_params.push_back(
        {test.M, test.N, test.K, test.d_A, test.d_B, test.d_C, 1.0f, 0.0f});
  }

  // Create and initialize GroupGEMM
  groupgemm::GroupGemm<> group_gemm;
  group_gemm.initializeBatch(batch_params);

  // Timing events
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Warmup phase
  printf("\nWarmup phase...\n");
  for (int i = 0; i < 3; ++i) {
    group_gemm.launch();
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  // Verification phase
  printf("\nVerification phase...\n");
  for (size_t i = 0; i < test_cases.size(); ++i) {
    auto &test = test_cases[i];

    // Run cuBLAS reference
    cublas_gemm(test.M, test.N, test.K, test.d_A, test.d_B, test.d_C_ref);

    // Run our implementation
    group_gemm.launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy results back and verify
    CHECK_CUDA(cudaMemcpy(test.C.data(), test.d_C,
                          test.M * test.N * sizeof(bf16),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(test.C_ref.data(), test.d_C_ref,
                          test.M * test.N * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    printf("Verifying GEMM %zu (M=%d, N=%d, K=%d): ", i, test.M, test.N,
           test.K);
    if (!verify_results(test.C_ref, test.C, test.M * test.N)) {
      printf("FAILED!\n");
      return;
    }
    printf("PASSED\n");
  }

  // Performance benchmark
  printf("\nPerformance benchmark...\n");
  const int NUM_ITERS = 100;
  float elapsed_ms;

  // Benchmark cuBLAS
  CHECK_CUDA(cudaEventRecord(start));
  for (int iter = 0; iter < NUM_ITERS; ++iter) {
    for (const auto &test : test_cases) {
      cublas_gemm(test.M, test.N, test.K, test.d_A, test.d_B, test.d_C_ref);
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

  double cublas_ms = elapsed_ms / NUM_ITERS;
  double total_flops = 0;
  for (const auto &test : test_cases) {
    total_flops += 2.0 * test.M * test.N * test.K;
  }
  double cublas_tflops = (total_flops * 1e-12) / (cublas_ms * 1e-3);

  printf("cuBLAS Batch Time: %.3f ms, Performance: %.2f TFLOPS\n", cublas_ms,
         cublas_tflops);

  // Benchmark GroupGEMM
  CHECK_CUDA(cudaEventRecord(start));
  for (int iter = 0; iter < NUM_ITERS; ++iter) {
    group_gemm.launch();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

  double group_ms = elapsed_ms / NUM_ITERS;
  double group_tflops = (total_flops * 1e-12) / (group_ms * 1e-3);
  double speedup = cublas_ms / group_ms;

  printf("GroupGEMM Batch Time: %.3f ms, Performance: %.2f TFLOPS (%.2fx vs "
         "cuBLAS)\n",
         group_ms, group_tflops, speedup);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
  // Initialize cuBLAS
  CHECK_CUBLAS(cublasCreate(&cublas_handle));

  // Test configurations
  std::vector<std::tuple<int, int, int>> test_configs = {
      {4096, 4096, 4096}, // Square
      {3072, 4096, 2048}, // Rectangular
      {2048, 2048, 2048}, // Smaller square
      {4096, 2048, 3072}  // Another rectangular
  };

  // Run benchmarks with different batch sizes
  std::vector<int> batch_sizes = {1, 2, 4};
  for (int batch_size : batch_sizes) {
    printf("\n=== Testing with batch size %d ===\n", batch_size);
    std::vector<std::tuple<int, int, int>> batch_config;
    for (int i = 0; i < batch_size && i < test_configs.size(); ++i) {
      batch_config.push_back(test_configs[i]);
    }
    run_benchmark(batch_config);
  }

  // Cleanup
  CHECK_CUBLAS(cublasDestroy(cublas_handle));
  return 0;
}
