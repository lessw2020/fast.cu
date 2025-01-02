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

/////////

#include "examples/matmul/group_gemm.cuh"
#include "examples/matmul/wgmax.cuh"

// Error checking macros
#define cudaCheck(err)                                                         \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      printf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,         \
             cudaGetErrorString(err_));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define cublasCheck(err)                                                       \
  do {                                                                         \
    cublasStatus_t err_ = (err);                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                       \
      printf("cuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// TMA alignment requirements
constexpr int M_ALIGN = 128;
constexpr int N_ALIGN = 256;
constexpr int K_ALIGN = 64;

// Random number generator setup
std::default_random_engine generator(42);
std::normal_distribution<float> distribution(0.0f, 1.0f);

// cuBLAS handle declaration
cublasHandle_t cublas_handle;

class MatrixTest {
public:
  int M, N, K;
  int M_padded, N_padded, K_padded;
  std::vector<bf16> A, B, C, C_ref;
  bf16 *d_A, *d_B, *d_C, *d_C_ref;

  MatrixTest(int m, int n, int k) : M(m), N(n), K(k) {
    // Calculate padded dimensions for TMA alignment
    M_padded = ((M + M_ALIGN - 1) / M_ALIGN) * M_ALIGN;
    N_padded = ((N + N_ALIGN - 1) / N_ALIGN) * N_ALIGN;
    K_padded = ((K + K_ALIGN - 1) / K_ALIGN) * K_ALIGN;

    printf("Matrix %dx%dx%d (Padded: %dx%dx%d)\n", M, N, K, M_padded, N_padded,
           K_padded);

    // Allocate and initialize host memory
    size_t a_size = M_padded * K_padded;
    size_t b_size = K_padded * N_padded;
    size_t c_size = M_padded * N_padded;

    A.resize(a_size, __float2bfloat16(0.0f));
    B.resize(b_size, __float2bfloat16(0.0f));
    C.resize(c_size);
    C_ref.resize(c_size);

    // Initialize matrices with random values
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < K; ++j) {
        A[i * K_padded + j] = __float2bfloat16(distribution(generator));
      }
    }

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < N; ++j) {
        B[i * N_padded + j] = __float2bfloat16(distribution(generator));
      }
    }

    // Allocate device memory
    cudaCheck(cudaMalloc(&d_A, a_size * sizeof(bf16)));
    cudaCheck(cudaMalloc(&d_B, b_size * sizeof(bf16)));
    cudaCheck(cudaMalloc(&d_C, c_size * sizeof(bf16)));
    cudaCheck(cudaMalloc(&d_C_ref, c_size * sizeof(bf16)));

    // Copy data to device
    cudaCheck(cudaMemcpy(d_A, A.data(), a_size * sizeof(bf16),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, B.data(), b_size * sizeof(bf16),
                         cudaMemcpyHostToDevice));

    // Zero initialize output matrices
    cudaCheck(cudaMemset(d_C, 0, c_size * sizeof(bf16)));
    cudaCheck(cudaMemset(d_C_ref, 0, c_size * sizeof(bf16)));
  }

  ~MatrixTest() {
    if (d_A)
      cudaFree(d_A);
    if (d_B)
      cudaFree(d_B);
    if (d_C)
      cudaFree(d_C);
    if (d_C_ref)
      cudaFree(d_C_ref);
  }

  void runCuBLAS() {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasCheck(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, d_B, CUDA_R_16BF, N_padded, d_A,
                             CUDA_R_16BF, K_padded, &beta, d_C_ref, CUDA_R_16BF,
                             N_padded, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  }

  bool verify(float tolerance = 0.1f) {
    // Copy results back
    cudaCheck(cudaMemcpy(C.data(), d_C, M_padded * N_padded * sizeof(bf16),
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(C_ref.data(), d_C_ref,
                         M_padded * N_padded * sizeof(bf16),
                         cudaMemcpyDeviceToHost));

    bool passed = true;
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    int num_errors = 0;
    const int max_errors_to_print = 10;

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        int idx = i * N_padded + j;
        float ref_val = __bfloat162float(C_ref[idx]);
        float test_val = __bfloat162float(C[idx]);
        float diff = std::abs(ref_val - test_val);

        if (diff > max_diff) {
          max_diff = diff;
          max_diff_idx = idx;
        }

        if (diff > tolerance) {
          if (num_errors < max_errors_to_print) {
            printf("Mismatch at [%d,%d]: ref=%f test=%f (diff=%f)\n", i, j,
                   ref_val, test_val, diff);
          }
          passed = false;
          num_errors++;
        }
      }
    }

    printf("Max difference: %f at index %d\n", max_diff, max_diff_idx);
    if (!passed) {
      printf("Total errors: %d\n", num_errors);
    }
    return passed;
  }
};

void run_benchmark(const std::vector<std::tuple<int, int, int>> &test_sizes,
                   int batch_size) {
  printf("\n=== Testing batch size %d ===\n", batch_size);

  // Create test cases
  std::vector<MatrixTest> tests;
  std::vector<int> Ms, Ns, Ks;
  std::vector<bf16 *> As, Bs, Cs;

  for (int i = 0; i < batch_size; ++i) {
    auto [m, n, k] = test_sizes[i % test_sizes.size()];
    tests.emplace_back(m, n, k);

    auto &test = tests.back();
    Ms.push_back(test.M_padded);
    Ns.push_back(test.N_padded);
    Ks.push_back(test.K_padded);
    As.push_back(test.d_A);
    Bs.push_back(test.d_B);
    Cs.push_back(test.d_C);
  }

  // Initialize group GEMM
  groupgemm::GroupGemm<> group_gemm;
  group_gemm.initializeBatch(Ms, Ns, Ks, As, Bs, Cs);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));

  // Warmup phase
  printf("\nWarmup phase...\n");
  for (int i = 0; i < 3; ++i) {
    group_gemm.launch();
    cudaCheck(cudaDeviceSynchronize());
  }

  // Verification phase
  printf("\nVerification phase...\n");
  for (size_t i = 0; i < tests.size(); ++i) {
    auto &test = tests[i];

    // Run reference implementation
    test.runCuBLAS();

    // Run our implementation
    group_gemm.launch();
    cudaCheck(cudaDeviceSynchronize());

    printf("Verifying GEMM %zu (%dx%dx%d): ", i, test.M, test.N, test.K);
    if (!test.verify()) {
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
  cudaCheck(cudaEventRecord(start));
  for (int iter = 0; iter < NUM_ITERS; ++iter) {
    for (auto &test : tests) {
      test.runCuBLAS();
    }
  }
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));

  double cublas_ms = elapsed_ms / NUM_ITERS;
  double total_flops = 0;
  for (const auto &test : tests) {
    total_flops += 2.0 * test.M * test.N * test.K;
  }
  double cublas_tflops = (total_flops * 1e-12) / (cublas_ms * 1e-3);

  printf("cuBLAS Batch Time: %.3f ms, Performance: %.2f TFLOPS\n", cublas_ms,
         cublas_tflops);

  // Benchmark GroupGEMM
  cudaCheck(cudaEventRecord(start));
  for (int iter = 0; iter < NUM_ITERS; ++iter) {
    group_gemm.launch();
  }
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));

  double group_ms = elapsed_ms / NUM_ITERS;
  double group_tflops = (total_flops * 1e-12) / (group_ms * 1e-3);
  double speedup = cublas_ms / group_ms;

  printf(
      "GroupGEMM Time: %.3f ms, Performance: %.2f TFLOPS (%.2fx vs cuBLAS)\n",
      group_ms, group_tflops, speedup);

  cudaCheck(cudaEventDestroy(start));
  cudaCheck(cudaEventDestroy(stop));
}

int main() {
  // Initialize cuBLAS
  cublasCheck(cublasCreate(&cublas_handle));

  // Test configurations
  std::vector<std::tuple<int, int, int>> test_configs = {
      //{1024, 1024, 1024}, // Base case
      {2048, 2048, 2048}, // Larger size
      {3072, 2048, 1024}, // Rectangular
      {4096, 4096, 4096}  // Very large
  };

  // Test different batch sizes
  std::vector<int> batch_sizes = {1, 2, 4, 8};
  for (int batch_size : batch_sizes) {
    run_benchmark(test_configs, batch_size);
  }

  // Cleanup
  cublasCheck(cublasDestroy(cublas_handle));
  return 0;
}
