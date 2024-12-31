
// nvcc - o group_gemm_bench group_gemm_bench.cu -lcublas
//./ group_gemm_bench

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

using bf16 = __nv_bfloat16;

// Error checking helper
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

#include "examples/matmul/group_gemm.cuh" // group GEMM implementation
#include "examples/matmul/matmul_10.cuh"

// Random number generator
std::default_random_engine generator(69);
cublasHandle_t cublas_handle;

// Helper functions from better_bench.cu
void randomize_matrix(bf16 *mat, int N) {
  std::normal_distribution<float> distribution(0, 1);
  for (int i = 0; i < N; i++) {
    mat[i] = distribution(generator);
  }
}

bool verify_matrix(bf16 *matRef, bf16 *matOut, int N) {
  double max_diff = 0.0;
  for (int i = 0; i < N; i++) {
    double diff = std::fabs(__bfloat162float(matRef[i] - matOut[i]));
    max_diff = std::max(max_diff, diff);
    if (diff > 0.1) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
      return false;
    }
  }
  printf("Maximum difference: %f\n", max_diff);
  return true;
}

// cuBLAS reference implementation
void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t status =
      cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                   CUDA_R_16BF, N, A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N,
                   CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS error: " << status << std::endl;
    exit(1);
  }
}

// Structure to hold test cases
struct TestCase {
  int M, N, K;
  std::vector<bf16> A;
  std::vector<bf16> B;
  std::vector<bf16> C;
  std::vector<bf16> C_ref;
  bf16 *d_A = nullptr;
  bf16 *d_B = nullptr;
  bf16 *d_C = nullptr;
  bf16 *d_C_ref = nullptr;

  TestCase(int m, int n, int k) : M(m), N(n), K(k) {
    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    // Host memory
    A.resize(sizeA);
    B.resize(sizeB);
    C.resize(sizeC);
    C_ref.resize(sizeC);

    // Initialize with random data
    randomize_matrix(A.data(), sizeA);
    randomize_matrix(B.data(), sizeB);

    // Device memory
    cudaCheck(cudaMalloc(&d_A, sizeA * sizeof(bf16)));
    cudaCheck(cudaMalloc(&d_B, sizeB * sizeof(bf16)));
    cudaCheck(cudaMalloc(&d_C, sizeC * sizeof(bf16)));
    cudaCheck(cudaMalloc(&d_C_ref, sizeC * sizeof(bf16)));

    // Copy to device
    cudaCheck(cudaMemcpy(d_A, A.data(), sizeA * sizeof(bf16),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, B.data(), sizeB * sizeof(bf16),
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

void run_benchmark(const std::vector<std::tuple<int, int, int>> &sizes,
                   int batch_size) {
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Create test cases
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

  // Initialize GroupGEMM
  groupgemm::GroupGemm<> group_gemm;
  group_gemm.initializeBatch(batch_params);

  // Warmup phase
  printf("\nWarmup phase...\n");
  for (int i = 0; i < 3; i++) {
    group_gemm.launch();
    cudaCheck(cudaDeviceSynchronize());
  }

  // Verification phase
  printf("\nVerification phase...\n");
  for (size_t i = 0; i < test_cases.size(); i++) {
    auto &test = test_cases[i];

    // Run cuBLAS reference
    runCublasGemmBF16(test.M, test.N, test.K, test.d_A, test.d_B, test.d_C_ref);

    // Run our implementation
    group_gemm.launch();
    cudaCheck(cudaDeviceSynchronize());

    // Verify results
    cudaCheck(cudaMemcpy(test.C.data(), test.d_C,
                         test.M * test.N * sizeof(bf16),
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(test.C_ref.data(), test.d_C_ref,
                         test.M * test.N * sizeof(bf16),
                         cudaMemcpyDeviceToHost));

    printf("Verifying GEMM %zu (M=%d, N=%d, K=%d): ", i, test.M, test.N,
           test.K);
    if (!verify_matrix(test.C_ref.data(), test.C.data(), test.M * test.N)) {
      printf("FAILED verification!\n");
      return;
    }
    printf("PASSED verification\n");
  }

  // Benchmark phase
  printf("\nBenchmark phase...\n");
  const int NUM_ITERS = 100;

  // Benchmark cuBLAS
  cudaEventRecord(start);
  for (int iter = 0; iter < NUM_ITERS; iter++) {
    for (const auto &test : test_cases) {
      runCublasGemmBF16(test.M, test.N, test.K, test.d_A, test.d_B,
                        test.d_C_ref);
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);

  double cublas_time = elapsed_time / NUM_ITERS;
  double cublas_tflops = 0;
  for (const auto &test : test_cases) {
    cublas_tflops += (2.0 * test.M * test.N * test.K) * 1e-12;
  }
  cublas_tflops = (cublas_tflops * 1000) / cublas_time; // Convert to TFLOPS

  printf("cuBLAS batch time: %f ms, Performance: %f TFLOPS\n", cublas_time,
         cublas_tflops);

  // Benchmark GroupGEMM
  cudaEventRecord(start);
  for (int iter = 0; iter < NUM_ITERS; iter++) {
    group_gemm.launch();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);

  double group_time = elapsed_time / NUM_ITERS;
  double group_tflops = (cublas_tflops * cublas_time) / group_time;
  double speedup = cublas_time / group_time;

  printf(
      "GroupGEMM batch time: %f ms, Performance: %f TFLOPS (%.2fx vs cuBLAS)\n",
      group_time, group_tflops, speedup);
}

int main(int argc, char *argv[]) {
  // Initialize cuBLAS
  cublasCreate(&cublas_handle);

  // Test configurations
  std::vector<std::tuple<int, int, int>> test_sizes = {{4096, 4096, 4096},
                                                       {3072, 4096, 2048},
                                                       {2048, 2048, 2048},
                                                       {4096, 2048, 3072}};

  // Run benchmark with different batch sizes
  std::vector<int> batch_sizes = {1, 2, 4};
  for (int batch_size : batch_sizes) {
    printf("\n=== Testing with batch size %d ===\n", batch_size);
    std::vector<std::tuple<int, int, int>> batch_config;
    for (int i = 0; i < batch_size && i < test_sizes.size(); i++) {
      batch_config.push_back(test_sizes[i]);
    }
    run_benchmark(batch_config, batch_size);
  }

  // Cleanup
  cublasDestroy(cublas_handle);
  return 0;
}
