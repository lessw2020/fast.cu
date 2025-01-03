// group_matmul_test.cu

#include <algorithm>
#include <cassert>
#include <chrono>
#include <ctime>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

typedef __nv_bfloat16 bf16;
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#include "examples/matmul/group_gemm.cuh"

// Helper macro for CUDA error checking
#define cudaCheck(err)                                                         \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess)                                                   \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err_) + " at " + __FILE__ +  \
                               ":" + std::to_string(__LINE__));                \
  }

namespace GroupMatmulTest {

void runCublasGemmBF16(cublasHandle_t handle, int M, int N, int K,
                       const bf16 *A, const bf16 *B, bf16 *C) {
  float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t status =
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                   CUDA_R_16BF, N, A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("CUBLAS GEMM failed: " + std::to_string(status));
  }
}

void randomize_matrix(bf16 *mat, int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> distribution(0.0f, 1.0f);

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    float val = distribution(gen);
    // Ensure values are in a reasonable range for numerical stability
    val = std::min(std::max(val, -10.0f), 10.0f);
    mat[i] = __float2bfloat16(val);
  }
}

struct MatrixResources {
  std::vector<bf16 *> h_A, h_B, h_C, h_C_ref;
  std::vector<bf16 *> d_A, d_B, d_C, d_C_ref;
  std::vector<size_t> matrix_bytes;
  int group_size;
  bool verify;

  MatrixResources(const std::vector<int> &matrix_sizes, bool do_verify)
      : group_size(matrix_sizes.size()), verify(do_verify) {
    try {
      h_A.resize(group_size);
      h_B.resize(group_size);
      h_C.resize(group_size);
      d_A.resize(group_size);
      d_B.resize(group_size);
      d_C.resize(group_size);
      matrix_bytes.resize(group_size);

      if (verify) {
        h_C_ref.resize(group_size);
        d_C_ref.resize(group_size);
      }

      for (int i = 0; i < group_size; i++) {
        matrix_bytes[i] = static_cast<size_t>(matrix_sizes[i]) *
                          matrix_sizes[i] * sizeof(bf16);

        // Host allocations
        h_A[i] = new bf16[matrix_sizes[i] * matrix_sizes[i]];
        h_B[i] = new bf16[matrix_sizes[i] * matrix_sizes[i]];
        h_C[i] = new bf16[matrix_sizes[i] * matrix_sizes[i]];
        if (verify)
          h_C_ref[i] = new bf16[matrix_sizes[i] * matrix_sizes[i]];

        // Device allocations
        cudaCheck(cudaMalloc(&d_A[i], matrix_bytes[i]));
        cudaCheck(cudaMalloc(&d_B[i], matrix_bytes[i]));
        cudaCheck(cudaMalloc(&d_C[i], matrix_bytes[i]));
        if (verify)
          cudaCheck(cudaMalloc(&d_C_ref[i], matrix_bytes[i]));

        // Initialize data
        randomize_matrix(h_A[i], matrix_sizes[i] * matrix_sizes[i]);
        randomize_matrix(h_B[i], matrix_sizes[i] * matrix_sizes[i]);

        // Copy to device
        cudaCheck(cudaMemcpy(d_A[i], h_A[i], matrix_bytes[i],
                             cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_B[i], h_B[i], matrix_bytes[i],
                             cudaMemcpyHostToDevice));
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  void cleanup() {
    for (int i = 0; i < group_size; i++) {
      delete[] h_A[i];
      delete[] h_B[i];
      delete[] h_C[i];
      if (verify)
        delete[] h_C_ref[i];

      cudaFree(d_A[i]);
      cudaFree(d_B[i]);
      cudaFree(d_C[i]);
      if (verify)
        cudaFree(d_C_ref[i]);
    }
  }

  ~MatrixResources() { cleanup(); }
};

class BenchmarkResults {
public:
  float avg_time;
  float min_time;
  float max_time;
  float total_tflops;
  std::vector<float> per_matrix_tflops;
  bool verification_passed;

  void print() const {
    std::cout << "\nBenchmark Results:\n";
    std::cout << "Average time: " << avg_time << " ms\n";
    std::cout << "Min time: " << min_time << " ms\n";
    std::cout << "Max time: " << max_time << " ms\n";
    std::cout << "Total Performance: " << total_tflops << " TFLOPS\n";

    std::cout << "\nPer-Matrix Performance:\n";
    for (size_t i = 0; i < per_matrix_tflops.size(); i++) {
      std::cout << "Matrix " << i << ": " << per_matrix_tflops[i]
                << " TFLOPS\n";
    }

    if (verification_passed) {
      std::cout << "\nVerification: PASSED\n";
    }
  }
};

class Benchmark {
public:
  static BenchmarkResults run(const std::vector<int> &matrix_sizes,
                              int num_iterations = 10, bool verify = true) {
    BenchmarkResults results;
    results.verification_passed = false;

    cublasHandle_t cublas_handle;
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create cuBLAS handle");
    }

    // Initialize resources
    MatrixResources resources(matrix_sizes, verify);
    std::vector<GroupM10::GEMMParams> params(matrix_sizes.size());

    // Calculate total FLOPs
    double total_flops = 0.0;
    for (int size : matrix_sizes) {
      total_flops += 2.0 * static_cast<double>(size) * size * size;
    }

    // Setup GEMM parameters
    for (size_t i = 0; i < matrix_sizes.size(); i++) {
      params[i] = {matrix_sizes[i],  matrix_sizes[i],  matrix_sizes[i],
                   resources.d_A[i], resources.d_B[i], resources.d_C[i]};
    }

    // Warmup run
    GroupM10::runGroupMatmul(params);
    cudaCheck(cudaDeviceSynchronize());

    if (verify) {
      // Run reference computations
      for (size_t i = 0; i < matrix_sizes.size(); i++) {
        runCublasGemmBF16(cublas_handle, matrix_sizes[i], matrix_sizes[i],
                          matrix_sizes[i], resources.d_A[i], resources.d_B[i],
                          resources.d_C_ref[i]);
      }
      cudaCheck(cudaDeviceSynchronize());

      // Run group GEMM
      GroupM10::runGroupMatmul(params);
      cudaCheck(cudaDeviceSynchronize());

      // Verify results
      bool passed = true;
      constexpr float tolerance = 0.1f;
      constexpr float rtol = 1e-3f; // relative tolerance

      for (size_t i = 0; i < matrix_sizes.size(); i++) {
        cudaCheck(cudaMemcpy(resources.h_C[i], resources.d_C[i],
                             resources.matrix_bytes[i],
                             cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(resources.h_C_ref[i], resources.d_C_ref[i],
                             resources.matrix_bytes[i],
                             cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int j = 0; j < matrix_sizes[i] * matrix_sizes[i]; j++) {
          float ref = __bfloat162float(resources.h_C_ref[i][j]);
          float val = __bfloat162float(resources.h_C[i][j]);
          float abs_diff = std::fabs(ref - val);
          float rel_diff = ref != 0.0f ? abs_diff / std::fabs(ref) : abs_diff;

          if (abs_diff > tolerance && rel_diff > rtol) {
            if (errors < 10) { // Only print first 10 errors
              printf("Matrix %zu: Error at index %d. "
                     "Expected %f, Got %f (abs_diff: %f, rel_diff: %f)\n",
                     i, j, ref, val, abs_diff, rel_diff);
            }
            errors++;
            if (errors > 100) { // Early exit if too many errors
              passed = false;
              break;
            }
          }
        }
        if (errors > 0) {
          printf("Matrix %zu: Total %d errors found\n", i, errors);
          passed = false;
        }
      }
      results.verification_passed = passed;
    }

    // Performance measurement
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    std::vector<float> times;
    times.reserve(num_iterations);

    for (int iter = 0; iter < num_iterations; iter++) {
      cudaCheck(cudaEventRecord(start));

      GroupM10::runGroupMatmul(params);

      cudaCheck(cudaEventRecord(stop));
      cudaCheck(cudaEventSynchronize(stop));

      float milliseconds = 0;
      cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
      times.push_back(milliseconds);
    }

    // Calculate statistics
    results.avg_time = 0.0f;
    results.min_time = times[0];
    results.max_time = times[0];

    for (float t : times) {
      results.avg_time += t;
      results.min_time = std::min(results.min_time, t);
      results.max_time = std::max(results.max_time, t);
    }
    results.avg_time /= num_iterations;

    // Calculate TFLOPS
    results.total_tflops = (total_flops * 1e-12) / (results.avg_time * 1e-3);

    // Calculate per-matrix TFLOPS
    results.per_matrix_tflops.resize(matrix_sizes.size());
    for (size_t i = 0; i < matrix_sizes.size(); i++) {
      int size = matrix_sizes[i];
      double matrix_flops = 2.0 * size * size * size;
      results.per_matrix_tflops[i] =
          (matrix_flops * 1e-12) / (results.avg_time * 1e-3);
    }

    // Cleanup
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cublasDestroy(cublas_handle);

    return results;
  }
};

void run_benchmark_suite() {
  const std::vector<std::vector<int>> test_configs = {
      // Group size 2, same sizes
      {4096, 4096},
      {8192, 8192},
      {16384, 16384},

      // Group size 2, different sizes
      {4096, 8192},
      {8192, 16384},

      // Group size 3, same sizes
      {4096, 4096, 4096},
      {8192, 8192, 8192},

      // Group size 3, mixed sizes
      {4096, 8192, 16384},

      // Group size 4, same sizes
      {4096, 4096, 4096, 4096},
      {8192, 8192, 8192, 8192},

      // Group size 4, mixed sizes
      {4096, 8192, 8192, 16384}};

  std::cout << "\nRunning Group GEMM Benchmark Suite\n";
  std::cout << "===================================\n";

  for (const auto &config : test_configs) {
    std::cout << "\nConfiguration: [";
    for (size_t i = 0; i < config.size(); i++) {
      if (i > 0)
        std::cout << ", ";
      std::cout << config[i];
    }
    std::cout << "]\n";

    try {
      BenchmarkResults results = Benchmark::run(config, 10, true);
      results.print();
    } catch (const std::exception &e) {
      std::cerr << "Error running configuration: " << e.what() << std::endl;
    }
  }
}

} // namespace GroupMatmulTest

int main(int argc, char *argv[]) {
  try {
    // Initialize CUDA
    cudaCheck(cudaFree(0));

    if (argc > 1) {
      // TODO: Add command line parameter handling if needed
      std::cout << "Command line parameters not yet implemented\n";
      return 1;
    }

    // Run default benchmark suite
    GroupMatmulTest::run_benchmark_suite();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
