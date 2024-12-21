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

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

#include "examples/matmul/matmul_1.cuh"
#include "examples/matmul/matmul_10.cuh"
#include "examples/matmul/matmul_11.cuh"
#include "examples/matmul/matmul_2.cuh"
#include "examples/matmul/matmul_3.cuh"
#include "examples/matmul/matmul_4.cuh"
#include "examples/matmul/matmul_5.cuh"
#include "examples/matmul/matmul_6.cuh"
#include "examples/matmul/matmul_7.cuh"
#include "examples/matmul/matmul_8.cuh"
#include "examples/matmul/matmul_9.cuh"

std::default_random_engine generator(69);
cublasHandle_t cublas_handle;

void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status =
      cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A,
                   CUDA_R_16BF, N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS error: " << status << std::endl;
    exit(1);
  }
}

void run_kernel(int kernel_num, int M, int N, int K, bf16 *A, bf16 *B, bf16 *C,
                int *DB = nullptr) {
  switch (kernel_num) {
  case 0:
    runCublasGemmBF16(M, N, K, A, B, C);
    break;
  case 1:
    runKernel1(M, N, K, A, B, C);
    break;
  case 2:
    runKernel2(M, N, K, A, B, C);
    break;
  case 3:
    runKernel3(M, N, K, A, B, C, DB);
    break;
  case 4:
    runKernel4(M, N, K, A, B, C, DB);
    break;
  case 5:
    runKernel5(M, N, K, A, B, C, DB);
    break;
  case 6:
    runKernel6(M, N, K, A, B, C, DB);
    break;
  case 7:
    runKernel7(M, N, K, A, B, C, DB);
    break;
  case 8:
    runKernel8(M, N, K, A, B, C, DB);
    break;
  case 9:
    runKernel9(M, N, K, A, B, C, DB);
    break;
  case 10:
    runKernel10(M, N, K, A, B, C, DB);
    break;
  case 11:
    runKernel11(M, N, K, A, B, C, DB);
    break;
  }
}

void randomize_matrix(bf16 *mat, int N) {
  std::normal_distribution<float> distribution(0, 1);
  for (int i = 0; i < N; i++) {
    mat[i] = distribution(generator);
  }
}

bool verify_matrix(bf16 *matRef, bf16 *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    int r = i / 8192, c = i % 8192;
    int it = c * 8192 + r;
    diff = std::fabs(__bfloat162float(matRef[i] - matOut[i]));
    if (diff > 0.1) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             __bfloat162float(matRef[i]), __bfloat162float(matOut[i]), diff, i);
      return false;
    }
  }
  return true;
}

__global__ void warmupKernel() {
  __shared__ int s[100];
  s[0] += s[1];
}

void print_usage() {
  std::cout
      << "Usage: ./program [kernel_num] [matrix_size]\n"
      << "  kernel_num: Optional. Number between 0-11. Default: run all "
         "kernels\n"
      << "  matrix_size: Optional. Size of the square matrix.\n"
      << "               Use -1 for comparison set (4096,5120,8192,16384)\n"
      << "               Default: 8192\n";
}

void run_benchmark(long matrix_size, const std::vector<int> &kernels_to_run,
                   bool append_results = false) {
  float elapsed_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  long m = matrix_size, n = matrix_size, k = matrix_size;

  bf16 *A = nullptr, *B = nullptr, *C = nullptr,
       *C_ref = nullptr; // host matrices
  bf16 *dA = nullptr, *dB = nullptr, *dC = nullptr,
       *dC_ref = nullptr; // device matrices

  int *DB = nullptr;
  int *dDB = nullptr;

  A = (bf16 *)malloc(sizeof(bf16) * matrix_size * matrix_size);
  B = (bf16 *)malloc(sizeof(bf16) * matrix_size * matrix_size);
  C = (bf16 *)malloc(sizeof(bf16) * matrix_size * matrix_size);
  C_ref = (bf16 *)malloc(sizeof(bf16) * matrix_size * matrix_size);
  DB = (int *)malloc(sizeof(int) * matrix_size * 128);
  cudaCheck(cudaMalloc((void **)&dDB, sizeof(int) * matrix_size * 128));

  randomize_matrix(A, matrix_size * matrix_size);
  randomize_matrix(B, matrix_size * matrix_size);
  randomize_matrix(C, matrix_size * matrix_size);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(bf16) * matrix_size * matrix_size));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(bf16) * matrix_size * matrix_size));
  cudaCheck(cudaMalloc((void **)&dC, sizeof(bf16) * matrix_size * matrix_size));
  cudaCheck(
      cudaMalloc((void **)&dC_ref, sizeof(bf16) * matrix_size * matrix_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(bf16) * matrix_size * matrix_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(bf16) * matrix_size * matrix_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 8;
  bool run_verif = true;

  // Create/append to results file with header
  if (!append_results) {
    if (FILE *csv = fopen("benchmark_results.csv", "w")) {
      fprintf(csv, "kernel,matrix_size,time,tflops,relative_perf\n");
      fclose(csv);
    }
  }

  // Store cuBLAS performance for relative comparison
  float cublas_tflops = 0.0;

  for (int kernel_num : kernels_to_run) {
    // Give the GPU some rest to avoid thermal throttling
    sleep(5);
    std::cout << "\nKERNEL " << kernel_num << " (Matrix size: " << matrix_size
              << "x" << matrix_size << ")" << std::endl;

    // Warmup phase - run multiple warmup iterations
    memset(C, 0, sizeof(bf16) * matrix_size * matrix_size);
    cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * matrix_size * matrix_size,
                         cudaMemcpyHostToDevice));

    // Run multiple warmup iterations to ensure steady state
    const int warmup_iterations = 3;
    for (int w = 0; w < warmup_iterations; w++) {
      run_kernel(kernel_num, m, n, k, dA, dB, dC);
      // cudaCheck(cudaDeviceSynchronize());
    }

    // Additional sync to ensure warmup is complete
    cudaCheck(cudaDeviceSynchronize());

    // Verification phase
    if (run_verif && kernel_num != 0) {
      cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * matrix_size * matrix_size,
                           cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(dC_ref, C, sizeof(bf16) * matrix_size * matrix_size,
                           cudaMemcpyHostToDevice));
      memset(DB, ~0, sizeof(int) * matrix_size * 128);
      cudaCheck(cudaMemcpy(dDB, DB, sizeof(int) * matrix_size * 128,
                           cudaMemcpyHostToDevice));

      run_kernel(0, m, n, k, dA, dB, dC_ref);           // cuBLAS reference
      run_kernel(kernel_num, m, n, k, dA, dB, dC, dDB); // Test kernel
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError());
      cudaMemcpy(C, dC, sizeof(bf16) * matrix_size * matrix_size,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(bf16) * matrix_size * matrix_size,
                 cudaMemcpyDeviceToHost);

      if (kernel_num > 1 && !verify_matrix(C_ref, C, m * n)) {
        std::cout << "~~~~~~~~~~~~~~~~ Failed to pass the correctness "
                     "verification against cuBLAS. ~~~~~~~~~~~~~~~~"
                  << std::endl;
        printf("%f\n", __bfloat162float(C_ref[m]));
      }

      // Performance counters analysis
      cudaMemcpy(DB, dDB, sizeof(int) * matrix_size * 8,
                 cudaMemcpyDeviceToHost);
      int i = 0;
      long sumLoad = 0, cntLoad = 0;
      long sumCompute = 0, cntCompute = 0;
      long sumStore = 0, cntStore = 0;
      int times = 0;
      while (DB[i] != ~0) {
        sumLoad += DB[i], cntLoad += DB[i + 1];
        sumCompute += DB[i + 2], cntCompute += DB[i + 3];
        sumStore += DB[i + 4], cntStore += DB[i + 5];
        i += 6;
        times++;
      }
      if (times > 0) {
        printf("Load: %f, Compute: %f, Store: %f, Datapoints: %d\n",
               (sumLoad + .0) / cntLoad, (sumCompute + .0) / cntCompute,
               (sumStore + .0) / cntStore, times);
      }
    }

    // Benchmark
    cudaEventRecord(start);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, dA, dB, dC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    long flops = (2LL * m) * (n * k);
    float current_tflops = (repeat_times * flops * 1e-9) / elapsed_time;

    // Store cuBLAS performance for comparison
    if (kernel_num == 0) {
      cublas_tflops = current_tflops;
    }

    // Calculate relative performance
    char perf_str[64] = "";
    if (kernel_num != 0 && cublas_tflops > 0) {
      float perf_ratio = ((current_tflops / cublas_tflops) - 1.0) * 100;
      snprintf(perf_str, sizeof(perf_str), " (%+.1f%% vs cuBLAS)", perf_ratio);
    }

    // Print human-readable output
    printf("Matrix Size: %ld x %ld, Average elapsed time: (%7.6f) s, "
           "performance: (%7.1f) TFLOPS%s\n",
           matrix_size, matrix_size, elapsed_time / 1000.0 / repeat_times,
           current_tflops, perf_str);

    // Print CSV format for plotting
    if (FILE *csv = fopen("benchmark_results.csv", "a")) {
      float relative_perf = 0.0;
      if (kernel_num != 0 && cublas_tflops > 0) {
        relative_perf = ((current_tflops / cublas_tflops) - 1.0) * 100;
      }
      fprintf(csv, "%d,%ld,%f,%f,%f\n", kernel_num, matrix_size,
              elapsed_time / 1000.0 / repeat_times, current_tflops,
              relative_perf);
      fclose(csv);
    }
  }

  // Cleanup
  free(A);
  free(B);
  free(C);
  free(C_ref);
  free(DB);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cudaFree(dDB);
}

int main(int argc, char *argv[]) {
  warmupKernel<<<1024, 1024>>>();

  // Parse command line arguments
  int specific_kernel = -1;
  long matrix_size = 8192;

  if (argc > 1) {
    specific_kernel = atoi(argv[1]);
    if (specific_kernel < -1 || specific_kernel > 11) {
      std::cout
          << "Invalid kernel number. Please specify a number between -1 and 11."
          << std::endl;
      print_usage();
      return 1;
    }
  }

  if (argc > 2) {
    matrix_size = atol(argv[2]);
    if (matrix_size <= 0 && matrix_size != -1) {
      std::cout << "Invalid matrix size. Please specify a positive number or "
                   "-1 for comparison set."
                << std::endl;
      print_usage();
      return 1;
    }
  }

  cublasCreate(&cublas_handle);

  std::vector<int> kernels_to_run;
  if (specific_kernel >= 0) {
    kernels_to_run = {
        0, specific_kernel}; // Always include cuBLAS (0) for reference
  } else if (specific_kernel == -1) {
    kernels_to_run = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  }

  if (matrix_size == -1) {
    // Run the comparison set
    std::vector<long> sizes = {4096, 5120, 8192, 16384};
    for (size_t i = 0; i < sizes.size(); i++) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Running benchmark with matrix size: " << sizes[i] << "x"
                << sizes[i] << std::endl;
      std::cout << "========================================" << std::endl;
      run_benchmark(sizes[i], kernels_to_run,
                    i > 0); // append results after first run
    }
  } else {
    // Run single size
    std::cout << "Running benchmark with matrix size: " << matrix_size << "x"
              << matrix_size << std::endl;
    run_benchmark(matrix_size, kernels_to_run);
  }

  cublasDestroy(cublas_handle);
  return 0;
}
