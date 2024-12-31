#pragma once
#include "wgmax.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

namespace groupgemm {

using namespace wgmma_utils;
using bf16 = __nv_bfloat16;

// Structure to hold the parameters for each GEMM in the batch
struct GemmParams {
  int M;       // Matrix A rows
  int N;       // Matrix B columns
  int K;       // Matrix A columns/Matrix B rows
  bf16 *A;     // Input matrix A
  bf16 *B;     // Input matrix B
  bf16 *C;     // Output matrix C
  float alpha; // Scale factor for AB
  float beta;  // Scale factor for C
};

// Main GroupGEMM class for handling batched operations
template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int NUM_SM = 128, int CLUSTER_M = 2, int CLUSTER_N = 1>
class GroupGemm {
private:
  // Host-side storage for batch information
  std::vector<GemmParams> h_params;
  GemmParams *d_params;

  // Device-side storage for TMA descriptors
  struct BatchTMADescriptors {
    CUtensorMap tma_A;
    CUtensorMap tma_B;
    CUtensorMap tma_C;
  };
  std::vector<BatchTMADescriptors> h_tma_descriptors;
  BatchTMADescriptors *d_tma_descriptors;

  int batch_size;

public:
  GroupGemm() : d_params(nullptr), d_tma_descriptors(nullptr), batch_size(0) {}

  ~GroupGemm() {
    if (d_params)
      cudaFree(d_params);
    if (d_tma_descriptors)
      cudaFree(d_tma_descriptors);
  }

  // Initialize a batch of GEMMs
  void initializeBatch(const std::vector<GemmParams> &params) {
    batch_size = params.size();
    h_params = params;
    h_tma_descriptors.resize(batch_size);

    // Create TMA descriptors for each GEMM in the batch
    for (int i = 0; i < batch_size; i++) {
      const auto &param = params[i];
      h_tma_descriptors[i].tma_A = TensorMapManager::create_tensor_map<BM, BK>(
          param.A, param.M, param.K);
      h_tma_descriptors[i].tma_B = TensorMapManager::create_tensor_map<BN, BK>(
          param.B, param.N, param.K);
      h_tma_descriptors[i].tma_C =
          TensorMapManager::create_tensor_map<BN, BM, false>(param.C, param.N,
                                                             param.M);
    }

    // Allocate and copy batch information to device
    if (d_params)
      cudaFree(d_params);
    if (d_tma_descriptors)
      cudaFree(d_tma_descriptors);

    cudaMalloc(&d_params, batch_size * sizeof(GemmParams));
    cudaMalloc(&d_tma_descriptors, batch_size * sizeof(BatchTMADescriptors));

    cudaMemcpy(d_params, h_params.data(), batch_size * sizeof(GemmParams),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_tma_descriptors, h_tma_descriptors.data(),
               batch_size * sizeof(BatchTMADescriptors),
               cudaMemcpyHostToDevice);
  }

  // Main compute kernel adapted for batch processing
  __global__
  __launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M *CLUSTER_N, 1,
                                                       1)
      computeBatch(const GemmParams *params,
                   const BatchTMADescriptors *tma_descs, int batch_size) {
    // Get batch index from block ID
    int batch_idx = blockIdx.x / ((NUM_SM + batch_size - 1) / batch_size);
    if (batch_idx >= batch_size)
      return;

    // Adjust block index for current batch
    int adjusted_block_idx =
        blockIdx.x % ((NUM_SM + batch_size - 1) / batch_size);

    // Get parameters for current GEMM
    const auto &cur_params = params[batch_idx];
    const auto &cur_tma = tma_descs[batch_idx];

    extern __shared__ __align__(128) uint8_t smem[];
    SharedMemoryLayout<BM, BN, BK, QSIZE> &s =
        *reinterpret_cast<SharedMemoryLayout<BM, BN, BK, QSIZE> *>(smem);

    // Initialize synchronization
    if (threadIdx.x == 0) {
      for (int i = 0; i < QSIZE; ++i) {
        PTXBarrier::init_barrier(&s.full[i], 0, 1);
        PTXBarrier::init_barrier(
            &s.empty[i], 0, (NUM_THREADS / 128 - 1) * (CLUSTER_M * CLUSTER_N));
      }
    }
    ClusterOps::cluster_sync();

    Schedule<1, NUM_SM / (CLUSTER_M * CLUSTER_N), BM * CLUSTER_M,
             BN * CLUSTER_N, 16 / CLUSTER_M, 8 / CLUSTER_N>
        schedule(cur_params.M, cur_params.N, adjusted_block_idx);

    uint32_t cluster_rank = ClusterOps::get_cluster_rank();
    uint32_t rank_m = cluster_rank / CLUSTER_N;
    uint32_t rank_n = cluster_rank % CLUSTER_N;

    // Split work between producer and consumer threads
    int wg_idx = threadIdx.x / 128;
    int tid = threadIdx.x % 128;

    if (wg_idx == 0) {
      producerThread(s, schedule, rank_m, rank_n, tid, cur_params, cur_tma);
    } else {
      consumerThread(s, schedule, rank_m, rank_n, wg_idx - 1, tid, cur_params,
                     cur_tma);
    }
  }

  // Launch the batch computation
  void launch() {
    dim3 grid(NUM_SM); // We'll adjust this in the kernel based on batch size
    dim3 block(NUM_THREADS);
    size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);

    computeBatch<<<grid, block, smem_size>>>(d_params, d_tma_descriptors,
                                             batch_size);
  }

private:
  // Producer and consumer thread implementations using
  // batch parameters
  template <typename SchedType>
  __device__ void
  producerThread(SharedMemoryLayout<BM, BN, BK, QSIZE> &s, SchedType &schedule,
                 uint32_t rank_m, uint32_t rank_n, int tid,
                 const GemmParams &params, const BatchTMADescriptors &tma) {
    if (tid == 0) {
      int p = 0;
      int qidx = 0;
      uint32_t col_mask = 0;
      for (int i = 0; i < CLUSTER_M; ++i) {
        col_mask |= (1 << (i * CLUSTER_N));
      }

      int num_block_m, num_block_n;
      while (schedule.next(num_block_m, num_block_n)) {
        num_block_n = num_block_n * CLUSTER_N + rank_n;
        num_block_m = num_block_m * CLUSTER_M + rank_m;

        for (int block_k_iter = 0; block_k_iter < params.K / BK;
             ++block_k_iter, ++qidx) {
          if (qidx == QSIZE) {
            qidx = 0;
            p ^= 1;
          }
          PTXBarrier::wait(&s.empty[qidx], p);

          PTXBarrier::expect_bytes_tx(&s.full[qidx],
                                      (BK * BN + BK * BM) * sizeof(bf16));

          if constexpr (CLUSTER_N > 1) {
            uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
            if (rank_n == 0) {
              TMAOps::load_async_multicast(&s.A[qidx * BK * BM], &tma.tma_A,
                                           &s.full[qidx], block_k_iter * BK,
                                           num_block_m * BM, mask);
            }
          } else {
            TMAOps::load_async(&s.A[qidx * BK * BM], &tma.tma_A, &s.full[qidx],
                               block_k_iter * BK, num_block_m * BM);
          }

          if constexpr (CLUSTER_M > 1) {
            if (rank_m == 0) {
              TMAOps::load_async_multicast(
                  &s.B[qidx * BK * BN], &tma.tma_B, &s.full[qidx],
                  block_k_iter * BK, num_block_n * BN, col_mask << rank_n);
            }
          } else {
            TMAOps::load_async(&s.B[qidx * BK * BN], &tma.tma_B, &s.full[qidx],
                               block_k_iter * BK, num_block_n * BN);
          }
        }
      }
    }
  }
  template <typename SchedType>
  __device__ void consumerThread(SharedMemoryLayout<BM, BN, BK, QSIZE> &s,
                                 SchedType &schedule, uint32_t rank_m,
                                 uint32_t rank_n, int wg_idx, int tid) {
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_K = 16;
    constexpr int WGMMA_N = BN;
    constexpr int num_consumers = (NUM_THREADS / 128) - 1;
    constexpr int B_WG_M = BM / num_consumers;

    RegisterManager::warpgroup_reg_alloc<
        num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160)>();

    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
    int p = 0;
    int qidx = 0;

    // Process tiles
    int num_block_m, num_block_n;
    while (schedule.next(num_block_m, num_block_n)) {
      num_block_n = num_block_n * CLUSTER_N + rank_n;
      num_block_m = num_block_m * CLUSTER_M + rank_m;

      // Main computation loop using WGMMA operations
      for (int block_k_iter = 0; block_k_iter < K / BK;
           ++block_k_iter, ++qidx) {
        if (qidx == QSIZE) {
          qidx = 0;
          p ^= 1;
        }

        PTXBarrier::wait(&s.full[qidx], p);
        WGMMASyncOps::warpgroup_arrive();

        // Compute using WGMMA operations
        computeWGMMA<WGMMA_N, WGMMA_M, WGMMA_K>(
            d, &s.A[qidx * BK * BM], &s.B[qidx * BK * BN], wg_idx, B_WG_M);

        if (tid < (CLUSTER_M * CLUSTER_N))
          PTXBarrier::arrive_cluster(&s.empty[qidx], tid);
      }

      // Store results
      storeResults(s.C, d, num_block_m, num_block_n, wg_idx, tid);
    }
  }

  template <int WGMMA_N, int WGMMA_M, int WGMMA_K>
  __device__ void computeWGMMA(float d[][WGMMA_N / 16][8], bf16 *wgmma_sA,
                               bf16 *wgmma_sB, int wg_idx, int B_WG_M) {
#pragma unroll
    for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
      bf16 *cur_sA =
          wgmma_sA + 64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
      bf16 *cur_sB = wgmma_sB;

      // Initial WGMMA compute
      wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[m_it], &cur_sA[0], &cur_sB[0]);

// Remaining iterations
#pragma unroll
      for (int k_it = 1; k_it < 64 / WGMMA_K; ++k_it) {
        wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &cur_sA[k_it * WGMMA_K],
                                      &cur_sB[k_it * WGMMA_K]);
      }
    }
  }

  __device__ void storeResults(bf16 *sC, float d[][BN / 16][8], int block_m,
                               int block_n, int wg_idx, int tid) {
    WGMMAGlobalStore::wait_previous();

    int lane = tid % 32;
    int warp = tid / 32;
    int row = warp * 16 + lane / 4;
    bf16 *block_sC = sC + wg_idx * (BM / (NUM_THREADS / 128 - 1)) * BN;

    // Store results to shared memory
    constexpr int B_WG_M = BM / ((NUM_THREADS / 128) - 1);
#pragma unroll
    for (int m_it = 0; m_it < B_WG_M / 64; ++m_it) {
      int yo = m_it * 64;
#pragma unroll
      for (int w = 0; w < BN; w += 16) {
        int col = w + 2 * (tid % 4);
#define ST(i, j, v) block_sC[(j) * B_WG_M + (i) + yo] = v

        ST(row, col, d[m_it][w / 16][0]);
        ST(row + 8, col, d[m_it][w / 16][2]);
        ST(row, col + 1, d[m_it][w / 16][1]);
        ST(row + 8, col + 1, d[m_it][w / 16][3]);
        ST(row, col + 8, d[m_it][w / 16][4]);
        ST(row + 8, col + 8, d[m_it][w / 16][6]);
        ST(row, col + 9, d[m_it][w / 16][5]);
        ST(row + 8, col + 9, d[m_it][w / 16][7]);

#undef ST
      }
    }

    WGMMAGlobalStore::sync_threads();

    if (threadIdx.x == 128) {
      TMAOps::store_async(&d_tma_map_C, sC, block_m * BM, block_n * BN);
      WGMMAGlobalStore::commit_group();
    }
  }
};

} // namespace groupgemm
