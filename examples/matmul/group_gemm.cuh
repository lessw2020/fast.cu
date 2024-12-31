#pragma once
#include "wgmax.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

namespace groupgemm {

using namespace wgmma_utils;
using bf16 = __nv_bfloat16;

// Forward declarations
template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM,
          int CLUSTER_M, int CLUSTER_N>
class GroupGemm;

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

// Helper functions for kernel operations
namespace detail {
// WGMMA dispatcher - similar to matmul_10.cuh
template <int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA,
          int TransB>
__device__ __forceinline__ void wgmma(float d[WGMMA_N / 16][8], bf16 *sA,
                                      bf16 *sB) {
  static_assert(WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 ||
                WGMMA_N == 192 || WGMMA_N == 208 || WGMMA_N == 256);
  if constexpr (WGMMA_N == 256)
    wgmma256<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
  if constexpr (WGMMA_N == 192)
    wgmma192<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
  if constexpr (WGMMA_N == 128)
    wgmma128<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
  if constexpr (WGMMA_N == 64)
    wgmma64<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
  if constexpr (WGMMA_N == 32)
    wgmma32<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
}

template <int WGMMA_N, int WGMMA_M, int WGMMA_K>
__device__ void computeWGMMA(float d[][WGMMA_N / 16][8], bf16 *wgmma_sA,
                             bf16 *wgmma_sB, int wg_idx, int B_WG_M) {
#pragma unroll
  for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
    bf16 *cur_sA = wgmma_sA + 64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
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

template <int BM, int BN, int BK, int QSIZE, int CLUSTER_M, int CLUSTER_N>
__device__ void
producerThread(SharedMemoryLayout<BM, BN, BK, QSIZE> &s,
               BlockScheduler &schedule, uint32_t rank_m, uint32_t rank_n,
               int tid, const GemmParams &params,
               const BatchTMAManager::BatchTMADescriptors &tma) {

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
            TMAOps::load_async_multicast(&s.B[qidx * BK * BN], &tma.tma_B,
                                         &s.full[qidx], block_k_iter * BK,
                                         num_block_n * BN, col_mask << rank_n);
          }
        } else {
          TMAOps::load_async(&s.B[qidx * BK * BN], &tma.tma_B, &s.full[qidx],
                             block_k_iter * BK, num_block_n * BN);
        }
      }
    }
  }
}

template <int BM, int BN, int BK, int QSIZE, int CLUSTER_M, int CLUSTER_N>
__device__ void
consumerThread(SharedMemoryLayout<BM, BN, BK, QSIZE> &s,
               BlockScheduler &schedule, uint32_t rank_m, uint32_t rank_n,
               int wg_idx, int tid, const GemmParams &params,
               const BatchTMAManager::BatchTMADescriptors &tma) {

  constexpr int WGMMA_M = 64;
  constexpr int WGMMA_K = 16;
  constexpr int WGMMA_N = BN;
  constexpr int num_consumers = (384 / 128) - 1; // NUM_THREADS = 384
  constexpr int B_WG_M = BM / num_consumers;

  RegisterManager::warpgroup_reg_alloc<
      num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160)>();

  float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8] = {};
  int p = 0;
  int qidx = 0;

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

      PTXBarrier::wait(&s.full[qidx], p);
      WGMMASyncOps::warpgroup_arrive();

      computeWGMMA<WGMMA_N, WGMMA_M, WGMMA_K>(
          d, &s.A[qidx * BK * BM], &s.B[qidx * BK * BN], wg_idx, B_WG_M);

      if (tid < (CLUSTER_M * CLUSTER_N))
        PTXBarrier::arrive_cluster(&s.empty[qidx], tid);
    }

    // Store results
    WGMMAGlobalStore::wait_previous();

    int lane = tid % 32;
    int warp = tid / 32;
    int row = warp * 16 + lane / 4;
    bf16 *block_sC = &s.C[wg_idx * B_WG_M * BN];

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
      TMAOps::store_async(&tma.tma_C, &s.C[0], num_block_m * BM,
                          num_block_n * BN);
      WGMMAGlobalStore::commit_group();
    }
  }
}
} // namespace detail

// CUDA kernel (must be outside the class)
template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM,
          int CLUSTER_M, int CLUSTER_N>
__global__ void
__launch_bounds__(NUM_THREADS) __cluster_dims__(CLUSTER_M *CLUSTER_N, 1, 1)
    computeBatchKernel(const GemmParams *params,
                       const BatchTMAManager::BatchTMADescriptors *tma_descs,
                       int batch_size) {
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
  auto &s = *reinterpret_cast<SharedMemoryLayout<BM, BN, BK, QSIZE> *>(smem);

  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
      PTXBarrier::init_barrier(&s.full[i], 0, 1);
      PTXBarrier::init_barrier(
          &s.empty[i], 0, (NUM_THREADS / 128 - 1) * (CLUSTER_M * CLUSTER_N));
    }
  }
  ClusterOps::cluster_sync();

  auto schedule = BlockScheduler::create(
      cur_params.M, cur_params.N, BM, BN, 16 / CLUSTER_M, 8 / CLUSTER_N,
      NUM_SM / (CLUSTER_M * CLUSTER_N), adjusted_block_idx);

  uint32_t cluster_rank = ClusterOps::get_cluster_rank();
  uint32_t rank_m = cluster_rank / CLUSTER_N;
  uint32_t rank_n = cluster_rank % CLUSTER_N;

  int wg_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;

  if (wg_idx == 0) {
    detail::producerThread<BM, BN, BK, QSIZE, CLUSTER_M, CLUSTER_N>(
        s, schedule, rank_m, rank_n, tid, cur_params, cur_tma);
  } else {
    detail::consumerThread<BM, BN, BK, QSIZE, CLUSTER_M, CLUSTER_N>(
        s, schedule, rank_m, rank_n, wg_idx - 1, tid, cur_params, cur_tma);
  }
}

// Main GroupGEMM class
template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int NUM_SM = 128, int CLUSTER_M = 2, int CLUSTER_N = 1>
class GroupGemm {
private:
  std::vector<GemmParams> h_params;
  GemmParams *d_params;
  std::vector<BatchTMAManager::BatchTMADescriptors> h_tma_descriptors;
  BatchTMAManager::BatchTMADescriptors *d_tma_descriptors;
  int batch_size;

public:
  GroupGemm() : d_params(nullptr), d_tma_descriptors(nullptr), batch_size(0) {}

  ~GroupGemm() {
    if (d_params)
      cudaFree(d_params);
    if (d_tma_descriptors)
      cudaFree(d_tma_descriptors);
  }

  void initializeBatch(const std::vector<GemmParams> &params) {
    batch_size = params.size();
    h_params = params;
    h_tma_descriptors.resize(batch_size);

    // Create TMA descriptors
    for (int i = 0; i < batch_size; i++) {
      const auto &param = params[i];
      h_tma_descriptors[i] = BatchTMAManager::createDescriptors<bf16>(
          param.A, param.B, param.C, param.M, param.N, param.K, BM, BN, BK);
    }

    // Device memory allocation and transfer
    if (d_params)
      cudaFree(d_params);
    if (d_tma_descriptors)
      cudaFree(d_tma_descriptors);

    cudaMalloc(&d_params, batch_size * sizeof(GemmParams));
    cudaMalloc(&d_tma_descriptors,
               batch_size * sizeof(BatchTMAManager::BatchTMADescriptors));

    cudaMemcpy(d_params, h_params.data(), batch_size * sizeof(GemmParams),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_tma_descriptors, h_tma_descriptors.data(),
               batch_size * sizeof(BatchTMAManager::BatchTMADescriptors),
               cudaMemcpyHostToDevice);
  }

  void launch() {
    dim3 grid(NUM_SM); // Will be adjusted in kernel based on batch size
    dim3 block(NUM_THREADS);
    size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);

    computeBatchKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                       CLUSTER_N>
        <<<grid, block, smem_size>>>(d_params, d_tma_descriptors, batch_size);
  }
};

} // namespace groupgemm
