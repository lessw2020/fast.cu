#pragma once
#include "wgmax.cuh"
#include <vector>

namespace groupgemm {
using namespace wgmma_utils;

// WGMMA dispatcher
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

// Per-GEMM parameters
struct GemmParams {
  int M, N, K;
  bf16 *A;
  bf16 *B;
  bf16 *C;
  float alpha;
  float beta;
};

template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;
  int group_idx;
  const GemmParams *params;
  bool valid;

  // Precompute schedule parameters
  int blocks_per_group;
  int tiles_m, tiles_n;

  __device__ __forceinline__ Schedule(int block_idx, int group_size,
                                      const GemmParams *all_params) {
    blocks_per_group = NUM_SM / (TM * TN);
    block = block_idx % blocks_per_group;
    group_idx = block_idx / blocks_per_group;
    valid = (group_idx < group_size);

    if (valid) {
      params = &all_params[group_idx];
      it = 0;
      // Round up to next multiple of block size
      total_blocks_m = (params->M + BM - 1) / BM;
      total_blocks_n = (params->N + BN - 1) / BN;
      // Precompute tile counts
      tiles_m = (total_blocks_m + TM - 1) / TM;
      tiles_n = (total_blocks_n + TN - 1) / TN;
    }
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    if (!valid)
      return false;

    int num = it * blocks_per_group + block;
    int total_tiles = tiles_m * tiles_n;
    if (num >= total_tiles * (TM * TN))
      return false;

    // More efficient tile computation
    int tile_idx = num / (TM * TN);
    int pos_in_tile = num % (TM * TN);

    int tile_m = tile_idx / tiles_n;
    int tile_n = tile_idx % tiles_n;

    block_m = tile_m * TM + (pos_in_tile / TN);
    block_n = tile_n * TN + (pos_in_tile % TN);

    ++it;
    return true;
  }

  __device__ __forceinline__ bool is_valid() const { return valid; }
};

template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int NUM_SM = 128, int CLUSTER_M = 2, int CLUSTER_N = 1>
__global__
__launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M *CLUSTER_N, 1, 1)
    groupGemmKernel(int group_size, const GemmParams *params) {
  constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  constexpr int B_WG_M = BM / num_consumers;
  constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

  extern __shared__ __align__(128) uint8_t smem[];
  SharedMemoryLayout<BM, BN, BK, QSIZE> &s =
      *reinterpret_cast<SharedMemoryLayout<BM, BN, BK, QSIZE> *>(smem);

  bf16 *sA = s.A, *sB = s.B, *sC = s.C;
  uint64_t *full = s.full, *empty = s.empty;

  uint32_t cluster_id = ClusterOps::get_cluster_id();

  Schedule<1, NUM_SM / CLUSTERS, BM * CLUSTER_M, BN * CLUSTER_N, 16 / CLUSTER_M,
           8 / CLUSTER_N>
      schedule(blockIdx.x, group_size, params);

  if (!schedule.is_valid())
    return;

  const int num_blocks_k = (schedule.params->K + BK - 1) / BK;
  int wg_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;

  // Initialize barriers
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < QSIZE; ++i) {
      PTXBarrier::init_barrier(&full[i], 0, 1);
      PTXBarrier::init_barrier(&empty[i], 0, num_consumers * CLUSTERS);
    }
  }

  ClusterOps::cluster_sync();

  uint32_t cluster_rank = ClusterOps::get_cluster_rank();
  uint32_t rank_m = cluster_rank / CLUSTER_N;
  uint32_t rank_n = cluster_rank % CLUSTER_N;

  // Create TMA descriptors for current GEMM with proper padding handling
  CUtensorMap tma_A = TensorMapManager::create_tensor_map<BM, BK>(
      schedule.params->A, schedule.params->M, schedule.params->K);
  CUtensorMap tma_B = TensorMapManager::create_tensor_map<BN, BK>(
      schedule.params->B, schedule.params->N, schedule.params->K);
  CUtensorMap tma_C = TensorMapManager::create_tensor_map<BN, BM, false>(
      schedule.params->C, schedule.params->N, schedule.params->M);

  // Producer thread
  if (wg_idx == 0) {
    if (tid == 0) {
      int p = 0;
      int qidx = 0;
      uint32_t col_mask = 0;
#pragma unroll
      for (int i = 0; i < CLUSTER_M; ++i) {
        col_mask |= (1 << (i * CLUSTER_N));
      }

      int num_block_m, num_block_n;
      while (schedule.next(num_block_m, num_block_n)) {
        num_block_n = num_block_n * CLUSTER_N + rank_n;
        num_block_m = num_block_m * CLUSTER_M + rank_m;

        // Check for matrix bounds
        bool is_valid_block = (num_block_m * BM < schedule.params->M) &&
                              (num_block_n * BN < schedule.params->N);

        if (is_valid_block) {
          for (int block_k_iter = 0; block_k_iter < num_blocks_k;
               ++block_k_iter, ++qidx) {
            if (qidx == QSIZE) {
              qidx = 0;
              p ^= 1;
            }

            PTXBarrier::wait(&empty[qidx], p);
            PTXBarrier::expect_bytes_tx(&full[qidx],
                                        (BK * BN + BK * BM) * sizeof(bf16));

            int k_offset = block_k_iter * BK;
            bool valid_k = k_offset < schedule.params->K;

            if (valid_k) {
              // Load matrix A with bounds checking
              if (CLUSTER_N > 1) {
                uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
                if (rank_n == 0) {
                  TMAOps::load_async_multicast(&sA[qidx * BK * BM], &tma_A,
                                               &full[qidx], k_offset,
                                               num_block_m * BM, mask);
                }
              } else {
                TMAOps::load_async(&sA[qidx * BK * BM], &tma_A, &full[qidx],
                                   k_offset, num_block_m * BM);
              }

              // Load matrix B with bounds checking
              if (CLUSTER_M > 1) {
                if (rank_m == 0) {
                  TMAOps::load_async_multicast(
                      &sB[qidx * BK * BN], &tma_B, &full[qidx], k_offset,
                      num_block_n * BN, col_mask << rank_n);
                }
              } else {
                TMAOps::load_async(&sB[qidx * BK * BN], &tma_B, &full[qidx],
                                   k_offset, num_block_n * BN);
              }
            }
          }
        }
      }
    }
  } else {
    // Consumer threads
    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];

// Initialize accumulator registers to zero
#pragma unroll
    for (int i = 0; i < B_WG_M / WGMMA_M; ++i) {
#pragma unroll
      for (int j = 0; j < WGMMA_N / 16; ++j) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          d[i][j][k] = 0.0f;
        }
      }
    }

    --wg_idx;

// Initialize empty flags
#pragma unroll
    for (int qidx = 0; qidx < QSIZE; ++qidx) {
      if (tid < CLUSTERS)
        PTXBarrier::arrive_cluster(&empty[qidx], tid);
    }

    int p = 0;
    int qidx = 0;
    int num_block_m, num_block_n;

    while (schedule.next(num_block_m, num_block_n)) {
      num_block_n = num_block_n * CLUSTER_N + rank_n;
      num_block_m = num_block_m * CLUSTER_M + rank_m;

      // Check matrix bounds
      bool is_valid_block = (num_block_m * BM < schedule.params->M) &&
                            (num_block_n * BN < schedule.params->N);

      if (is_valid_block) {
        // Process all K blocks
        for (int block_k_iter = 0; block_k_iter < num_blocks_k;
             ++block_k_iter, ++qidx) {
          if (qidx == QSIZE) {
            qidx = 0;
            p ^= 1;
          }
          PTXBarrier::wait(&full[qidx], p);
          WGMMASyncOps::warpgroup_arrive();

          int k_offset = block_k_iter * BK;
          bool valid_k = k_offset < schedule.params->K;

          if (valid_k) {
#pragma unroll
            for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
              bf16 *wgmma_sA =
                  sA + qidx * BK * BM +
                  64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
              bf16 *wgmma_sB = sB + qidx * BK * BN;

              // First K iteration
              wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[m_it], &wgmma_sA[0],
                                            &wgmma_sB[0]);

// Remaining K iterations
#pragma unroll
              for (int k_it = 1; k_it < BK / WGMMA_K; ++k_it) {
                wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it],
                                              &wgmma_sA[k_it * WGMMA_K],
                                              &wgmma_sB[k_it * WGMMA_K]);
              }
            }
          }

          WGMMASyncOps::warpgroup_commit_batch();
          WGMMASyncOps::warpgroup_wait<0>();
          if (tid < CLUSTERS)
            PTXBarrier::arrive_cluster(&empty[qidx], tid);
        }

        // Store results
        WGMMAGlobalStore::wait_previous();
        __syncthreads();

        // Store to shared memory
        int lane = tid % 32;
        int warp = tid / 32;
        int row = warp * 16 + lane / 4;
        bf16 *block_sC = sC + wg_idx * B_WG_M * BN;

#pragma unroll
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
          int yo = m_it * WGMMA_M;
#pragma unroll
          for (int w = 0; w < WGMMA_N; w += 16) {
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

        __syncthreads();

        // Store to global memory using TMA
        if (threadIdx.x == 128) {
          TMAOps::store_async(&tma_C, (bf16 *)&sC[0], num_block_m * BM,
                              num_block_n * BN);
          WGMMAGlobalStore::commit_group();
        }
      }
    }
  }
}

template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int CLUSTER_M = 2, int CLUSTER_N = 1, int NUM_SM = 128>
class GroupGemm {
private:
  std::vector<GemmParams> params;
  int group_size;

public:
  GroupGemm() : group_size(0) {}

  void initializeBatch(const std::vector<GemmParams> &batch_params) {
    params = batch_params;
    group_size = batch_params.size();
  }

  void launch() {
    if (group_size == 0)
      return;

    static_assert(NUM_SM % (CLUSTER_M * CLUSTER_N) == 0);
    constexpr int blocks_per_sm = NUM_SM / (CLUSTER_M * CLUSTER_N);

    dim3 grid(blocks_per_sm * group_size);
    dim3 block(NUM_THREADS);

    size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);

    cudaCheck(cudaFuncSetAttribute(
        groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                        CLUSTER_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaCheck(cudaDeviceSynchronize());

    // Launch kernel with parameters on device
    GemmParams *d_params;
    cudaCheck(cudaMalloc(&d_params, group_size * sizeof(GemmParams)));
    cudaCheck(cudaMemcpy(d_params, params.data(),
                         group_size * sizeof(GemmParams),
                         cudaMemcpyHostToDevice));

    groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                    CLUSTER_N>
        <<<grid, block, smem_size>>>(group_size, d_params);

    cudaCheck(cudaFree(d_params));
  }
};

} // namespace groupgemm
