#pragma once
#include "wgmini.cuh"

namespace M8 {

using namespace wgmma_utils;

// Matmul configuration
template <int BM = 128, int BN = 256, int BK = 64>
struct Matmul8Config : public KernelConfig<bf16, BM, BN, BK> {
  static constexpr int NumThreads = 128 * 3; // 3 warpgroups
  static constexpr int QueueSize = 3;        // Triple buffering
  static constexpr int ClusterM = 2;
  static constexpr int ClusterN = 1;
  static constexpr int NumSMs = 128;

  static constexpr int WarpgroupM = 64;
  static constexpr int WarpgroupK = 16;

  static_assert(NumSMs % (ClusterM * ClusterN) == 0,
                "Number of SMs must be divisible by cluster size");
};

// Main matmul kernel
template <typename Config = Matmul8Config<>>
__global__ __launch_bounds__(Config::NumThreads)
    __cluster_dims__(Config::ClusterM, Config::ClusterN, 1) void matmul8Kernel(
        int M, int N, int K, bf16 *C,
        const __grid_constant__ CUtensorMap tensorMapA,
        const __grid_constant__ CUtensorMap tensorMapB) {

  // Calculate constants
  constexpr int num_consumers = (Config::NumThreads / 128) - 1;
  constexpr int B_WG_M = Config::BlockM / num_consumers;
  constexpr int CLUSTERS = Config::ClusterM * Config::ClusterN;

  // Validate input dimensions
  assert((M / Config::BlockM) % Config::ClusterM == 0);
  assert((N / Config::BlockN) % Config::ClusterN == 0);

  // Setup shared memory
  extern __shared__ uint8_t shared_memory[];
  using SharedMemType = typename SharedMemoryLayout<
      Config>::template QueuedBuffer<Config::QueueSize>;
  auto &smem = *reinterpret_cast<SharedMemType *>(shared_memory);

  // Setup barriers
  __shared__ uint64_t full[Config::QueueSize];
  __shared__ uint64_t empty[Config::QueueSize];

  // Get cluster information
  auto cluster_info = ClusterInfo::get();
  ClusterInfo::syncCluster();

  const int num_blocks_k = K / Config::BlockK;
  int wg_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;

  // Producer thread
  if (wg_idx == 0) {
    RegisterManager::dealloc<24>();

    if (tid == 0) {
      // Initialize barriers
      for (int i = 0; i < Config::QueueSize; ++i) {
        PTXBarrier::init_barrier(&full[i], 0, 1);
        PTXBarrier::init_barrier(&empty[i], 0, num_consumers * CLUSTERS);
      }

      // Setup scheduling
      Schedule<1, Config::NumSMs / CLUSTERS, Config::BlockM * Config::ClusterM,
               Config::BlockN * Config::ClusterN, 16 / Config::ClusterM,
               8 / Config::ClusterN>
          schedule(M, N, cluster_info.cluster_id);

      int p = 0;
      int qidx = 0;
      uint32_t col_mask = 0;
      for (int i = 0; i < Config::ClusterM; ++i) {
        col_mask |= (1 << (i * Config::ClusterN));
      }

      int block_m, block_n;
      while (schedule.next(block_m, block_n)) {
        block_n = block_n * Config::ClusterN + cluster_info.rank_n;
        block_m = block_m * Config::ClusterM + cluster_info.rank_m;

        for (int k = 0; k < num_blocks_k; ++k, ++qidx) {
          if (qidx == Config::QueueSize) {
            qidx = 0;
            p ^= 1;
          }

          PTXBarrier::wait(&empty[qidx], p);
          PTXBarrier::expect_tx(&full[qidx], (Config::BlockK * Config::BlockN +
                                              Config::BlockK * Config::BlockM) *
                                                 sizeof(bf16));

          if constexpr (Config::ClusterN > 1) {
            uint32_t mask = ((1 << Config::ClusterN) - 1)
                            << (cluster_info.rank_m * Config::ClusterN);
            if (cluster_info.rank_n == 0) {
              TMAOps::load_async_multicast(
                  &smem.A[qidx * Config::BlockK * Config::BlockM], &tensorMapA,
                  &full[qidx], k * Config::BlockK, block_m * Config::BlockM,
                  mask);
            }
          } else {
            TMAOps::load_async(&smem.A[qidx * Config::BlockK * Config::BlockM],
                               &tensorMapA, &full[qidx], k * Config::BlockK,
                               block_m * Config::BlockM);
          }

          if constexpr (Config::ClusterM > 1) {
            if (cluster_info.rank_m == 0) {
              TMAOps::load_async_multicast(
                  &smem.B[qidx * Config::BlockK * Config::BlockN], &tensorMapB,
                  &full[qidx], k * Config::BlockK, block_n * Config::BlockN,
                  col_mask << cluster_info.rank_n);
            }
          } else {
            TMAOps::load_async(&smem.B[qidx * Config::BlockK * Config::BlockN],
                               &tensorMapB, &full[qidx], k * Config::BlockK,
                               block_n * Config::BlockN);
          }
        }
      }
    }
  }
  // Consumer threads
  else {
    RegisterManager::alloc<160>();
    float d[B_WG_M / Config::WarpgroupM][Config::BlockN / 16][8];
    --wg_idx;

    // Initialize empty barriers
    for (int qidx = 0; qidx < Config::QueueSize; ++qidx) {
      if (tid < CLUSTERS) {
        PTXBarrier::arrive_cluster(&empty[qidx], tid);
      }
    }

    int p = 0;
    int qidx = 0;
    int lane = tid % 32;
    int warp = tid / 32;
    int row = warp * 16 + lane / 4;

    Schedule<1, Config::NumSMs / CLUSTERS, Config::BlockM * Config::ClusterM,
             Config::BlockN * Config::ClusterN, 16 / Config::ClusterM,
             8 / Config::ClusterN>
        schedule(M, N, cluster_info.cluster_id);

    int block_m, block_n;
    while (schedule.next(block_m, block_n)) {
      block_n = block_n * Config::ClusterN + cluster_info.rank_n;
      block_m = block_m * Config::ClusterM + cluster_info.rank_m;

      // Zero initialize accumulator
      memset(d, 0, sizeof(d));

      for (int k = 0; k < num_blocks_k; ++k, ++qidx) {
        if (qidx == Config::QueueSize) {
          qidx = 0;
          p ^= 1;
        }

        PTXBarrier::wait(&full[qidx], p);
        WGMMASyncOps::arrive();

        for (int m_it = 0; m_it < B_WG_M / Config::WarpgroupM; ++m_it) {
          bf16 *wgmma_sA =
              &smem.A[qidx * Config::BlockK * Config::BlockM +
                      64 * (m_it + wg_idx * B_WG_M / Config::WarpgroupM) *
                          Config::WarpgroupM];
          bf16 *wgmma_sB = &smem.B[qidx * Config::BlockK * Config::BlockN];

          for (int bk = 0; bk < Config::BlockK; bk += 64) {
            for (int k_it = 0; k_it < 64 / Config::WarpgroupK; ++k_it) {
              wgmma_dispatch<Config::BlockN>(
                  &d[m_it][0][0], &wgmma_sA[k_it * Config::WarpgroupK],
                  &wgmma_sB[k_it * Config::WarpgroupK]);
            }
            wgmma_sA += 64 * Config::BlockM;
            wgmma_sB += 64 * Config::BlockN;
          }
        }

        WGMMASyncOps::commit_group();
        WGMMASyncOps::wait_group<0>();

        if (tid < CLUSTERS) {
          PTXBarrier::arrive_cluster(&empty[qidx], tid);
        }
      }

      // Store results
      bf16 *block_C =
          C + block_n * Config::BlockN * M + block_m * Config::BlockM;

      for (int m_it = 0; m_it < B_WG_M / Config::WarpgroupM; ++m_it) {
        int yo = m_it * Config::WarpgroupM + wg_idx * B_WG_M;
        if (row + 8 + yo + block_m * Config::BlockM >= M)
          continue;

        for (int w = 0; w < Config::BlockN; w += 16) {
          if (w + block_n * Config::BlockN < N) {
            int col = w + 2 * (tid % 4);

#define IDX(i, j) ((j) * M + ((i) + yo))
#define ST(i, j, v) block_C[IDX(i, j)] = v;

            ST(row, col, d[m_it][w / 16][0]);
            ST(row, col + 1, d[m_it][w / 16][1]);
            ST(row + 8, col, d[m_it][w / 16][2]);
            ST(row + 8, col + 1, d[m_it][w / 16][3]);
            ST(row, col + 8, d[m_it][w / 16][4]);
            ST(row, col + 9, d[m_it][w / 16][5]);
            ST(row + 8, col + 8, d[m_it][w / 16][6]);
            ST(row + 8, col + 9, d[m_it][w / 16][7]);

#undef IDX
#undef ST
          }
        }
      }
    }
  }
}

// Host-side launch function
template <typename Config = Matmul8Config<>>
void runMatmul8(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  // Initialize TMA descriptors
  TMACache<Config>::initializeMaps(M, N, K, A, B);

  // Calculate shared memory size
  size_t smem_size = sizeof(typename SharedMemoryLayout<
                            Config>::template QueuedBuffer<Config::QueueSize>);

  // Set maximum shared memory size
  cudaFuncSetAttribute(matmul8Kernel<Config>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  // Launch kernel
  matmul8Kernel<Config><<<Config::NumSMs, Config::NumThreads, smem_size>>>(
      M, N, K, C, TMACache<Config>::getMapA(), TMACache<Config>::getMapB());
}

} // namespace M8

using M8::runMatmul8;
