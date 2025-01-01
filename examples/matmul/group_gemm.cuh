#pragma once
#include "wgmax.cuh"
#include <vector>

// Error checking helper
#define cudaCheck(err)                                                         \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      printf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,         \
             cudaGetErrorString(err_));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace groupgemm {
using namespace wgmma_utils;

// Per-GEMM parameters
struct GemmParams {
  int M, N, K;
  bf16 *A;
  bf16 *B;
  bf16 *C;
  float alpha;
  float beta;
};

// TMA descriptors for each GEMM in the group
struct GemmDescriptors {
  CUtensorMap tma_A;
  CUtensorMap tma_B;
  CUtensorMap tma_C;
};

// Forward declare kernel since it can't be a member function
template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int CLUSTER_M,
          int CLUSTER_N, int NUM_SM>
__global__ void groupGemmKernel(int group_size, const GemmParams *params,
                                const GemmDescriptors *descs);

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

template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int CLUSTER_M = 2, int CLUSTER_N = 1, int NUM_SM = 128>
class GroupGemm {
private:
  std::vector<GemmDescriptors> descriptors;
  std::vector<GemmParams> params;
  int group_size;

  // Create TMA descriptors for a single GEMM
  GemmDescriptors createDescriptors(const GemmParams &param) {
    GemmDescriptors desc;
    desc.tma_A =
        TensorMapManager::create_tensor_map<BM, BK>(param.A, param.M, param.K);
    desc.tma_B =
        TensorMapManager::create_tensor_map<BN, BK>(param.B, param.N, param.K);
    desc.tma_C = TensorMapManager::create_tensor_map<BN, BM, false>(
        param.C, param.N, param.M);
    return desc;
  }

public:
  GroupGemm() : group_size(0) {}
  ~GroupGemm() { cleanup(); }

  void cleanup() {
    descriptors.clear();
    params.clear();
    group_size = 0;
  }

  void initializeBatch(const std::vector<GemmParams> &batch_params) {
    cleanup();
    group_size = batch_params.size();
    params = batch_params;
    descriptors.reserve(group_size);
    for (const auto &param : batch_params) {
      descriptors.push_back(createDescriptors(param));
    }
  }

  void launch() {
    static_assert(NUM_SM % (CLUSTER_M * CLUSTER_N) == 0);

    // Calculate grid size ensuring proper alignment
    int blocks_per_group = NUM_SM / (CLUSTER_M * CLUSTER_N);
    dim3 grid(blocks_per_group * group_size);
    dim3 block(NUM_THREADS);

    size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);

    cudaCheck(cudaFuncSetAttribute(
        groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, CLUSTER_M, CLUSTER_N,
                        NUM_SM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Ensure CUDA device is properly synchronized
    cudaCheck(cudaDeviceSynchronize());

    groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, CLUSTER_M, CLUSTER_N,
                    NUM_SM><<<grid, block, smem_size>>>(
        group_size, params.data(), descriptors.data());
  }
};

// Forward declare kernel
template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int CLUSTER_M,
          int CLUSTER_N, int NUM_SM>
__global__ void groupGemmKernel(int group_size, const GemmParams *params,
                                const GemmDescriptors *descs);

// Kernel implementation
template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int CLUSTER_M,
          int CLUSTER_N, int NUM_SM>
__global__ void groupGemmKernel(int group_size, const GemmParams *params,
                                const GemmDescriptors *descs) {
  constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  constexpr int B_WG_M = BM / num_consumers;
  constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

  // Calculate group index and validate
  int group_idx = blockIdx.x / (NUM_SM / group_size);
  if (group_idx >= group_size)
    return;

  // Get parameters for this group member
  const auto &param = params[group_idx];
  const auto &desc = descs[group_idx];

  // Adjust block index for this group member
  int adjusted_block_idx = blockIdx.x % (NUM_SM / group_size);

  // Shared memory setup
  extern __shared__ uint8_t shared_mem[];
  auto &smem =
      *reinterpret_cast<SharedMemoryLayout<BM, BN, BK, QSIZE> *>(shared_mem);

  bf16 *sA = smem.A;
  bf16 *sB = smem.B;
  bf16 *sC = smem.C;
  uint64_t *full = smem.full;
  uint64_t *empty = smem.empty;

  uint32_t cluster_id = ClusterOps::get_cluster_id();
  const int num_blocks_k = CEIL_DIV(param.K, BK);
  int wg_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;

  // Initialize barriers
  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
      PTXBarrier::init_barrier(&full[i], 0, 1);
      PTXBarrier::init_barrier(&empty[i], 0, num_consumers * CLUSTERS);
    }
  }

  ClusterOps::cluster_sync();

  // Ensure warpgroup sync before register allocation
  WGMMASyncOps::warpgroup_arrive();

  // Create block schedule for this group
  BlockScheduler scheduler(param.M, param.N, BM, BN, 16 / CLUSTER_M,
                           8 / CLUSTER_N, NUM_SM / (group_size * CLUSTERS),
                           adjusted_block_idx);

  uint32_t cluster_rank = ClusterOps::get_cluster_rank();
  uint32_t rank_m = cluster_rank / CLUSTER_N;
  uint32_t rank_n = cluster_rank % CLUSTER_N;

  // Producer thread
  if (wg_idx == 0) {
    WGMMASyncOps::warpgroup_sync();
    constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
    RegisterManager::warpgroup_reg_dealloc<num_regs>();

    if (tid == 0) {
      int p = 0;
      int qidx = 0;
      uint32_t col_mask = 0;
      for (int i = 0; i < CLUSTER_M; ++i) {
        col_mask |= (1 << (i * CLUSTER_N));
      }

      int num_block_m, num_block_n;
      while (scheduler.next(num_block_m, num_block_n)) {
        num_block_n = num_block_n * CLUSTER_N + rank_n;
        num_block_m = num_block_m * CLUSTER_M + rank_m;

        // Validate block coordinates
        bool is_valid =
            (num_block_m * BM < param.M) && (num_block_n * BN < param.N);

        if (is_valid) {
          for (int block_k_iter = 0; block_k_iter < num_blocks_k;
               ++block_k_iter, ++qidx) {
            if (qidx == QSIZE) {
              qidx = 0;
              p ^= 1;
            }

            PTXBarrier::wait(&empty[qidx], p);
            PTXBarrier::expect_bytes_tx(&full[qidx],
                                        (BK * BN + BK * BM) * sizeof(bf16));

            // Load matrix A
            if (CLUSTER_N > 1) {
              uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
              if (rank_n == 0 && block_k_iter * BK < param.K) {
                TMAOps::load_async_multicast(&sA[qidx * BK * BM], &desc.tma_A,
                                             &full[qidx], block_k_iter * BK,
                                             num_block_m * BM, mask);
              }
            } else if (block_k_iter * BK < param.K) {
              TMAOps::load_async(&sA[qidx * BK * BM], &desc.tma_A, &full[qidx],
                                 block_k_iter * BK, num_block_m * BM);
            }

            // Load matrix B
            if (CLUSTER_M > 1) {
              if (rank_m == 0 && block_k_iter * BK < param.K) {
                TMAOps::load_async_multicast(
                    &sB[qidx * BK * BN], &desc.tma_B, &full[qidx],
                    block_k_iter * BK, num_block_n * BN, col_mask << rank_n);
              }
            } else if (block_k_iter * BK < param.K) {
              TMAOps::load_async(&sB[qidx * BK * BN], &desc.tma_B, &full[qidx],
                                 block_k_iter * BK, num_block_n * BN);
            }
          }
        }
      }
    }
  } else {
    // Consumer threads
    WGMMASyncOps::warpgroup_sync();
    constexpr int num_regs =
        (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
    RegisterManager::warpgroup_reg_alloc<num_regs>();

    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
    --wg_idx;

    // Initialize empty flags
    for (int qidx = 0; qidx < QSIZE; ++qidx) {
      if (tid < CLUSTERS)
        PTXBarrier::arrive_cluster(&empty[qidx], tid);
    }

    // Setup output handler
    WGMMAOutputHandler<bf16, B_WG_M, WGMMA_M, WGMMA_N> output_handler(
        sC, threadIdx.x, wg_idx);

    int p = 0;
    int qidx = 0;
    int num_block_m, num_block_n;

    while (scheduler.next(num_block_m, num_block_n)) {
      num_block_n = num_block_n * CLUSTER_N + rank_n;
      num_block_m = num_block_m * CLUSTER_M + rank_m;

      // Validate block coordinates
      bool is_valid =
          (num_block_m * BM < param.M) && (num_block_n * BN < param.N);

      if (is_valid) {
        // First block
        {
          if (qidx == QSIZE) {
            qidx = 0;
            p ^= 1;
          }

          PTXBarrier::wait(&full[qidx], p);
          WGMMASyncOps::warpgroup_arrive();

// Process sub-blocks
#pragma unroll
          for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
            bf16 *wgmma_sA = sA + qidx * BK * BM +
                             64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
            bf16 *wgmma_sB = sB + qidx * BK * BN;

            // Initial compute
            wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[m_it], &wgmma_sA[0], &wgmma_sB[0]);

            // Process remaining iterations within 64 elem boundary

            // Process remaining iterations within 64 elem boundary
#pragma unroll
            for (int k_it = 1; k_it < 64 / WGMMA_K; ++k_it) {
              wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
                                            &wgmma_sB[k_it * WGMMA_K]);
            }

            wgmma_sA += 64 * BM;
            wgmma_sB += 64 * BN;

// Process remaining blocks
#pragma unroll
            for (int bk = 64; bk < BK; bk += 64) {
#pragma unroll
              for (int k_it = 0; k_it < 64 / WGMMA_K; ++k_it) {
                wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it],
                                              &wgmma_sA[k_it * WGMMA_K],
                                              &wgmma_sB[k_it * WGMMA_K]);
              }
              wgmma_sA += 64 * BM;
              wgmma_sB += 64 * BN;
            }
          }

          WGMMASyncOps::warpgroup_commit_batch();
          WGMMASyncOps::warpgroup_wait<0>();

          if (tid < CLUSTERS)
            PTXBarrier::arrive_cluster(&empty[qidx], tid);
        }

        // Output storage
        WGMMAGlobalStore::wait_previous();

        int lane = tid % 32, warp = tid / 32;
        int row = warp * 16 + lane / 4;
        bf16 *block_sC = sC + wg_idx * B_WG_M * BN;

#pragma unroll
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
          int yo = m_it * WGMMA_M;
#pragma unroll
          for (int w = 0; w < WGMMA_N; w += 16) {
            int col = w + 2 * (tid % 4);

// Store results using macro for cleaner code
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
          TMAOps::store_async(&desc.tma_C, (bf16 *)&sC[0], num_block_m * BM,
                              num_block_n * BN);
          WGMMAGlobalStore::commit_group();
        }
      }
    }
  }
}

} // namespace groupgemm
