#pragma once
#include "wgmax.cuh"
#include <vector>

namespace groupgemm {
using namespace wgmma_utils;

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

struct GemmParams {
  int M, N, K;
  int total_blocks_m; // Added for scheduling
  int total_blocks_n; // Added for scheduling
  bf16 *A;
  bf16 *B;
  bf16 *C;
};

// TMA descriptors for each GEMM
struct GemmDescriptors {
  CUtensorMap tma_A;
  CUtensorMap tma_B;
  CUtensorMap tma_C;
};

//// /////
template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;

  __device__ __forceinline__ Schedule(const GemmParams &params, int _block) {
    block = _block;
    it = 0;
    total_blocks_m = CEIL_DIV(params.M, BM);
    total_blocks_n = CEIL_DIV(params.N, BN);
    assert(CEIL_DIV(params.M, BM) % TM == 0 && total_blocks_n % TN == 0);
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    int num = it * NUM_SM + block;
    if (num >= total_blocks_m * total_blocks_n) {
      return false;
    }

    int cur_tile = num / (TM * TN);
    int cur_tile_pos = num % (TM * TN);
    block_m = TM * (cur_tile / (total_blocks_n / TN));
    block_n = TN * (cur_tile % (total_blocks_n / TN));
    block_m += cur_tile_pos / TN;
    block_n += cur_tile_pos % TN;
    ++it;
    return true;
  }
};

template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int NUM_SM = 128, int CLUSTER_M = 2, int CLUSTER_N = 1>
__global__
__launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M *CLUSTER_N, 1, 1)
    groupGemmKernel(const __grid_constant__ GemmParams params,
                    const __grid_constant__ CUtensorMap tensorMapC,
                    const __grid_constant__ CUtensorMap tensorMapA,
                    const __grid_constant__ CUtensorMap tensorMapB) {
  constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  constexpr int B_WG_M = BM / num_consumers;
  constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

  // Verify dimensions are compatible with clustering
  assert((params.M / BM) % CLUSTER_M == 0);
  assert((params.N / BN) % CLUSTER_N == 0);

  extern __shared__ __align__(128) uint8_t smem[];
  SharedMemoryLayout<BM, BN, BK, QSIZE> &s =
      *reinterpret_cast<SharedMemoryLayout<BM, BN, BK, QSIZE> *>(smem);

  bf16 *sA = s.A, *sB = s.B, *sC = s.C;
  uint64_t *full = s.full, *empty = s.empty;

  uint32_t cluster_id = ClusterOps::get_cluster_id();
  const int num_blocks_k = params.K / BK;
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

  Schedule<1, NUM_SM / CLUSTERS, BM * CLUSTER_M, BN * CLUSTER_N, 16 / CLUSTER_M,
           8 / CLUSTER_N>
      schedule(params, cluster_id);

  uint32_t cluster_rank = ClusterOps::get_cluster_rank();
  uint32_t rank_m = cluster_rank / CLUSTER_N;
  uint32_t rank_n = cluster_rank % CLUSTER_N;

  // Producer thread
  if (wg_idx == 0) {
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
      while (schedule.next(num_block_m, num_block_n)) {
        num_block_n = num_block_n * CLUSTER_N + rank_n;
        num_block_m = num_block_m * CLUSTER_M + rank_m;

        for (int block_k_iter = 0; block_k_iter < num_blocks_k;
             ++block_k_iter, ++qidx) {
          if (qidx == QSIZE) {
            qidx = 0;
            p ^= 1;
          }

          PTXBarrier::wait(&empty[qidx], p);
          PTXBarrier::expect_bytes_tx(&full[qidx],
                                      (BK * BN + BK * BM) * sizeof(bf16));

          // Handle loads for A matrix
          if (CLUSTER_N > 1) {
            uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
            if (rank_n == 0) {
              TMAOps::load_async_multicast(&sA[qidx * BK * BM], &tensorMapA,
                                           &full[qidx], block_k_iter * BK,
                                           num_block_m * BM, mask);
            }
          } else {
            TMAOps::load_async(&sA[qidx * BK * BM], &tensorMapA, &full[qidx],
                               block_k_iter * BK, num_block_m * BM);
          }

          // Handle loads for B matrix
          if (CLUSTER_M > 1) {
            if (rank_m == 0) {
              TMAOps::load_async_multicast(
                  &sB[qidx * BK * BN], &tensorMapB, &full[qidx],
                  block_k_iter * BK, num_block_n * BN, col_mask << rank_n);
            }
          } else {
            TMAOps::load_async(&sB[qidx * BK * BN], &tensorMapB, &full[qidx],
                               block_k_iter * BK, num_block_n * BN);
          }
        }
      }
    }
  } else {
    // Consumer threads
    constexpr int num_regs =
        (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
    RegisterManager::warpgroup_reg_alloc<num_regs>();

    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
    --wg_idx;

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
    while (schedule.next(num_block_m, num_block_n)) {
      num_block_n = num_block_n * CLUSTER_N + rank_n;
      num_block_m = num_block_m * CLUSTER_M + rank_m;
      {
        if (qidx == QSIZE) {
          qidx = 0;
          p ^= 1;
        }

        PTXBarrier::wait(&full[qidx], p);
        WGMMASyncOps::warpgroup_arrive();

#pragma unroll
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
          bf16 *wgmma_sA = sA + qidx * BK * BM +
                           64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
          bf16 *wgmma_sB = sB + qidx * BK * BN;
          {
            // Initial WGMMA compute
            wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[m_it], &wgmma_sA[0], &wgmma_sB[0]);

#pragma unroll
            for (int k_it = 1; k_it < 64 / WGMMA_K; ++k_it) {
              wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
                                            &wgmma_sB[k_it * WGMMA_K]);
            }
            wgmma_sA += 64 * BM;
            wgmma_sB += 64 * BN;
          }

#pragma unroll
          for (int bk = 64; bk < BK; bk += 64) {
#pragma unroll
            for (int k_it = 0; k_it < 64 / WGMMA_K; ++k_it) {
              wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
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
        ++qidx;
      }

      for (int block_k_iter = 1; block_k_iter < num_blocks_k;
           ++block_k_iter, ++qidx) {
        if (qidx == QSIZE) {
          qidx = 0;
          p ^= 1;
        }

        PTXBarrier::wait(&full[qidx], p);
        WGMMASyncOps::warpgroup_arrive();

#pragma unroll
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
          bf16 *wgmma_sA = sA + qidx * BK * BM +
                           64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
          bf16 *wgmma_sB = sB + qidx * BK * BN;

#pragma unroll
          for (int bk = 0; bk < BK; bk += 64) {
#pragma unroll
            for (int k_it = 0; k_it < 64 / WGMMA_K; ++k_it) {
              wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
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
        TMAOps::store_async(&tensorMapC, (bf16 *)&sC[0], num_block_m * BM,
                            num_block_n * BN);
        WGMMAGlobalStore::commit_group();
      }
    }
  }
}

template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int CLUSTER_M = 2, int CLUSTER_N = 1, int NUM_SM = 128>
class GroupGemm {
private:
  std::vector<GemmParams> params;
  std::vector<CUtensorMap> tmaAs, tmaBs, tmaCs;
  int blocks_per_group;

public:
  GroupGemm() { blocks_per_group = NUM_SM / (CLUSTER_M * CLUSTER_N); }

  void initializeBatch(const std::vector<int> &m_dims,
                       const std::vector<int> &n_dims,
                       const std::vector<int> &k_dims,
                       const std::vector<bf16 *> &as,
                       const std::vector<bf16 *> &bs,
                       const std::vector<bf16 *> &cs) {
    params.clear();
    tmaAs.clear();
    tmaBs.clear();
    tmaCs.clear();

    const int group_size = m_dims.size();
    printf("Initializing batch with %d groups\n", group_size);

    for (int i = 0; i < group_size; ++i) {
      printf("Creating descriptors for group %d: M=%d, N=%d, K=%d\n", i,
             m_dims[i], n_dims[i], k_dims[i]);

      // Calculate total blocks for this GEMM
      int total_blocks_m = CEIL_DIV(m_dims[i], BM);
      int total_blocks_n = CEIL_DIV(n_dims[i], BN);

      // Store parameters with block counts
      params.push_back({m_dims[i], n_dims[i], k_dims[i], total_blocks_m,
                        total_blocks_n, as[i], bs[i], cs[i]});

      // Create TMA descriptors
      tmaAs.push_back(TensorMapManager::create_tensor_map<BM, BK>(
          as[i], m_dims[i], k_dims[i]));
      tmaBs.push_back(TensorMapManager::create_tensor_map<BN, BK>(
          bs[i], n_dims[i], k_dims[i]));
      tmaCs.push_back(TensorMapManager::create_tensor_map<BN, BM, false>(
          cs[i], n_dims[i], m_dims[i]));

      cudaCheck(cudaDeviceSynchronize());
    }
  }

  void launch() {
    if (params.empty())
      return;

    constexpr size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);
    static_assert(smem_size < 256 * 1024);

    cudaCheck(cudaFuncSetAttribute(
        groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                        CLUSTER_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(NUM_SM);
    dim3 block(NUM_THREADS);

    // Launch kernel for each group
    for (size_t i = 0; i < params.size(); i++) {
      printf("Launching group %d: M=%d, N=%d, K=%d (blocks: %d x %d)\n", i,
             params[i].M, params[i].N, params[i].K, params[i].total_blocks_m,
             params[i].total_blocks_n);

      groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                      CLUSTER_N>
          <<<grid, block, smem_size>>>(params[i], tmaCs[i], tmaAs[i], tmaBs[i]);

      cudaCheck(cudaDeviceSynchronize());
    }
  }

  ~GroupGemm() {
    params.clear();
    tmaAs.clear();
    tmaBs.clear();
    tmaCs.clear();
  }
};

} // namespace groupgemm

////////////////////

// Per-GEMM parameters struct updated to include dimensions for the schedule

// namespace groupgemm
