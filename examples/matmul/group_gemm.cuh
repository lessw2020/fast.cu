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

// Debug print macro
#define DEBUG_PRINT(...)                                                       \
  if (threadIdx.x == 0 && blockIdx.x == 0) {                                   \
    printf(__VA_ARGS__);                                                       \
  }

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

// Per-GEMM parameters struct
struct GemmParams {
  int M, N, K;
  bf16 *A, *B, *C;
  CUtensorMap tmaA;
  CUtensorMap tmaB;
  CUtensorMap tmaC;
};

template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;

  __device__ __forceinline__ Schedule(int M, int N, int _block) {
    block = _block;
    it = 0;
    total_blocks_m = CEIL_DIV(M, BM);
    total_blocks_n = CEIL_DIV(N, BN);
    DEBUG_PRINT("Schedule initialized: M=%d, N=%d, total_blocks_m=%d, "
                "total_blocks_n=%d\n",
                M, N, total_blocks_m, total_blocks_n);
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    int num = it * NUM_SM + block;
    if (num >= total_blocks_m * total_blocks_n) {
      DEBUG_PRINT("Schedule complete: it=%d\n", it);
      return false;
    }

    int cur_tile = num / (TM * TN);
    int cur_tile_pos = num % (TM * TN);
    block_m = TM * (cur_tile / (total_blocks_n / TN));
    block_n = TN * (cur_tile % (total_blocks_n / TN));
    block_m += cur_tile_pos / TN;
    block_n += cur_tile_pos % TN;
    DEBUG_PRINT("Next block: m=%d, n=%d (it=%d)\n", block_m, block_n, it);
    ++it;
    return true;
  }
};

// Matmul Kernel
// ================
template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int NUM_SM = 128, int CLUSTER_M = 2, int CLUSTER_N = 1>
__global__
__launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M *CLUSTER_N, 1, 1)
    groupGemmKernel(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C,
                    const __grid_constant__ CUtensorMap tensorMapC,
                    const __grid_constant__ CUtensorMap tensorMapA,
                    const __grid_constant__ CUtensorMap tensorMapB) {
  constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  constexpr int B_WG_M = BM / num_consumers;
  constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;
  assert((M / BM) % CLUSTER_M == 0);
  assert((N / BN) % CLUSTER_N == 0);

  extern __shared__ __align__(128) uint8_t smem[];
  SharedMemoryLayout<BM, BN, BK, QSIZE> &s =
      *reinterpret_cast<SharedMemoryLayout<BM, BN, BK, QSIZE> *>(smem);

  bf16 *sA = s.A, *sB = s.B, *sC = s.C;
  uint64_t *full = s.full, *empty = s.empty;

  uint32_t cluster_id = ClusterOps::get_cluster_id();
  const int num_blocks_k = K / BK;
  int wg_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;

  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
      PTXBarrier::init_barrier(&full[i], 0, 1);
      PTXBarrier::init_barrier(&empty[i], 0, num_consumers * CLUSTERS);
    }
  }
  ClusterOps::cluster_sync();

  Schedule<1, NUM_SM / CLUSTERS, BM * CLUSTER_M, BN * CLUSTER_N, 16 / CLUSTER_M,
           8 / CLUSTER_N>
      schedule(M, N, cluster_id);

  uint32_t cluster_rank = ClusterOps::get_cluster_rank();
  uint32_t rank_m = cluster_rank / CLUSTER_N;
  uint32_t rank_n = cluster_rank % CLUSTER_N;

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
    constexpr int num_regs =
        (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
    RegisterManager::warpgroup_reg_alloc<num_regs>();
    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
    --wg_idx;

    for (int qidx = 0; qidx < QSIZE; ++qidx) {
      if (tid < CLUSTERS)
        PTXBarrier::arrive_cluster(&empty[qidx], tid);
    }

    // Setup output handler exactly as in matmul_10
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

// GroupGemm launcher
// ===================

template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int NUM_SM = 128, int CLUSTER_M = 2, int CLUSTER_N = 1>
class GroupGemm {
private:
  std::vector<GemmParams> params;
  std::vector<CUtensorMap> tmaAs, tmaBs, tmaCs;
  int blocks_per_group;

public:
  GroupGemm() {
    blocks_per_group = NUM_SM / (CLUSTER_M * CLUSTER_N);
    printf("GroupGemm initialized: blocks_per_group=%d\n", blocks_per_group);
  }
  void initializeBatch(const std::vector<int> &m_dims,
                       const std::vector<int> &n_dims,
                       const std::vector<int> &k_dims,
                       const std::vector<bf16 *> &as,
                       const std::vector<bf16 *> &bs,
                       const std::vector<bf16 *> &cs) {
    params.clear();

    const int group_size = m_dims.size();
    printf("Initializing batch with %d groups\n", group_size);
    printf("Configuration: BM=%d, BN=%d, BK=%d, CLUSTER_M=%d, CLUSTER_N=%d\n",
           BM, BN, BK, CLUSTER_M, CLUSTER_N);

    for (int i = 0; i < group_size; ++i) {
      printf("\nGroup %d:\n", i);
      printf("  Dimensions: M=%d, N=%d, K=%d\n", m_dims[i], n_dims[i],
             k_dims[i]);
      printf("  Memory: A=%p, B=%p, C=%p\n", (void *)as[i], (void *)bs[i],
             (void *)cs[i]);

      // Verify dimensions
      assert(m_dims[i] % (BM * CLUSTER_M) == 0);
      assert(n_dims[i] % (BN * CLUSTER_N) == 0);
      assert(k_dims[i] % BK == 0);

      GemmParams param;
      param.M = m_dims[i];
      param.N = n_dims[i];
      param.K = k_dims[i];
      param.A = as[i];
      param.B = bs[i];
      param.C = cs[i];

      // Create TMA descriptors
      printf("  Creating TMA descriptors...\n");
      param.tmaA = TensorMapManager::create_tensor_map<BM, BK>(as[i], m_dims[i],
                                                               k_dims[i]);
      cudaCheck(cudaDeviceSynchronize());

      param.tmaB = TensorMapManager::create_tensor_map<BN, BK>(bs[i], n_dims[i],
                                                               k_dims[i]);
      cudaCheck(cudaDeviceSynchronize());

      param.tmaC = TensorMapManager::create_tensor_map<BN, BM, false>(
          cs[i], n_dims[i], m_dims[i]);
      cudaCheck(cudaDeviceSynchronize());

      params.push_back(param);
      printf("  Group %d initialization complete\n", i);
    }
  }
  void launch() {
    if (params.empty())
      return;

    constexpr size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);
    static_assert(smem_size < 256 * 1024);

    printf("\nLaunching kernels:\n");
    printf("Shared memory size: %zu bytes\n", smem_size);
    printf("Grid configuration: blocks=%d, threads=%d\n", NUM_SM, NUM_THREADS);

    cudaCheck(cudaFuncSetAttribute(
        groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                        CLUSTER_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(NUM_SM);
    dim3 block(NUM_THREADS);

    for (size_t i = 0; i < params.size(); i++) {
      printf("\nLaunching group %zu:\n", i);
      printf("  Matrix: M=%d, N=%d, K=%d\n", params[i].M, params[i].N,
             params[i].K);
      printf("  Memory: A=%p, B=%p, C=%p\n", (void *)params[i].A,
             (void *)params[i].B, (void *)params[i].C);

      // Fixed kernel launch parameters to match kernel signature
      groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                      CLUSTER_N><<<grid, block, smem_size>>>(
          params[i].M, params[i].N, params[i].K, params[i].A, params[i].B,
          params[i].C, tmaCs[i], tmaAs[i], tmaBs[i]);

      cudaCheck(cudaDeviceSynchronize());
      printf("  Group %zu complete\n", i);
    }
    printf("\nAll kernels complete\n");
  }

  ~GroupGemm() {
    printf("Cleaning up GroupGemm\n");
    params.clear();
    tmaAs.clear();
    tmaBs.clear();
    tmaCs.clear();
  }
};

} // namespace groupgemm

/// mods ////////

// Previous struct and error macros remain the same...

// Rest of GroupGemm class remains the same...

// namespace groupgemm
