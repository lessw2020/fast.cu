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
};

// TMA descriptors for each GEMM
struct GemmDescriptors {
  CUtensorMap tma_A;
  CUtensorMap tma_B;
  CUtensorMap tma_C;
};

////
template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;
  int group_idx;
  bool valid;

  __device__ __forceinline__ Schedule(int block_idx, int group_size) {
    constexpr int blocks_per_sm = NUM_SM / (TM * TN);
    block = block_idx % blocks_per_sm;
    group_idx = block_idx / blocks_per_sm;
    valid = (group_idx < group_size);

    it = 0;
    if (valid) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Schedule init: block_idx=%d, group_size=%d, group_idx=%d\n",
               block_idx, group_size, group_idx);
        printf("Blocks per SM=%d\n", blocks_per_sm);
      }
    }
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    if (!valid)
      return false;

    int num = it * block;
    if (num >= total_blocks_m * total_blocks_n)
      return false;

    int tiles_m = CEIL_DIV(total_blocks_m, TM);
    int tiles_n = CEIL_DIV(total_blocks_n, TN);

    int cur_tile = num / (TM * TN);
    int cur_tile_pos = num % (TM * TN);

    block_m = TM * (cur_tile / tiles_n);
    block_n = TN * (cur_tile % tiles_n);
    block_m += cur_tile_pos / TN;
    block_n += cur_tile_pos % TN;

    if (threadIdx.x == 0 && blockIdx.x == 0 && it == 0) {
      printf("First iteration: num=%d, block_m=%d, block_n=%d\n", num, block_m,
             block_n);
      printf("Tiles: M=%d, N=%d\n", tiles_m, tiles_n);
    }

    ++it;
    return (block_m < total_blocks_m) && (block_n < total_blocks_n);
  }

  __device__ __forceinline__ bool is_valid() const { return valid; }

  __device__ __forceinline__ void set_dimensions(int M, int N) {
    total_blocks_m = CEIL_DIV(M, BM);
    total_blocks_n = CEIL_DIV(N, BN);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("Set dimensions: M=%d, N=%d\n", M, N);
      printf("Total blocks: M=%d, N=%d\n", total_blocks_m, total_blocks_n);
    }
  }
};

template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int NUM_SM = 128, int CLUSTER_M = 2, int CLUSTER_N = 1>
__global__
__launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M *CLUSTER_N, 1, 1)
    groupGemmKernel(int group_size, int M, int N, int K,
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
      schedule(blockIdx.x, group_size);

  if (!schedule.is_valid())
    return;

  schedule.set_dimensions(M, N);

  uint32_t cluster_rank = ClusterOps::get_cluster_rank();
  uint32_t rank_m = cluster_rank / CLUSTER_N;
  uint32_t rank_n = cluster_rank % CLUSTER_N;

  // Producer thread
  if (wg_idx == 0) {
    WGMMASyncOps::warpgroup_arrive();
    if (threadIdx.x < 128) {
      constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
      RegisterManager::warpgroup_reg_dealloc<num_regs>();
    }

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

          int k_offset = block_k_iter * BK;
          if (k_offset < K) {
            if (CLUSTER_N > 1) {
              uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
              if (rank_n == 0) {
                TMAOps::load_async_multicast(&sA[qidx * BK * BM], &tensorMapA,
                                             &full[qidx], k_offset,
                                             num_block_m * BM, mask);
              }
            } else {
              TMAOps::load_async(&sA[qidx * BK * BM], &tensorMapA, &full[qidx],
                                 k_offset, num_block_m * BM);
            }

            if (CLUSTER_M > 1) {
              if (rank_m == 0) {
                TMAOps::load_async_multicast(
                    &sB[qidx * BK * BN], &tensorMapB, &full[qidx], k_offset,
                    num_block_n * BN, col_mask << rank_n);
              }
            } else {
              TMAOps::load_async(&sB[qidx * BK * BN], &tensorMapB, &full[qidx],
                                 k_offset, num_block_n * BN);
            }
          }
        }
      }
    }
  } else {
    // Consumer threads
    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8] = {};
    --wg_idx;

    WGMMASyncOps::warpgroup_arrive();
    if (threadIdx.x >= 128) {
      constexpr int num_regs =
          (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
      RegisterManager::warpgroup_reg_alloc<num_regs>();
    }

    // Initialize empty flags
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

      WGMMAOutputHandler<bf16, B_WG_M, WGMMA_M, WGMMA_N> output_handler(
          sC, threadIdx.x, wg_idx);

#pragma unroll
      for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
        output_handler.store_output(d, m_it);
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
  std::vector<int> Ms, Ns, Ks;
  std::vector<bf16 *> As, Bs, Cs;
  std::vector<CUtensorMap> tmaAs, tmaBs, tmaCs;
  int group_size;

  CUtensorMap createTensorMap(bf16 *data, int m, int n, bool isA) {
    if (isA) {
      return TensorMapManager::create_tensor_map<BM, BK>(data, m, n);
    } else {
      return TensorMapManager::create_tensor_map<BN, BK>(data, m, n);
    }
  }

public:
  GroupGemm() : group_size(0) {}

  void initializeBatch(const std::vector<int> &m_dims,
                       const std::vector<int> &n_dims,
                       const std::vector<int> &k_dims,
                       const std::vector<bf16 *> &as,
                       const std::vector<bf16 *> &bs,
                       const std::vector<bf16 *> &cs) {
    Ms = m_dims;
    Ns = n_dims;
    Ks = k_dims;
    As = as;
    Bs = bs;
    Cs = cs;
    group_size = Ms.size();

    // Create TMA descriptors
    tmaAs.clear();
    tmaBs.clear();
    tmaCs.clear();

    tmaAs.reserve(group_size);
    tmaBs.reserve(group_size);
    tmaCs.reserve(group_size);

    for (int i = 0; i < group_size; ++i) {
      tmaAs.push_back(createTensorMap(As[i], Ms[i], Ks[i], true));
      tmaBs.push_back(createTensorMap(Bs[i], Ns[i], Ks[i], false));
      tmaCs.push_back(TensorMapManager::create_tensor_map<BN, BM, false>(
          Cs[i], Ns[i], Ms[i]));
    }
    cudaCheck(cudaDeviceSynchronize());
  }

  void launch() {
    if (group_size == 0)
      return;

    cudaCheck(cudaDeviceSynchronize());

    dim3 grid(NUM_SM / (CLUSTER_M * CLUSTER_N) * group_size);
    dim3 block(NUM_THREADS);

    size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);

    printf("Grid size: %d, Block size: %d\n", grid.x, block.x);

    cudaCheck(cudaFuncSetAttribute(
        groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                        CLUSTER_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Launch kernel for group 0 as test
    groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                    CLUSTER_N><<<grid, block, smem_size>>>(
        group_size, Ms[0], Ns[0], Ks[0], tmaCs[0], tmaAs[0], tmaBs[0]);

    cudaCheck(cudaDeviceSynchronize());
  }

  ~GroupGemm() {
    Ms.clear();
    Ns.clear();
    Ks.clear();
    As.clear();
    Bs.clear();
    Cs.clear();
    tmaAs.clear();
    tmaBs.clear();
    tmaCs.clear();
    group_size = 0;
  }
};

} // namespace groupgemm
