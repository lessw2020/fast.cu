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

////

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
/*
template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;
  int group_idx;
  const GemmParams *params;
  bool valid;
  int blocks_per_group;

  __device__ __forceinline__ Schedule(int block_idx, int group_size,
                                      const GemmParams *all_params) {
    blocks_per_group = NUM_SM / (TM * TN);
    block = block_idx % blocks_per_group;
    group_idx = block_idx / blocks_per_group;
    valid = (group_idx < group_size);

    if (valid) {
      params = &all_params[group_idx];
      it = 0;
      // Round up dimensions to block size
      total_blocks_m = CEIL_DIV(params->M, BM);
      total_blocks_n = CEIL_DIV(params->N, BN);
    }
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    if (!valid)
      return false;

    int total_tiles =
        CEIL_DIV(total_blocks_m, TM) * CEIL_DIV(total_blocks_n, TN);
    int num = it * blocks_per_group + block;
    if (num >= total_tiles * (TM * TN))
      return false;

    int tiles_m = CEIL_DIV(total_blocks_m, TM);
    int tiles_n = CEIL_DIV(total_blocks_n, TN);
    int tile_size = TM * TN;

    int tile_idx = num / tile_size;
    int pos_in_tile = num % tile_size;

    int tile_m = tile_idx / tiles_n;
    int tile_n = tile_idx % tiles_n;

    block_m = tile_m * TM + pos_in_tile / TN;
    block_n = tile_n * TN + pos_in_tile % TN;

    ++it;
    return (block_m < total_blocks_m) && (block_n < total_blocks_n);
  }

  __device__ __forceinline__ bool is_valid() const { return valid; }
};

template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;
  int group_idx;
  const GemmParams *params;
  bool valid;
  int blocks_per_group;
  int blocks_per_cluster;

  __device__ __forceinline__ Schedule(int block_idx, int group_size,
                                      const GemmParams *all_params) {
    constexpr int CLUSTERS = (TM * TN);
    blocks_per_cluster = NUM_SM / CLUSTERS;
    blocks_per_group = blocks_per_cluster / group_size;

    block = block_idx % blocks_per_group;
    group_idx = block_idx / blocks_per_group;
    valid = (group_idx < group_size);

    if (valid) {
      params = &all_params[group_idx];
      it = 0;
      // Calculate total blocks accounting for clusters
      total_blocks_m = CEIL_DIV(params->M, BM * TM);
      total_blocks_n = CEIL_DIV(params->N, BN * TN);

      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Schedule init: block_idx=%d, group_size=%d, group_idx=%d\n",
               block_idx, group_size, group_idx);
        printf("Dimensions: M=%d, N=%d, K=%d\n", params->M, params->N,
               params->K);
        printf("Total blocks: M=%d, N=%d\n", total_blocks_m, total_blocks_n);
        printf("Blocks per group=%d, Blocks per cluster=%d\n", blocks_per_group,
               blocks_per_cluster);
      }
    }
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    if (!valid)
      return false;

    int total_blocks = total_blocks_m * total_blocks_n;
    int block_num = it * blocks_per_group + block;

    if (block_num >= total_blocks)
      return false;

    block_m = block_num / total_blocks_n;
    block_n = block_num % total_blocks_n;

    // Debug print first iteration
    if (threadIdx.x == 0 && blockIdx.x == 0 && it == 0) {
      printf("First block assignment: block_m=%d, block_n=%d\n", block_m,
             block_n);
    }

    ++it;
    return true;
  }

template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;
  int group_idx;
  const GemmParams *params;
  bool valid;

  __device__ __forceinline__ Schedule(int block_idx, int group_size,
                                      const GemmParams *all_params) {
    constexpr int blocks_per_sm = NUM_SM / (TM * TN);
    block = block_idx % blocks_per_sm;
    group_idx = block_idx / blocks_per_sm;
    valid = (group_idx < group_size);

    if (valid) {
      params = &all_params[group_idx];
      it = 0;

      // Calculate blocks matching matmul_10's pattern
      total_blocks_m = CEIL_DIV(params->M, BM);
      total_blocks_n = CEIL_DIV(params->N, BN);

      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Schedule init: block_idx=%d, group_size=%d, group_idx=%d\n",
               block_idx, group_size, group_idx);
        printf("Blocks per SM: %d\n", blocks_per_sm);
        printf("Dimensions: M=%d, N=%d, K=%d\n", params->M, params->N,
               params->K);
        printf("Total blocks: M=%d, N=%d\n", total_blocks_m, total_blocks_n);
      }
    }
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    if (!valid)
      return false;

    int num = it * NUM_SM / (TM * TN) + block;
    if (num >= total_blocks_m * total_blocks_n)
      return false;

    // Matching matmul_10 tile calculation
    int cur_tile = num / (TM * TN);
    int cur_tile_pos = num % (TM * TN);

    block_m = TM * (cur_tile / (total_blocks_n / TN));
    block_n = TN * (cur_tile % (total_blocks_n / TN));
    block_m += cur_tile_pos / TN;
    block_n += cur_tile_pos % TN;

    if (threadIdx.x == 0 && blockIdx.x == 0 && it == 0) {
      printf("First iteration: block_m=%d, block_n=%d\n", block_m, block_n);
    }

    ++it;
    return (block_m < total_blocks_m) && (block_n < total_blocks_n);
  }

  __device__ __forceinline__ bool is_valid() const { return valid; }
};
*/
template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;
  int group_idx;
  const GemmParams *params;
  bool valid;

  __device__ __forceinline__ Schedule(int block_idx, int group_size,
                                      const GemmParams *all_params) {
    constexpr int CLUSTERS =
        (16 / TM) * (8 / TN); // Match matmul_10's cluster count calculation
    constexpr int blocks_per_sm = NUM_SM / CLUSTERS;
    block = block_idx % blocks_per_sm;
    group_idx = block_idx / blocks_per_sm;
    valid = (group_idx < group_size);

    if (valid) {
      params = &all_params[group_idx];
      it = 0;
      total_blocks_m = CEIL_DIV(params->M, BM);
      total_blocks_n = CEIL_DIV(params->N, BN);

      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Schedule init: block_idx=%d, group_size=%d, group_idx=%d\n",
               block_idx, group_size, group_idx);
        printf("Clusters=%d, Blocks per SM=%d\n", CLUSTERS, blocks_per_sm);
        printf("Dimensions: M=%d, N=%d, K=%d\n", params->M, params->N,
               params->K);
        printf("Total blocks: M=%d, N=%d\n", total_blocks_m, total_blocks_n);
      }
    }
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    if (!valid)
      return false;

    constexpr int CLUSTERS = (16 / TM) * (8 / TN);
    constexpr int blocks_per_sm = NUM_SM / CLUSTERS;
    int num = it * blocks_per_sm + block;

    // Calculate total tiles needed
    int tiles_m = CEIL_DIV(total_blocks_m, TM);
    int tiles_n = CEIL_DIV(total_blocks_n, TN);
    int total_tiles = tiles_m * tiles_n;

    if (num >= total_tiles * (TM * TN))
      return false;

    int cur_tile = num / (TM * TN);
    int cur_tile_pos = num % (TM * TN);

    block_m = TM * (cur_tile / tiles_n);
    block_n = TN * (cur_tile % tiles_n);
    block_m += cur_tile_pos / TN;
    block_n += cur_tile_pos % TN;

    if (threadIdx.x == 0 && blockIdx.x == 0 && it == 0) {
      printf("First iteration: num=%d, block_m=%d, block_n=%d\n", num, block_m,
             block_n);
      printf("Tiles: M=%d, N=%d, total=%d\n", tiles_m, tiles_n, total_tiles);
    }

    ++it;
    return (block_m < total_blocks_m) && (block_n < total_blocks_n);
  }

  __device__ __forceinline__ bool is_valid() const { return valid; }
};

// GroupGEMM Kernel
template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int NUM_SM = 128, int CLUSTER_M = 2, int CLUSTER_N = 1>
__global__
__launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M *CLUSTER_N, 1, 1)
    groupGemmKernel(int group_size, const GemmParams *params,
                    const GemmDescriptors *descs) {
  constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  constexpr int B_WG_M = BM / num_consumers;
  constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

  extern __shared__ __align__(128) uint8_t smem[];
  SharedMemoryLayout<BM, BN, BK, QSIZE> &s =
      *reinterpret_cast<SharedMemoryLayout<BM, BN, BK, QSIZE> *>(smem);

  bf16 *sA = s.A, *sB = s.B, *sC = s.C;
  uint64_t *full = s.full, *empty = s.empty;

  // Initialize barriers
  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
      PTXBarrier::init_barrier(&full[i], 0, 1);
      PTXBarrier::init_barrier(&empty[i], 0, num_consumers * CLUSTERS);
    }
  }

  ClusterOps::cluster_sync();

  uint32_t cluster_id = ClusterOps::get_cluster_id();

  Schedule<1, NUM_SM / CLUSTERS, BM * CLUSTER_M, BN * CLUSTER_N, 16 / CLUSTER_M,
           8 / CLUSTER_N>
      schedule(blockIdx.x, group_size, params);

  if (!schedule.is_valid())
    return;

  const GemmDescriptors &desc = descs[schedule.group_idx];
  const int num_blocks_k = (schedule.params->K + BK - 1) / BK;
  int wg_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;

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
          bool valid_k = k_offset < schedule.params->K;

          if (valid_k) {
            if (CLUSTER_N > 1) {
              uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
              if (rank_n == 0) {
                TMAOps::load_async_multicast(&sA[qidx * BK * BM], &desc.tma_A,
                                             &full[qidx], k_offset,
                                             num_block_m * BM, mask);
              }
            } else {
              TMAOps::load_async(&sA[qidx * BK * BM], &desc.tma_A, &full[qidx],
                                 k_offset, num_block_m * BM);
            }

            if (CLUSTER_M > 1) {
              if (rank_m == 0) {
                TMAOps::load_async_multicast(
                    &sB[qidx * BK * BN], &desc.tma_B, &full[qidx], k_offset,
                    num_block_n * BN, col_mask << rank_n);
              }
            } else {
              TMAOps::load_async(&sB[qidx * BK * BN], &desc.tma_B, &full[qidx],
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
        TMAOps::store_async(&desc.tma_C, (bf16 *)&sC[0], num_block_m * BM,
                            num_block_n * BN);
        WGMMAGlobalStore::commit_group();
      }
    }
  }
}
/*
template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int CLUSTER_M = 2, int CLUSTER_N = 1, int NUM_SM = 128>
class GroupGemm {
private:
  std::vector<GemmParams> params;
  std::vector<GemmDescriptors> descs;
  int group_size;

  GemmDescriptors createDescriptors(const GemmParams &param) {
    GemmDescriptors desc;
    // Ensure proper alignment for TMA
    assert(param.M % BM == 0 && param.N % BN == 0 && param.K % BK == 0);

    // Create descriptors with synchronization
    cudaCheck(cudaDeviceSynchronize());
    desc.tma_A =
        TensorMapManager::create_tensor_map<BM, BK>(param.A, param.M, param.K);
    cudaCheck(cudaDeviceSynchronize());
    desc.tma_B =
        TensorMapManager::create_tensor_map<BN, BK>(param.B, param.N, param.K);
    cudaCheck(cudaDeviceSynchronize());
    desc.tma_C = TensorMapManager::create_tensor_map<BN, BM, false>(
        param.C, param.N, param.M);
    cudaCheck(cudaDeviceSynchronize());

    return desc;
  }

public:
  GroupGemm() : group_size(0) {}

  void initializeBatch(const std::vector<GemmParams> &batch_params) {
    // Cleanup previous state
    descs.clear();
    params.clear();
    cudaCheck(cudaDeviceSynchronize());

    params = batch_params;
    group_size = batch_params.size();

    // Create descriptors for each GEMM
    descs.reserve(group_size);
    for (const auto &param : batch_params) {
      descs.push_back(createDescriptors(param));
    }
    cudaCheck(cudaDeviceSynchronize());
  }

  void launch() {
    if (group_size == 0)
      return;

    static_assert(NUM_SM % (CLUSTER_M * CLUSTER_N) == 0);
    constexpr int blocks_per_sm = NUM_SM / (CLUSTER_M * CLUSTER_N);

    // Ensure grid size is proper multiple of cluster size
    int grid_size = blocks_per_sm * group_size;
    grid_size = (grid_size + CLUSTER_M * CLUSTER_N - 1) /
                (CLUSTER_M * CLUSTER_N) * (CLUSTER_M * CLUSTER_N);

    dim3 grid(grid_size);
    dim3 block(NUM_THREADS);

    size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);

    cudaCheck(cudaFuncSetAttribute(
        groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                        CLUSTER_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaCheck(cudaDeviceSynchronize());

    // Allocate and copy parameters and descriptors to device
    GemmParams *d_params;
    GemmDescriptors *d_descs;
    cudaCheck(cudaMalloc(&d_params, group_size * sizeof(GemmParams)));
    cudaCheck(cudaMalloc(&d_descs, group_size * sizeof(GemmDescriptors)));
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(d_params, params.data(),
                         group_size * sizeof(GemmParams),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_descs, descs.data(),
                         group_size * sizeof(GemmDescriptors),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());

    // Launch kernel
    groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                    CLUSTER_N>
        <<<grid, block, smem_size>>>(group_size, d_params, d_descs);
    cudaCheck(cudaDeviceSynchronize());

    // Cleanup
    cudaCheck(cudaFree(d_params));
    cudaCheck(cudaFree(d_descs));
  }

  ~GroupGemm() {
    descs.clear();
    params.clear();
    group_size = 0;
  }
};
*/
template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int CLUSTER_M = 2, int CLUSTER_N = 1, int NUM_SM = 128>
class GroupGemm {
private:
  std::vector<GemmParams> params;
  std::vector<GemmDescriptors> descs;
  int group_size;

  GemmDescriptors createDescriptors(const GemmParams &param) {
    printf("Creating descriptors for GEMM with dims: M=%d, N=%d, K=%d\n",
           param.M, param.N, param.K);

    GemmDescriptors desc;

    // Ensure proper alignment and print details
    printf("Checking alignments: BM=%d, BN=%d, BK=%d\n", BM, BN, BK);
    printf("M mod BM = %d, N mod BN = %d, K mod BK = %d\n", param.M % BM,
           param.N % BN, param.K % BK);

    // Create descriptors with error checking
    printf("Creating TMA descriptor for A...\n");
    cudaCheck(cudaDeviceSynchronize());
    desc.tma_A =
        TensorMapManager::create_tensor_map<BM, BK>(param.A, param.M, param.K);

    printf("Creating TMA descriptor for B...\n");
    cudaCheck(cudaDeviceSynchronize());
    desc.tma_B =
        TensorMapManager::create_tensor_map<BN, BK>(param.B, param.N, param.K);

    printf("Creating TMA descriptor for C...\n");
    cudaCheck(cudaDeviceSynchronize());
    desc.tma_C = TensorMapManager::create_tensor_map<BN, BM, false>(
        param.C, param.N, param.M);

    printf("All descriptors created successfully\n");
    return desc;
  }

public:
  GroupGemm() : group_size(0) {}

  void initializeBatch(const std::vector<GemmParams> &batch_params) {
    printf("\nInitializing batch with %zu GEMMs\n", batch_params.size());

    // Cleanup previous state
    descs.clear();
    params.clear();
    cudaCheck(cudaDeviceSynchronize());

    params = batch_params;
    group_size = batch_params.size();

    // Create descriptors for each GEMM
    descs.reserve(group_size);
    for (int i = 0; i < group_size; ++i) {
      printf("\nCreating descriptors for GEMM %d\n", i);
      descs.push_back(createDescriptors(batch_params[i]));
    }
    printf("All batch descriptors created\n");
    cudaCheck(cudaDeviceSynchronize());
  }
  /*
  void launch() {
    if (group_size == 0)
      return;

    printf("\nPreparing to launch kernel\n");
    static_assert(NUM_SM % (CLUSTER_M * CLUSTER_N) == 0);
    constexpr int blocks_per_sm = NUM_SM / (CLUSTER_M * CLUSTER_N);

    // Calculate grid size matching matmul_10
    int grid_size = blocks_per_sm * group_size;

    printf("Launch config:\n");
    printf("- Total SMs: %d\n", NUM_SM);
    printf("- Clusters: %d\n", CLUSTER_M * CLUSTER_N);
    printf("- Blocks per SM: %d\n", blocks_per_sm);
    printf("- Total grid size: %d\n", grid_size);
    printf("- Block size: %d\n", NUM_THREADS);
    printf("- Shared memory: %zu bytes\n",
           sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>));

    dim3 grid(grid_size);
    dim3 block(NUM_THREADS);

    size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);

    printf("Setting shared memory size...\n");
    cudaCheck(cudaFuncSetAttribute(
        groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                        CLUSTER_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    cudaCheck(cudaDeviceSynchronize());
    */
  void launch() {
    if (group_size == 0)
      return;

    printf("\nPreparing to launch kernel\n");
    static_assert(NUM_SM % (CLUSTER_M * CLUSTER_N) == 0);

    // Match matmul_10's cluster calculations
    constexpr int CLUSTERS = (16 / CLUSTER_M) * (8 / CLUSTER_N);
    constexpr int blocks_per_sm = NUM_SM / CLUSTERS;
    int grid_size = blocks_per_sm * group_size;

    printf("Launch config:\n");
    printf("- Total SMs: %d\n", NUM_SM);
    printf("- Clusters: %d\n", CLUSTERS);
    printf("- Blocks per SM: %d\n", blocks_per_sm);
    printf("- Total grid size: %d\n", grid_size);
    printf("- Block size: %d\n", NUM_THREADS);
    printf("- Shared memory: %zu bytes\n",
           sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>));

    dim3 grid(grid_size);
    dim3 block(NUM_THREADS);

    size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);

    printf("Setting shared memory size...\n");
    cudaCheck(cudaFuncSetAttribute(
        groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                        CLUSTER_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    cudaCheck(cudaDeviceSynchronize());
    // Allocate device memory
    printf("Allocating device memory...\n");
    GemmParams *d_params;
    GemmDescriptors *d_descs;
    cudaCheck(cudaMalloc(&d_params, group_size * sizeof(GemmParams)));
    cudaCheck(cudaMalloc(&d_descs, group_size * sizeof(GemmDescriptors)));
    cudaCheck(cudaDeviceSynchronize());

    // Copy data to device
    printf("Copying data to device...\n");
    cudaCheck(cudaMemcpy(d_params, params.data(),
                         group_size * sizeof(GemmParams),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_descs, descs.data(),
                         group_size * sizeof(GemmDescriptors),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());

    // Launch kernel
    printf("Launching kernel...\n");
    groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                    CLUSTER_N>
        <<<grid, block, smem_size>>>(group_size, d_params, d_descs);

    cudaCheck(cudaDeviceSynchronize());

    // Cleanup
    cudaCheck(cudaFree(d_params));
    cudaCheck(cudaFree(d_descs));
  }
  /*
  void launch() {
    if (group_size == 0)
      return;

    printf("\nPreparing to launch kernel\n");
    static_assert(NUM_SM % (CLUSTER_M * CLUSTER_N) == 0);

    // Calculate launch dimensions
    constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;
    int blocks_per_cluster = NUM_SM / CLUSTERS;
    int blocks_per_group = blocks_per_cluster / group_size;
    int grid_size = blocks_per_group * group_size;

    printf("Launch config:\n");
    printf("- Total SMs: %d\n", NUM_SM);
    printf("- Clusters: %d\n", CLUSTERS);
    printf("- Blocks per cluster: %d\n", blocks_per_cluster);
    printf("- Blocks per group: %d\n", blocks_per_group);
    printf("- Total grid size: %d\n", grid_size);
    printf("- Block size: %d\n", NUM_THREADS);
    printf("- Shared memory: %zu bytes\n",
           sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>));

    dim3 grid(grid_size);
    dim3 block(NUM_THREADS);

    size_t smem_size = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);

    printf("Setting shared memory size...\n");
    cudaCheck(cudaFuncSetAttribute(
        groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                        CLUSTER_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    cudaCheck(cudaDeviceSynchronize());

    // Allocate device memory
    printf("Allocating device memory...\n");
    GemmParams *d_params;
    GemmDescriptors *d_descs;
    cudaCheck(cudaMalloc(&d_params, group_size * sizeof(GemmParams)));
    cudaCheck(cudaMalloc(&d_descs, group_size * sizeof(GemmDescriptors)));
    cudaCheck(cudaDeviceSynchronize());

    // Copy data to device
    printf("Copying data to device...\n");
    cudaCheck(cudaMemcpy(d_params, params.data(),
                         group_size * sizeof(GemmParams),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_descs, descs.data(),
                         group_size * sizeof(GemmDescriptors),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());

    // Launch kernel
    printf("Launching kernel...\n");
    groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                    CLUSTER_N>
        <<<grid, block, smem_size>>>(group_size, d_params, d_descs);

    printf("Waiting for kernel completion...\n");
    cudaCheck(cudaDeviceSynchronize());
    printf("Kernel completed\n");

    // Cleanup
    printf("Cleaning up device memory...\n");
    cudaCheck(cudaFree(d_params));
    cudaCheck(cudaFree(d_descs));
    printf("Launch completed successfully\n");
  }*/

  ~GroupGemm() {
    descs.clear();
    params.clear();
    group_size = 0;
  }
};

} // namespace groupgemm
