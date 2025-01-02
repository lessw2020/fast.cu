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
      total_blocks_m = (params->M + BM - 1) / BM;
      total_blocks_n = (params->N + BN - 1) / BN;
    }
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    if (!valid)
      return false;

    int num = it * blocks_per_group + block;
    if (num >= total_blocks_m * total_blocks_n)
      return false;

    int cur_tile = num / (TM * TN);
    int cur_tile_pos = num % (TM * TN);
    block_m = TM * (cur_tile / (total_blocks_n / TN));
    block_n = TN * (cur_tile % (total_blocks_n / TN));
    block_m += cur_tile_pos / TN;
    block_n += cur_tile_pos % TN;

    ++it;
    return (block_m * BM < params->M) && (block_n * BN < params->N);
  }

  __device__ __forceinline__ bool is_valid() const { return valid; }
};

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
  /////////
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
#pragma unroll
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

      for (int block_k_iter = 0; block_k_iter < num_blocks_k;
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

          // First K iteration with clear accumulator
          wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[m_it], &wgmma_sA[0], &wgmma_sB[0]);

// Remaining K iterations with accumulate
#pragma unroll
          for (int k_it = 1; k_it < BK / WGMMA_K; ++k_it) {
            wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
                                          &wgmma_sB[k_it * WGMMA_K]);
          }
        }

        WGMMASyncOps::warpgroup_commit_batch();
        WGMMASyncOps::warpgroup_wait<0>();
        if (tid < CLUSTERS)
          PTXBarrier::arrive_cluster(&empty[qidx], tid);
      }

      // Store results using WGMMA output handler
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

template <int BM = 128, int BN = 256, int BK = 64, int NUM_THREADS = 128 * 3,
          int QSIZE = 3, int CLUSTER_M = 2, int CLUSTER_N = 1, int NUM_SM = 128>
class GroupGemm {
private:
  std::vector<GemmParams> params;
  std::vector<GemmDescriptors> descs;
  int group_size;

  GemmDescriptors createDescriptors(const GemmParams &param) {
    GemmDescriptors desc;
    cudaCheck(cudaDeviceSynchronize()); // Ensure previous operations complete
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

  void initializeBatch(const std::vector<GemmParams> &batch_params) {
    // Cleanup previous state
    descs.clear();
    params.clear();

    params = batch_params;
    group_size = batch_params.size();

    // Create descriptors for each GEMM
    descs.reserve(group_size);
    for (const auto &param : batch_params) {
      descs.push_back(createDescriptors(param));
    }

    // Ensure descriptors are ready
    cudaCheck(cudaDeviceSynchronize());
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

    // Ensure previous operations are complete
    cudaCheck(cudaDeviceSynchronize());

    // Allocate and copy parameters and descriptors to device
    GemmParams *d_params;
    GemmDescriptors *d_descs;
    cudaCheck(cudaMalloc(&d_params, group_size * sizeof(GemmParams)));
    cudaCheck(cudaMalloc(&d_descs, group_size * sizeof(GemmDescriptors)));

    cudaCheck(cudaMemcpy(d_params, params.data(),
                         group_size * sizeof(GemmParams),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_descs, descs.data(),
                         group_size * sizeof(GemmDescriptors),
                         cudaMemcpyHostToDevice));

    // Ensure copies are complete before kernel launch
    cudaCheck(cudaDeviceSynchronize());

    groupGemmKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM, CLUSTER_M,
                    CLUSTER_N>
        <<<grid, block, smem_size>>>(group_size, d_params, d_descs);

    // Wait for kernel to complete before freeing memory
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaFree(d_params));
    cudaCheck(cudaFree(d_descs));
  }

  ~GroupGemm() {
    descs.clear();
    params.clear();
    group_size = 0;
  }
};

} // namespace groupgemm
