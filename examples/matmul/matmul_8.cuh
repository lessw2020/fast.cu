#include "wgmini.cuh";
namespace M8 {
using namespace wgmma_utils;

__device__ uint64_t make_smem_desc(bf16 *ptr) {
  return WGMMADescriptor::make_smem_desc(ptr);
}

__device__ void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N> __device__ void warpgroup_wait() {
  static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap
create_tensor_map(bf16 *gmem_ptr, int global_height, int global_width) {
  CUtensorMap tma_map;
  void *gmem_address = (void *)gmem_ptr;
  static_assert(BlockMinorSize >= 64);
  assert(global_width % 64 == 0);
  uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height,
                                 (uint64_t)global_width / 64, 1, 1};
  uint64_t gmem_prob_stride[5] = {sizeof(bf16) * global_width,
                                  64 * sizeof(bf16), 0, 0, 0};
  uint32_t smem_box_shape[5] = {64, uint32_t(BlockMajorSize),
                                uint32_t(BlockMinorSize / 64), 1, 1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

  CUresult result = cuTensorMapEncodeTiled(
      &tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gmem_address,
      gmem_prob_shape, gmem_prob_stride, smem_box_shape, smem_box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(result == CUDA_SUCCESS);
  return tma_map;
}

CUtensorMap d_tma_map_A;
CUtensorMap d_tma_map_B;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

template <int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA,
          int TransB>
__device__ __forceinline__ void wgmma(float d[WGMMA_N / 16][8], bf16 *sA,
                                      bf16 *sB) {
  static_assert(WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 ||
                WGMMA_N == 192 || WGMMA_N == 208 || WGMMA_N == 256);
  if constexpr (WGMMA_N == 256)
    wgmma256<DefaultConfig>(d, sA, sB);
  if constexpr (WGMMA_N == 192)
    wgmma192<DefaultConfig>(d, sA, sB);
  if constexpr (WGMMA_N == 128)
    wgmma128<DefaultConfig>(d, sA, sB);
  if constexpr (WGMMA_N == 64)
    wgmma64<DefaultConfig>(d, sA, sB);
  if constexpr (WGMMA_N == 32)
    wgmma32<DefaultConfig>(d, sA, sB);
}

template <int BM, int BN, int BK, int QSIZE> struct SMem {
  alignas(128) bf16 A[BM * BK * QSIZE];
  alignas(128) bf16 B[BK * BN * QSIZE];
};

__device__ static __forceinline__ void
init_barrier(uint64_t *bar, int thread_count, int transaction_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(thread_count + transaction_count));
}

__device__ static __forceinline__ void expect_bytes(uint64_t *bar,
                                                    uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
      "r"(bytes));
}

__device__ static inline void load_async(bf16 *dst,
                                         void const *const src_tma_map,
                                         uint64_t *bar, int global_col_idx,
                                         int global_row_idx) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
               "complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2];"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0),
                 "r"(global_row_idx), "r"(global_col_idx / 64)
               : "memory");
}

__device__ static __forceinline__ void wait(uint64_t *bar, int kPhaseBit) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(kPhaseBit));
}

__device__ static __forceinline__ void arrive(uint64_t *bar,
                                              uint32_t count = 1) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(mbar_ptr), "r"(count)
               : "memory");
}

__device__ static __forceinline__ void wait_cluster(uint64_t *bar,
                                                    int kPhaseBit) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(mbar_ptr),
      "r"(kPhaseBit));
}

__device__ static inline void
load_async_multicast(bf16 *dst, void const *const src_tma_map, uint64_t *bar,
                     int global_col_idx, int global_row_idx,
                     uint16_t cluster_mask) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
               "complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :
               : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0),
                 "r"(global_row_idx), "r"(global_col_idx / 64),
                 "h"(cluster_mask)
               : "memory");
}

template <int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule;

template <int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<1, NUM_SM, BM, BN, TM, TN> {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;

  __device__ __forceinline__ Schedule(int M, int N, int _block) {
    block = _block;
    it = 0;
    total_blocks_m = CEIL_DIV(M, BM);
    total_blocks_n = CEIL_DIV(N, BN);
    assert(CEIL_DIV(M, BM) % TM == 0 && total_blocks_n % TN == 0);
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

template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM,
          int CLUSTER_M, int CLUSTER_N>
__global__
__launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M *CLUSTER_N, 1, 1)
    matmulKernel8(int M, int N, int K, bf16 *C,
                  const __grid_constant__ CUtensorMap tensorMapA,
                  const __grid_constant__ CUtensorMap tensorMapB) {
  constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  constexpr int B_WG_M = BM / num_consumers;
  constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;
  assert((M / BM) % CLUSTER_M == 0);
  assert((N / BN) % CLUSTER_N == 0);

  extern __shared__ __align__(128) uint8_t smem[];
  SMem<BM, BN, BK, QSIZE> &s =
      *reinterpret_cast<SMem<BM, BN, BK, QSIZE> *>(smem);
  bf16 *sA = s.A;
  bf16 *sB = s.B;
  // Declare barriers
  __shared__ __align__(8) uint64_t full[QSIZE], empty[QSIZE];
  uint32_t rank;
  asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(rank) :);

  const int num_blocks_k = K / BK;
  int wg_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;

  // if (threadIdx.x == 0) {
  // for (int i = 0; i < QSIZE; ++i) {
  //   init_barrier(&full[i], 0, 1);
  //   init_barrier(&empty[i], 0, num_consumers * CLUSTERS);
  // }
  // Initialize barriers using new PTX barrier system
  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
      wgmma_utils::PTXBarrier::init_barrier(&full[i], 0, 1);
      wgmma_utils::PTXBarrier::init_barrier(&empty[i], 0,
                                            num_consumers * CLUSTERS);
    }
  }

  asm volatile("barrier.cluster.arrive;\n" : :);
  asm volatile("barrier.cluster.wait;\n" : :);

  Schedule<1, NUM_SM / CLUSTERS, BM * CLUSTER_M, BN * CLUSTER_N, 16 / CLUSTER_M,
           8 / CLUSTER_N>
      schedule(M, N, rank);

  asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
  uint32_t rank_m = rank / CLUSTER_N;
  uint32_t rank_n = rank % CLUSTER_N;

  // Producer
  if (wg_idx == 0) {
    constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
    // warpgroup_reg_dealloc<num_regs>();
    wgmma_utils::RegisterManager::dealloc<num_regs>();

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
          wgmma_utils::PTXBarrier::wait(&empty[qidx], p);

          wgmma_utils::PTXBarrier::expect_tx(&full[qidx], (BK * BN + BK * BM) *
                                                              sizeof(bf16));
          // wait(&empty[qidx], p);

          // expect_bytes(&full[qidx], (BK * BN + BK * BM) * sizeof(bf16));
          if constexpr (CLUSTER_N > 1) {
            uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
            if (rank_n == 0) {
              load_async_multicast(&sA[qidx * BK * BM], &tensorMapA,
                                   &full[qidx], block_k_iter * BK,
                                   num_block_m * BM, mask);
            }
          } else {
            load_async(&sA[qidx * BK * BM], &tensorMapA, &full[qidx],
                       block_k_iter * BK, num_block_m * BM);
          }

          if constexpr (CLUSTER_M > 1) {
            if (rank_m == 0) {
              load_async_multicast(&sB[qidx * BK * BN], &tensorMapB,
                                   &full[qidx], block_k_iter * BK,
                                   num_block_n * BN, col_mask << rank_n);
            }
          } else {
            load_async(&sB[qidx * BK * BN], &tensorMapB, &full[qidx],
                       block_k_iter * BK, num_block_n * BN);
          }
        }
      }
    }
  } else {
    constexpr int num_regs =
        (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
    // warpgroup_reg_alloc<num_regs>();
    wgmma_utils::RegisterManager::alloc<num_regs>();
    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
    --wg_idx;
    for (int qidx = 0; qidx < QSIZE; ++qidx) {
      if (tid < CLUSTERS)
        // arrive_cluster(&empty[qidx], tid);
        wgmma_utils::PTXBarrier::arrive_cluster(&empty[qidx], tid);
    }
    int p = 0;
    int qidx = 0;
    int lane = tid % 32, warp = tid / 32, row = warp * 16 + lane / 4;
    int num_block_m, num_block_n;
    while (schedule.next(num_block_m, num_block_n)) {
      num_block_n = num_block_n * CLUSTER_N + rank_n;
      num_block_m = num_block_m * CLUSTER_M + rank_m;
      memset(d, 0, sizeof(d));
      for (int block_k_iter = 0; block_k_iter < num_blocks_k;
           ++block_k_iter, ++qidx) {
        if (qidx == QSIZE) {
          qidx = 0;
          p ^= 1;
        };
        // wait(&full[qidx], p);
        // warpgroup_arrive();
        wgmma_utils::PTXBarrier::wait(&full[qidx], p);
        wgmma_utils::WGMMASyncOps::arrive();
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
        // warpgroup_commit_batch();
        // warpgroup_wait<0>();
        wgmma_utils::WGMMASyncOps::commit_group();
        wgmma_utils::WGMMASyncOps::wait_group<0>();
        if (tid < CLUSTERS)
          // arrive_cluster(&empty[qidx], tid);
          wgmma_utils::PTXBarrier::arrive_cluster(&empty[qidx], tid);
      }

      bf16 *block_C = C + num_block_n * BN * M + num_block_m * BM;

#pragma unroll
      for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
        int yo = m_it * WGMMA_M + wg_idx * B_WG_M;
        if (row + 8 + yo + num_block_m * BM >= M)
          continue;
#pragma unroll
        for (int w = 0; w < WGMMA_N; w += 16)
          if (w + num_block_n * BN < N) {
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

void runKernel8(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB) {
  constexpr int BM = 64 * 2;
  constexpr int BN = 256;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 128 * 3;
  constexpr int QSIZE = 3;
  constexpr int CLUSTER_M = 2;
  constexpr int CLUSTER_N = 1;
  constexpr int NUM_SM = 128;
  static_assert(NUM_SM % (CLUSTER_M * CLUSTER_N) == 0);

  if (_prev_m != M) {
    d_tma_map_A = create_tensor_map<BM, BK>(A, M, K);
    d_tma_map_B = create_tensor_map<BN, BK>(B, N, K);
    _prev_m = M;
    _prev_n = N;
    _prev_k = K;
  }
  // Assert cached values are of same size
  assert(M == _prev_m && N == _prev_n && K == _prev_k);
  auto *kernel = matmulKernel8<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM,
                               CLUSTER_M, CLUSTER_N>;
  size_t sMemSize = sizeof(SMem<BM, BN, BK, QSIZE>);
  cudaCheck(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

  kernel<<<NUM_SM, NUM_THREADS, sMemSize>>>(M, N, K, C, d_tma_map_A,
                                            d_tma_map_B);
}

} // namespace M8

using M8::runKernel8;
