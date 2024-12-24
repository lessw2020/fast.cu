#import "new_wgmma_utils.cuh";

namespace M3 {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace wgmmu = new_wgmma_utils;

/* __device__ void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N> __device__ void warpgroup_wait() {
  static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
*/

template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap *tma_map, bf16 *gmem_ptr, int blocks_height,
                       int blocks_width) {
  void *gmem_address = (void *)gmem_ptr;
  uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize * blocks_width,
                                 (uint64_t)BlockMajorSize * blocks_height, 1, 1,
                                 1};
  uint64_t gmem_prob_stride[5] = {
      sizeof(bf16), sizeof(bf16) * BlockMinorSize * blocks_width, 0, 0, 0};
  uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize),
                                uint32_t(BlockMajorSize), 1, 1, 1};
  uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

  CUresult result = cuTensorMapEncodeTiled(
      tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address,
      gmem_prob_shape, gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(result == CUDA_SUCCESS);
}

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m = 0, _prev_n = 0, _prev_k = 0;

template <int st_rows, int st_cols>
__host__ static inline CUtensorMap *
allocate_and_create_tensor_map(bf16 *src, int blocks_height, int blocks_width) {
  CUtensorMap *tma_map_d;
  cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
  CUtensorMap tma_map_host;
  create_tensor_map<st_rows, st_cols>(&tma_map_host, src, blocks_height,
                                      blocks_width);
  cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap),
             cudaMemcpyHostToDevice);
  return tma_map_d;
}

template <int BM, int BN, int BK> struct SMem {
  alignas(128) bf16 A[BM * BK];
  alignas(128) bf16 B[BK * BN];
};

template <int BM, int BN, int BK, int NUM_THREADS, bool DBG>
__global__ void __launch_bounds__(NUM_THREADS)
    matmulKernel3(int M, int N, int K, bf16 *C, const CUtensorMap *tensorMapA,
                  const CUtensorMap *tensorMapB, int *DB) {
  constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
  constexpr int B_WG_M = BM / (NUM_THREADS / 128);
  extern __shared__ SMem<BM, BN, BK> s;
  bf16 *sA = s.A;
  bf16 *sB = s.B;
// Barriers cannot be in the struct and have to be declared this way
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier barA, barB;
  float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
  static_assert(sizeof(d) * NUM_THREADS == BM * BN * sizeof(float));
  memset(d, 0, sizeof(d));

  const int num_blocks_k = K / BK;
  int num_block_n = blockIdx.x % (N / BN);
  int num_block_m = blockIdx.x / (N / BN);

  if (threadIdx.x == 0) {
    init(&barA, blockDim.x);
    init(&barB, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  int wg_idx = threadIdx.x / 128;

  barrier::arrival_token tokenA, tokenB;
  int sumLoad = 0, cntLoad = 0;
  int sumCompute = 0, cntCompute = 0;
  int sumStore = 0, cntStore = 0;

  wgmmu::WGMMA<WGMMA_M, WGMMA_N, WGMMA_K> wgmma_op;

  for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
    clock_t start = clock();
    // Load
    if (threadIdx.x == 0) {
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          &sA[0], tensorMapA, block_k_iter * BK, num_block_m * BM, barA);
      tokenA = cuda::device::barrier_arrive_tx(barA, 1, BK * BM * sizeof(bf16));
      cde::cp_async_bulk_tensor_2d_global_to_shared(
          &sB[0], tensorMapB, block_k_iter * BK, num_block_n * BN, barB);
      tokenB = cuda::device::barrier_arrive_tx(barB, 1, BK * BN * sizeof(bf16));
    } else {
      tokenA = barA.arrive();
      tokenB = barB.arrive();
    }
    barA.wait(std::move(tokenA));
    barB.wait(std::move(tokenB));
    __syncthreads();
    if constexpr (DBG) {
      sumLoad += clock() - start;
      cntLoad++;
      start = clock();
    }

    // Compute
    // warpgroup_arrive();
#pragma unroll
    for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
      bf16 *wgmma_sA = sA + BK * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
      // Use accumulation mode based on iteration

#pragma unroll
      for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
        // wgmma_tc<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
        //                                  &sB[k_it * WGMMA_K]);
        if (block_k_iter == 0) {
          wgmma_op.init_multiply(d[m_it], &wgmma_sA[k_it * WGMMA_K],
                                 &sB[k_it * WGMMA_K]);
        } else {
          wgmma_op.accumulate_multiply(d[m_it], &wgmma_sA[k_it * WGMMA_K],
                                       &sB[k_it * WGMMA_K]);
        }
      }
    }
    // warpgroup_commit_batch();
    // warpgroup_wait<0>();
    wgmma_op.commit_batch();
    wgmma_op.wait_batch();

    if constexpr (DBG) {
      sumCompute += clock() - start;
      cntCompute++;
    }
  }

  // Store
  {
    clock_t start = clock();

    uint32_t tid = threadIdx.x % 128;
    uint32_t lane = tid & 31;
    uint32_t warp = tid / 32;
    uint32_t row = warp * 16 + lane / 4;

    bf16 *block_C = C + num_block_n * BN * M + num_block_m * BM;

#pragma unroll
    for (uint32_t m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
      int yo = m_it * WGMMA_M + wg_idx * B_WG_M;
#pragma unroll
      for (uint32_t w = 0; w < WGMMA_N / 16; ++w) {
        int col = 16 * w + 2 * (tid % 4);
#define IDX(i, j) ((j) * M + ((i) + yo))

        block_C[IDX(row, col)] = d[m_it][w][0];
        block_C[IDX(row, col + 1)] = d[m_it][w][1];
        block_C[IDX(row + 8, col)] = d[m_it][w][2];
        block_C[IDX(row + 8, col + 1)] = d[m_it][w][3];
        block_C[IDX(row, col + 8)] = d[m_it][w][4];
        block_C[IDX(row, col + 9)] = d[m_it][w][5];
        block_C[IDX(row + 8, col + 8)] = d[m_it][w][6];
        block_C[IDX(row + 8, col + 9)] = d[m_it][w][7];

#undef IDX
      }
    }
    if constexpr (DBG) {
      sumStore += clock() - start;
      cntStore++;
      if (threadIdx.x == 63) {
        int i = blockIdx.x * 6;
        DB[i] = sumLoad;
        DB[i + 1] = cntLoad;
        DB[i + 2] = sumCompute;
        DB[i + 3] = cntCompute;
        DB[i + 4] = sumStore;
        DB[i + 5] = cntStore;
      }
    }
  }
}

void runKernel3(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 128;

  if (!d_tma_map_A) {
    d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
    d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
    _prev_m = M;
    _prev_n = N;
    _prev_k = K;
  }
  // Assert cached values are of same size
  assert(M == _prev_m && N == _prev_n && K == _prev_k);
  auto *kernel = DB ? matmulKernel3<BM, BN, BK, NUM_THREADS, true>
                    : matmulKernel3<BM, BN, BK, NUM_THREADS, false>;
  size_t sMemSize = sizeof(SMem<BM, BN, BK>);
  cudaCheck(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

  kernel<<<(M / BM) * (N / BN), NUM_THREADS, sMemSize>>>(
      M, N, K, C, d_tma_map_A, d_tma_map_B, DB);
}

} // namespace M3

using M3::runKernel3;
