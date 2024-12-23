#include "wgmma_utils.cuh"
namespace M4 {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace wgmmu = wgmma_utils;

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
  return (((x) & 0x3FFFF) >> 0x4);
}

__device__ uint64_t make_smem_desc(bf16 *ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0x0000000000000000;
  desc |= matrix_descriptor_encode(addr);
  desc |= matrix_descriptor_encode((uint64_t)16) << 16;
  desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
  desc |= 1llu << 62; // 128B swizzle
  return desc;
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

template <int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA,
          int TransB>
__device__ __forceinline__ void wgmma_tc(float d[WGMMA_N / 16][8], bf16 *sA,
                                         bf16 *sB) {
  static_assert(WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 ||
                WGMMA_N == 192 || WGMMA_N == 208 || WGMMA_N == 256);
  if constexpr (WGMMA_N == 256)
    wgmmu::wgmma256<1, 1, 1, 0, 0>(d, sA, sB);
  if constexpr (WGMMA_N == 192)
    wgmmu::wgmma192<1, 1, 1, 0, 0>(d, sA, sB);
  if constexpr (WGMMA_N == 128)
    wgmmu::wgmma128<1, 1, 1, 0, 0>(d, sA, sB);
  if constexpr (WGMMA_N == 64)
    wgmmu::wgmma64<1, 1, 1, 0, 0>(d, sA, sB);
  if constexpr (WGMMA_N == 32)
    wgmmu::wgmma32<1, 1, 1, 0, 0>(d, sA, sB);
}

template <int BM, int BN, int BK, int QSIZE> struct SMem {
  alignas(128) bf16 A[BM * BK * QSIZE];
  alignas(128) bf16 B[BK * BN * QSIZE];
};

template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE>
__global__ __launch_bounds__(NUM_THREADS) void matmulKernel4(
    int M, int N, int K, bf16 *C, const CUtensorMap *tensorMapA,
    const CUtensorMap *tensorMapB) {
  constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  constexpr int B_WG_M = BM / num_consumers;

  extern __shared__ __align__(128) uint8_t smem[];
  SMem<BM, BN, BK, QSIZE> &s =
      *reinterpret_cast<SMem<BM, BN, BK, QSIZE> *>(smem);
  bf16 *sA = s.A;
  bf16 *sB = s.B;
// Barriers cannot be in the struct and have to be declared this way
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier full[QSIZE], empty[QSIZE];

  const int num_blocks_k = K / BK;
  int num_block_n = blockIdx.x % (N / BN);
  int num_block_m = blockIdx.x / (N / BN);
  int wg_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;

  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
      init(&full[i], num_consumers * 128 + 1);
      init(&empty[i], num_consumers * 128 + 1);
    }
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // Producer
  if (wg_idx == 0) {
    constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
    if (tid == 0) {
      int qidx = 0;
      for (int block_k_iter = 0; block_k_iter < num_blocks_k;
           ++block_k_iter, ++qidx) {
        if (qidx == QSIZE)
          qidx = 0;
        empty[qidx].wait(empty[qidx].arrive());
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &sA[qidx * BK * BM], tensorMapA, block_k_iter * BK,
            num_block_m * BM, full[qidx]);
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &sB[qidx * BK * BN], tensorMapB, block_k_iter * BK,
            num_block_n * BN, full[qidx]);
        barrier::arrival_token _ = cuda::device::barrier_arrive_tx(
            full[qidx], 1, (BK * BN + BK * BM) * sizeof(bf16));
      }
    }
  } else {
    for (int i = 0; i < QSIZE; ++i) {
      barrier::arrival_token _ = empty[i].arrive();
    }
    float d[BM / WGMMA_M][WGMMA_N / 16][8];
    memset(d, 0, sizeof(d));
    int qidx = 0;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k;
         ++block_k_iter, ++qidx) {
      if (qidx == QSIZE)
        qidx = 0;
      full[qidx].wait(full[qidx].arrive());
      warpgroup_arrive();
#pragma unroll
      for (int m_it = 0; m_it < BM / WGMMA_M; ++m_it) {
        bf16 *wgmma_sA = sA + qidx * BK * BM + BK * m_it * WGMMA_M;
#pragma unroll
        for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
          wgmma_tc<WGMMA_N, 1, 1, 1, 0, 0>(
              d[m_it], &wgmma_sA[k_it * WGMMA_K],
              &sB[qidx * BK * BN + k_it * WGMMA_K]);
        }
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      barrier::arrival_token _ = empty[qidx].arrive();
    }

    tid = threadIdx.x % 128;
    int lane = tid % 32;
    int warp = tid / 32;
    int row = warp * 16 + lane / 4;
    bf16 *block_C = C + num_block_n * BN * M + num_block_m * BM;

#pragma unroll
    for (int m_it = 0; m_it < BM / WGMMA_M; ++m_it) {
      int yo = m_it * WGMMA_M;
#pragma unroll
      for (int w = 0; w < WGMMA_N / 16; ++w) {
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
  }
}

void runKernel4(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C, int *DB) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 128 * 2;
  constexpr int QSIZE = 5;

  if (!d_tma_map_A) {
    d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
    d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
    _prev_m = M;
    _prev_n = N;
    _prev_k = K;
  }
  // Assert cached values are of same size
  assert(M == _prev_m && N == _prev_n && K == _prev_k);
  auto *kernel = matmulKernel4<BM, BN, BK, NUM_THREADS, QSIZE>;
  size_t sMemSize = sizeof(SMem<BM, BN, BK, QSIZE>);
  cudaCheck(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

  kernel<<<(M / BM) * (N / BN), NUM_THREADS, sMemSize>>>(
      M, N, K, C, d_tma_map_A, d_tma_map_B);
}

} // namespace M4

using M4::runKernel4;
