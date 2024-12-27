#import "wgmma_utils.cuh";

namespace M4 {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace wgmmu = wgmma_utils;
using namespace wgmmu;

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

template <int BM, int BN, int BK, int QSIZE> struct SMem {
  alignas(128) bf16 A[BM * BK * QSIZE];
  alignas(128) bf16 B[BK * BN * QSIZE];
};

template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE>
__global__ __launch_bounds__(NUM_THREADS) void matmulKernel4(
    int M, int N, int K, bf16 *C, const CUtensorMap *tensorMapA,
    const CUtensorMap *tensorMapB) {

  using PCSystem = wgmma_utils::ProducerConsumerSystem<bf16, BM, BN, BK, QSIZE>;
  using Buffer = typename PCSystem::Buffer;

  extern __shared__ __align__(128) uint8_t smem[];
  Buffer *buffer = reinterpret_cast<Buffer *>(smem);

  // Initialize buffer with the total number of threads that will synchronize
  Buffer::init(buffer, NUM_THREADS);

  constexpr int WGMMA_M = 64;
  constexpr int num_warp_groups = NUM_THREADS / 128;
  const int wg_idx = threadIdx.x / 128;
  const int tid = threadIdx.x % 128;

  if (wg_idx == 0) {
    // Producer thread
    typename PCSystem::ProducerState state{
        tensorMapA, tensorMapB, K / BK,
        blockIdx.x / (N / BN), // block_m
        blockIdx.x % (N / BN)  // block_n
    };

    PCSystem::run_producer(buffer, state);
  } else {
    // Consumer threads
    alignas(128) float output[BM / WGMMA_M][BN / 16][8] = {};

    PCSystem::run_consumer<WGMMA_M, BN>(buffer, output, K / BK);

    // Write results to global memory
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int row = warp * 16 + lane / 4;
    const int block_m = blockIdx.x / (N / BN);
    const int block_n = blockIdx.x % (N / BN);
    bf16 *block_C = C + block_n * BN * M + block_m * BM;

#pragma unroll
    for (int m_it = 0; m_it < BM / WGMMA_M; ++m_it) {
      const int yo = m_it * WGMMA_M;
#pragma unroll
      for (int w = 0; w < BN / 16; ++w) {
        const int col = 16 * w + 2 * (tid % 4);
#define IDX(i, j) ((j) * M + ((i) + yo))

        block_C[IDX(row, col)] = output[m_it][w][0];
        block_C[IDX(row, col + 1)] = output[m_it][w][1];
        block_C[IDX(row + 8, col)] = output[m_it][w][2];
        block_C[IDX(row + 8, col + 1)] = output[m_it][w][3];

        block_C[IDX(row, col + 8)] = output[m_it][w][4];
        block_C[IDX(row, col + 9)] = output[m_it][w][5];
        block_C[IDX(row + 8, col + 8)] = output[m_it][w][6];
        block_C[IDX(row + 8, col + 9)] = output[m_it][w][7];
#undef IDX
      }
    }
  }
}

// Helper function to create tensor maps and launch kernel
void runKernel4(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 128 * 2; // 2 warp groups
  constexpr int QSIZE = 5;             // Size of circular buffer

  // Create tensor maps for input matrices if needed
  static CUtensorMap *d_tma_map_A = nullptr;
  static CUtensorMap *d_tma_map_B = nullptr;
  static int prev_m = 0, prev_n = 0, prev_k = 0;

  if (!d_tma_map_A || M != prev_m || N != prev_n || K != prev_k) {
    if (d_tma_map_A) {
      cudaFree(d_tma_map_A);
      cudaFree(d_tma_map_B);
    }

    // Create tensor maps
    d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
    d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);

    prev_m = M;
    prev_n = N;
    prev_k = K;
  }

  // Calculate shared memory size
  const size_t smem_size =
      sizeof(wgmma_utils::CircularBuffer<bf16, BM, BN, BK, QSIZE>);

  // Set maximum shared memory size for the kernel
  cudaFuncSetAttribute(matmulKernel4<BM, BN, BK, NUM_THREADS, QSIZE>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  // Launch kernel
  const dim3 grid((M / BM) * (N / BN));
  matmulKernel4<BM, BN, BK, NUM_THREADS, QSIZE>
      <<<grid, NUM_THREADS, smem_size>>>(M, N, K, C, d_tma_map_A, d_tma_map_B);
}
} // namespace M4

using M4::runKernel4;
