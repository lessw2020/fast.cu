#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

typedef __nv_bfloat16 bf16;

// ============================================================================
// Helper Macros and Constants
// ============================================================================
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

namespace wgmma_utils {

// ======================
// Forward Declares
// ======================
class TMAOps;

class TMAOps {
public:
  __device__ static void load_async(bf16 *dst, void const *const src_tma_map,
                                    uint64_t *bar, int global_col_idx,
                                    int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier:"
                 ":complete_tx::bytes"
                 " [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(dst_ptr),
                 "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(global_row_idx),
                 "r"(global_col_idx / 64)
                 : "memory");
  }

  __device__ static void load_async_multicast(bf16 *dst,
                                              void const *const src_tma_map,
                                              uint64_t *bar, int global_col_idx,
                                              int global_row_idx,
                                              uint16_t cluster_mask) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier:"
                 ":complete_tx::bytes.multicast::cluster"
                 " [%0], [%1, {%3, %4, %5}], [%2], %6;" ::"r"(dst_ptr),
                 "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(global_row_idx),
                 "r"(global_col_idx / 64), "h"(cluster_mask)
                 : "memory");
  }

  __device__ static void store_async(void const *dst_tma_map, bf16 *src,
                                     int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
                 " [%0, {%2, %3, %4}], [%1];"
                 :
                 : "l"(tma_ptr), "r"(src_ptr), "n"(0), "r"(global_row_idx),
                   "r"(global_col_idx / 64)
                 : "memory");
  }
};

// ============================================================================
// Core Data Structures
// ============================================================================

/**
 * @brief Shared Memory layout for matrix operations
 * @tparam BM Block size M dimension
 * @tparam BN Block size N dimension
 * @tparam BK Block size K dimension
 * @tparam QSIZE Queue size for synchronization
 */
template <int BM, int BN, int BK, int QSIZE> struct SharedMemoryLayout {
  alignas(128) bf16 A[BM * BK * QSIZE]; // Matrix A data
  alignas(128) bf16 B[BK * BN * QSIZE]; // Matrix B data
  alignas(128) bf16 C[BN * BM];         // Output matrix C
  alignas(8) uint64_t full[QSIZE];      // Full flags for sync
  alignas(8) uint64_t empty[QSIZE];     // Empty flags for sync
};

// ============================================================================
// Memory Management Classes
// ============================================================================

// WGMMA Descriptor for shared memory
class WGMMADescriptor {
private:
  static constexpr uint64_t SWIZZLE_BITS_128 = 1llu << 62;
  static constexpr uint32_t MATRIX_ENCODE_MASK = 0x3FFFF;
  static constexpr uint32_t MATRIX_ENCODE_SHIFT = 0x4;

protected:
  __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return ((x & MATRIX_ENCODE_MASK) >> MATRIX_ENCODE_SHIFT);
  }

public:
  __device__ static uint64_t make_smem_desc(bf16 *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;

    static constexpr uint64_t STRIDE = 16;
    static constexpr uint64_t BLOCK_SIZE = 1024;

    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode(STRIDE) << 16;
    desc |= matrix_descriptor_encode(BLOCK_SIZE) << 32;
    desc |= SWIZZLE_BITS_128;

    return desc;
  }
};

// TMA Descriptor for global memory
class TMADescriptor {
public:
  template <typename Config>
  static CUtensorMap createDesc(typename Config::DataType *gmem_ptr,
                                int global_height, int global_width,
                                int block_height, int block_width) {
    CUtensorMap desc;
    // Move from static_assert to runtime assert
    assert(block_width >= 64);
    assert(global_width % 64 == 0);

    uint64_t gmem_shape[5] = {64, (uint64_t)global_height,
                              (uint64_t)global_width / 64, 1, 1};
    uint64_t gmem_stride[5] = {sizeof(typename Config::DataType) * global_width,
                               64 * sizeof(typename Config::DataType), 0, 0, 0};
    uint32_t smem_shape[5] = {64, uint32_t(block_height),
                              uint32_t(block_width / 64), 1, 1};
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gmem_ptr, gmem_shape,
        gmem_stride, smem_shape, smem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
    return desc;
  }
};

// ======
// Output
// ======
// Storage Operations for Global Memory
// WGMMA Output Layout Handler
struct WGMMAOutputLayout {
  static constexpr int WARP_SIZE = 32;
  static constexpr int OUTPUT_CHUNK = 16;

  int tid;
  int wg_idx;
  int lane;
  int warp;
  int base_row;

  __device__ WGMMAOutputLayout(int thread_idx, int warpgroup_idx)
      : tid(thread_idx), wg_idx(warpgroup_idx), lane(thread_idx % WARP_SIZE),
        warp(thread_idx / WARP_SIZE), base_row(warp * OUTPUT_CHUNK + lane / 4) {
  }

  __device__ __forceinline__ int get_col(int w) const {
    return w + 2 * (tid % 4);
  }
};

// WGMMA Output Storage Handler
template <typename T, int B_WG_M, int WGMMA_M, int WGMMA_N>
struct WGMMAOutputHandler {
  WGMMAOutputLayout layout;
  T *shared_out;

  __device__ WGMMAOutputHandler(T *sC, int thread_idx, int warpgroup_idx)
      : layout(thread_idx, warpgroup_idx),
        shared_out(sC + warpgroup_idx * B_WG_M * WGMMA_N) {}

  __device__ __forceinline__ void store_output(float d[][WGMMA_N / 16][8],
                                               int m_it) {
    const int yo = m_it * WGMMA_M;
    const int row = layout.base_row;

#pragma unroll
    for (int w = 0; w < WGMMA_N; w += layout.OUTPUT_CHUNK) {
      const int col = layout.get_col(w);
      const int w_idx = w / layout.OUTPUT_CHUNK;

      // Store output in 8x2 blocks
      auto store = [&](int r, int c, int v) {
        shared_out[c * B_WG_M + (r + yo)] = (d[m_it][w_idx][v]);
      };

      store(row, col, 0);
      store(row + 8, col, 2);
      store(row, col + 1, 1);
      store(row + 8, col + 1, 3);
      store(row, col + 8, 4);
      store(row + 8, col + 8, 6);
      store(row, col + 9, 5);
      store(row + 8, col + 9, 7);
    }
  }
};

// Global memory storage handler for WGMMA output
struct WGMMAGlobalStore {
  static __device__ __forceinline__ void wait_previous() {
    asm volatile("cp.async.bulk.wait_group 0;");
  }

  static __device__ __forceinline__ void sync_threads() {
    asm volatile("bar.sync 10, 256;\n");
  }

  static __device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
  }

  template <typename T>
  static __device__ __forceinline__ void
  store_global(const CUtensorMap *tensorMap, T *shared_data, int block_m,
               int block_n) {
    if (threadIdx.x == 128) {
      TMAOps::store_async(tensorMap, shared_data, block_m, block_n);
      asm volatile("cp.async.bulk.commit_group;");
    }
  }
};

// ============================================================================
// Synchronization and Control Classes
// ============================================================================

// Enhanced PTX-based barrier system
class PTXBarrier {
public:
  __device__ static void init_barrier(uint64_t *bar, int thread_count,
                                      int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
                 "r"(thread_count + transaction_count));
  }

  __device__ static void expect_bytes_tx(uint64_t *bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            bar_ptr),
        "r"(bytes));
  }

  __device__ static void wait(uint64_t *bar, int phase_bit) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra.uni DONE;\n"
                 "bra.uni LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(bar_ptr),
                 "r"(phase_bit));
  }

  __device__ static void arrive(uint64_t *bar, uint32_t count = 1) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            bar_ptr),
        "r"(count)
        : "memory");
  }

  __device__ static void wait_cluster(uint64_t *bar, int phase_bit) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, "
                 "[%0], %1;\n"
                 "@P1 bra.uni DONE;\n"
                 "bra.uni LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(bar_ptr),
                 "r"(phase_bit));
  }

  __device__ static void arrive_cluster(uint64_t *bar, uint32_t cta_id,
                                        uint32_t count = 1) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("{\n\t"
                 ".reg .b32 remAddr32;\n\t"
                 "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                 "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                 "}" ::"r"(smem_addr),
                 "r"(cta_id), "r"(count));
  }
};

// WGMMA synchronization operations
struct WGMMASyncOps {
  __device__ static void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  }

  __device__ static void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  }

  template <int N> __device__ static void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
  }

  template <int N = 0> __device__ static inline void wg_commit_and_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.commit_group.sync.aligned;\n"
                 "wgmma.wait_group.sync.aligned %0;\n" ::"n"(N)
                 : "memory");
  }
};

// TMA Operations
// Tensor map creation and management
class TensorMapManager {
public:
  template <int BlockMajorSize, int BlockMinorSize, bool swizzle = true>
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
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
    return tma_map;
  }
};

// Cluster utility functions

class ClusterOps {
public:
  // Get the current cluster ID
  __device__ static uint32_t get_cluster_id() {
    uint32_t cluster_id;
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(cluster_id));
    return cluster_id;
  }

  // Get the current cluster rank
  __device__ static uint32_t get_cluster_rank() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);
    return rank;
  }
  // Get cluster dimensions
  __device__ static void get_cluster_dims(uint32_t &dim_x, uint32_t &dim_y,
                                          uint32_t &dim_z) {
    asm volatile("mov.u32 %0, %cluster_dim.x;\n" : "=r"(dim_x));
    asm volatile("mov.u32 %0, %cluster_dim.y;\n" : "=r"(dim_y));
    asm volatile("mov.u32 %0, %cluster_dim.z;\n" : "=r"(dim_z));
  }

  // Synchronize all threads in the cluster
  __device__ static void cluster_sync() {
    asm volatile("barrier.cluster.arrive;\n" : : : "memory");
    asm volatile("barrier.cluster.wait;\n" : : : "memory");
  }

  // Only arrive at the barrier
  __device__ static void cluster_arrive() {
    asm volatile("barrier.cluster.arrive;\n" : : : "memory");
  }

  // Only wait at the barrier
  __device__ static void cluster_wait() {
    asm volatile("barrier.cluster.wait;\n" : : : "memory");
  }
};

// Register Management
class RegisterManager {
public:
  template <uint32_t RegCount> __device__ static void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
  }

  template <uint32_t RegCount> __device__ static void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
  }
};

// ====================
// Group GEMM
// ====================

// Descriptor Manager for batch operations
/*
// Descriptor Manager for batch operations
class BatchTMAManager {
public:
  struct BatchTMADescriptors {
    CUtensorMap tma_A;
    CUtensorMap tma_B;
    CUtensorMap tma_C;
  };

public:
  template <typename T>
  static BatchTMADescriptors createDescriptors(T *A, T *B, T *C, int M, int N,
                                               int K, int BM, int BN, int BK) {

    BatchTMADescriptors descs;
    descs.tma_A = TensorMapManager::create_tensor_map<BM, BK>(A, M, K);
    descs.tma_B = TensorMapManager::create_tensor_map<BN, BK>(B, N, K);
    descs.tma_C = TensorMapManager::create_tensor_map<BN, BM, false>(C, N, M);
    return descs;
  }
};
*/
//  orig
class BatchTMAManager {
public:
  struct BatchTMADescriptors {
    CUtensorMap tma_A;
    CUtensorMap tma_B;
    CUtensorMap tma_C;
  };

  template <typename T>
  static BatchTMADescriptors createDescriptors(T *A, T *B, T *C, int M, int N,
                                               int K, int BM, int BN, int BK) {

    BatchTMADescriptors descs;

    // Create TMA descriptors without template parameters
    CUtensorMap tma_map_A;
    {
      void *gmem_ptr = (void *)A;
      assert(BK >= 64);
      assert(K % 64 == 0);

      uint64_t gmem_shape[5] = {64, (uint64_t)M, (uint64_t)K / 64, 1, 1};
      uint64_t gmem_stride[5] = {sizeof(T) * K, 64 * sizeof(T), 0, 0, 0};
      uint32_t smem_shape[5] = {64, uint32_t(BM), uint32_t(BK / 64), 1, 1};
      uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

      CUresult result = cuTensorMapEncodeTiled(
          &tma_map_A,
          std::is_same<T, __nv_bfloat16>::value
              ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
              : CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
          3, gmem_ptr, gmem_shape, gmem_stride, smem_shape, smem_stride,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      assert(result == CUDA_SUCCESS);
    }
    descs.tma_A = tma_map_A;

    // Similar for B
    CUtensorMap tma_map_B;
    {
      void *gmem_ptr = (void *)B;
      assert(BK >= 64);
      assert(K % 64 == 0);

      uint64_t gmem_shape[5] = {64, (uint64_t)N, (uint64_t)K / 64, 1, 1};
      uint64_t gmem_stride[5] = {sizeof(T) * K, 64 * sizeof(T), 0, 0, 0};
      uint32_t smem_shape[5] = {64, uint32_t(BN), uint32_t(BK / 64), 1, 1};
      uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

      CUresult result = cuTensorMapEncodeTiled(
          &tma_map_B,
          std::is_same<T, __nv_bfloat16>::value
              ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
              : CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
          3, gmem_ptr, gmem_shape, gmem_stride, smem_shape, smem_stride,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      assert(result == CUDA_SUCCESS);
    }
    descs.tma_B = tma_map_B;

    // For C
    CUtensorMap tma_map_C;
    {
      void *gmem_ptr = (void *)C;
      assert(BN >= 64);
      assert(N % 64 == 0);

      uint64_t gmem_shape[5] = {64, (uint64_t)N, (uint64_t)M / 64, 1, 1};
      uint64_t gmem_stride[5] = {sizeof(T) * M, 64 * sizeof(T), 0, 0, 0};
      uint32_t smem_shape[5] = {64, uint32_t(BN), uint32_t(BM / 64), 1, 1};
      uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

      CUresult result = cuTensorMapEncodeTiled(
          &tma_map_C,
          std::is_same<T, __nv_bfloat16>::value
              ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
              : CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
          3, gmem_ptr, gmem_shape, gmem_stride, smem_shape, smem_stride,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_NONE, // No swizzle for output
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      assert(result == CUDA_SUCCESS);
    }
    descs.tma_C = tma_map_C;

    return descs;
  }
};
/*
// Helper class for managing batch parameters
class BatchParamsManager {
public:
  template <typename T> struct BatchParams {
    int M;
    int N;
    int K;
    T *A;
    T *B;
    T *C;
    float alpha;
    float beta;
  };

  template <typename T> struct BatchDescriptors {
    BatchTMAManager::BatchTMADescriptors tma_descs;
    BatchParams<T> params;
  };

  template <typename T>
  static BatchDescriptors<T>
  createBatchDescriptor(T *A, T *B, T *C, int M, int N, int K, int BM, int BN,
                        int BK, float alpha = 1.0f, float beta = 0.0f) {

    BatchDescriptors<T> desc;
    desc.params = {M, N, K, A, B, C, alpha, beta};
    desc.tma_descs =
        BatchTMAManager::createDescriptors<T>(A, B, C, M, N, K, BM, BN, BK);
    return desc;
  }
};
*/

// Helper class for managing batch parameters
class BatchParamsManager {
public:
  template <typename T> struct BatchParams {
    int M;
    int N;
    int K;
    T *A;
    T *B;
    T *C;
    float alpha;
    float beta;
  };

  template <typename T> struct BatchDescriptors {
    BatchTMAManager::BatchTMADescriptors tma_descs;
    BatchParams<T> params;
  };

  template <typename T>
  static BatchDescriptors<T>
  createBatchDescriptor(T *A, T *B, T *C, int M, int N, int K, int BM, int BN,
                        int BK, float alpha = 1.0f, float beta = 0.0f) {

    BatchDescriptors<T> desc;
    desc.params = {M, N, K, A, B, C, alpha, beta};
    desc.tma_descs =
        BatchTMAManager::createDescriptors<T>(A, B, C, M, N, K, BM, BN, BK);
    return desc;
  }
};

// Block scheduler for matrix operations
class BlockScheduler {
private:
  int block;
  int it;
  int total_blocks_m;
  int total_blocks_n;
  int blocks_per_tile_m;
  int blocks_per_tile_n;
  int num_sm;

public:
  __device__ __forceinline__ BlockScheduler(int M, int N,   // Matrix dimensions
                                            int BM, int BN, // Block dimensions
                                            int TM, int TN, // Tile dimensions
                                            int _num_sm,    // Number of SMs
                                            int _block) {   // Block ID

    block = _block;
    it = 0;
    num_sm = _num_sm;

    total_blocks_m = CEIL_DIV(M, BM);
    total_blocks_n = CEIL_DIV(N, BN);

    assert(CEIL_DIV(M, BM) % TM == 0 && total_blocks_n % TN == 0);

    blocks_per_tile_m = TM;
    blocks_per_tile_n = TN;
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    int num = it * num_sm + block;
    if (num >= total_blocks_m * total_blocks_n) {
      return false;
    }

    // Calculate current tile and position within tile
    int blocks_per_tile = blocks_per_tile_m * blocks_per_tile_n;
    int cur_tile = num / blocks_per_tile;
    int cur_tile_pos = num % blocks_per_tile;

    // Calculate block coordinates
    int tiles_per_row = total_blocks_n / blocks_per_tile_n;
    block_m = blocks_per_tile_m * (cur_tile / tiles_per_row);
    block_n = blocks_per_tile_n * (cur_tile % tiles_per_row);

    // Add offset within tile
    block_m += cur_tile_pos / blocks_per_tile_n;
    block_n += cur_tile_pos % blocks_per_tile_n;

    ++it;
    return true;
  }

  __device__ __forceinline__ static BlockScheduler
  create(int M, int N, int BM, int BN, int TM, int TN, int num_sm, int block) {
    return BlockScheduler(M, N, BM, BN, TM, TN, num_sm, block);
  }
};

// =========== WGMMA (Tensor Core) ASM routines ================

// ============================================================================
// WGMMA Core Operations
// ============================================================================

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma256(float d[16][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::make_smem_desc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::make_smem_desc(&sB[0]);
  asm volatile("{\n"
               "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
               " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
               " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
               " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
               " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
               " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
               " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
               " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
               " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
               " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
               " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
               " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
               " %104, %105, %106, %107, %108, %109, %110, %111,  "
               " %112, %113, %114, %115, %116, %117, %118, %119,  "
               " %120, %121, %122, %123, %124, %125, %126, %127},"
               " %128,"
               " %129,"
               " %130,    %131,  %132,  %133,  %134;\n"
               "}\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                 "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                 "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                 "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                 "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                 "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                 "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                 "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
                 "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
                 "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
                 "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
                 "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
                 "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
                 "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
                 "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
                 "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
                 "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
                 "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
                 "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
                 "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
                 "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
                 "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
                 "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
                 "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
                 "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]),
                 "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
                 "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
                 "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
                 "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]),
                 "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
                 "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]),
                 "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
                 "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
                 "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma192(float d[12][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::make_smem_desc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::make_smem_desc(&sB[0]);
  asm volatile("{\n"
               "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
               " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
               " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
               " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
               " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
               " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
               " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
               " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
               " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
               " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
               " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},  "
               " %96,"
               " %97,"
               " %98,    %99,  %100,  %101,  %102;\n"
               "}\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                 "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                 "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                 "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                 "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                 "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                 "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                 "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
                 "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
                 "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
                 "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
                 "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
                 "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
                 "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
                 "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
                 "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
                 "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]),
                 "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
                 "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]),
                 "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
                 "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
                 "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
                 "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
                 "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7])
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
                 "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
                 "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma128(float d[8][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::make_smem_desc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::make_smem_desc(&sB[0]);
  asm volatile("{\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
               " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
               " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
               " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
               " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
               " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
               " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
               " %64,"
               " %65,"
               " %66,    %67,  %68,  %69,  %70;\n"
               "}\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                 "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                 "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                 "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                 "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                 "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                 "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                 "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
                 "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
                 "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
                 "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]),
                 "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
                 "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
                 "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
                 "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
                 "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
                 "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
                 "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::make_smem_desc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::make_smem_desc(&sB[0]);
  asm volatile("{\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
               " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
               " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
               " %32,"
               " %33,"
               " %34, %35, %36, %37, %38;\n"
               "}\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                 "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                 "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                 "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
                 "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
                 "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
                 "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
                 "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
                 "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
                 "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma32(float d[2][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::make_smem_desc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::make_smem_desc(&sB[0]);
  asm volatile("{\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},  "
               " %16,"
               " %17,"
               " %18, %19, %20, %21, %22;\n"
               "}\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                 "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
                 "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
                 "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7])
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
                 "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
                 "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

} // namespace wgmma_utils
