
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

namespace wgmma_utils {

// Base kernel configuration
template <typename T, int BM, int BN, int BK> struct KernelConfig {
  using DataType = T;
  static constexpr int BlockM = BM;
  static constexpr int BlockN = BN;
  static constexpr int BlockK = BK;

  static_assert(BM > 0 && BM % 64 == 0,
                "BM must be positive and aligned to 64");
  static_assert(BN > 0 && BN % 16 == 0,
                "BN must be positive and aligned to 16");
  static_assert(BK > 0 && BK % 16 == 0,
                "BK must be positive and aligned to 16");
  static_assert(std::is_same_v<T, bf16>, "Only bf16 data type is supported");
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
// TMA Descriptor Cache
template <typename Config> class TMACache {
private:
  static CUtensorMap tmaMapA;
  static CUtensorMap tmaMapB;
  static int prevM, prevN, prevK;

public:
  static void initializeMaps(int M, int N, int K, bf16 *A, bf16 *B) {
    if (M != prevM || N != prevN || K != prevK) {
      tmaMapA = TMADescriptor::createDesc<Config>(A, M, K, Config::BlockM,
                                                  Config::BlockK);
      tmaMapB = TMADescriptor::createDesc<Config>(B, N, K, Config::BlockN,
                                                  Config::BlockK);
      prevM = M;
      prevN = N;
      prevK = K;
    }
    assert(M == prevM && N == prevN && K == prevK);
  }

  static const CUtensorMap &getMapA() { return tmaMapA; }
  static const CUtensorMap &getMapB() { return tmaMapB; }
};

template <typename Config> CUtensorMap TMACache<Config>::tmaMapA;
template <typename Config> CUtensorMap TMACache<Config>::tmaMapB;
template <typename Config> int TMACache<Config>::prevM = 0;
template <typename Config> int TMACache<Config>::prevN = 0;
template <typename Config> int TMACache<Config>::prevK = 0;

// PTX Barrier System
class PTXBarrier {
public:
  __device__ static void init_barrier(uint64_t *bar, int thread_count,
                                      int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
                 "r"(thread_count + transaction_count));
  }

  __device__ static void expect_tx(uint64_t *bar, uint32_t bytes) {
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
  __device__ static void arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  }

  __device__ static void commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  }

  template <int N> __device__ static void wait_group() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
  }
};

// Register Management
class RegisterManager {
public:
  template <uint32_t RegCount> __device__ static void alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
  }

  template <uint32_t RegCount> __device__ static void dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
  }
};

// Shared Memory Layout
template <typename Config> struct SharedMemoryLayout {
  struct Buffer {
    alignas(128) typename Config::DataType A[Config::BlockM * Config::BlockK];
    alignas(128) typename Config::DataType B[Config::BlockK * Config::BlockN];
  };

  template <int QueueSize> struct QueuedBuffer {
    alignas(128) typename Config::DataType
        A[Config::BlockM * Config::BlockK * QueueSize];
    alignas(128) typename Config::DataType
        B[Config::BlockK * Config::BlockN * QueueSize];
  };
};

// WGMMA Descriptor for shared memory
class WGMMADescriptor {
private:
  static constexpr uint64_t SWIZZLE_BITS_128 = 1llu << 62;
  static constexpr uint32_t MATRIX_ENCODE_MASK = 0x3FFFF;
  static constexpr uint32_t MATRIX_ENCODE_SHIFT = 0x4;

public:
  __device__ static uint64_t makeSharedDesc(bf16 *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    desc |= ((addr & MATRIX_ENCODE_MASK) >> MATRIX_ENCODE_SHIFT);
    desc |= ((uint64_t)16 & MATRIX_ENCODE_MASK) << 16;
    desc |= ((uint64_t)1024 & MATRIX_ENCODE_MASK) << 32;
    desc |= SWIZZLE_BITS_128;
    return desc;
  }
};

// TMA Operations
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
};

// Schedule template
template <int Version, int NumSM, int BM, int BN, int TM, int TN>
struct Schedule {
  int block;
  int it;
  int total_blocks_m, total_blocks_n;

  __device__ __forceinline__ Schedule(int M, int N, int _block) {
    block = _block;
    it = 0;
    total_blocks_m = CEIL_DIV(M, BM);
    total_blocks_n = CEIL_DIV(N, BN);
    assert(total_blocks_m % TM == 0 && total_blocks_n % TN == 0);
  }

  __device__ __forceinline__ bool next(int &block_m, int &block_n) {
    int num = it * NumSM + block;
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

// Cluster Management
struct ClusterInfo {
  uint32_t cluster_id;
  uint32_t cluster_m;
  uint32_t cluster_n;
  uint32_t cta_rank;
  uint32_t rank_m;
  uint32_t rank_n;

  __device__ static ClusterInfo get() {
    ClusterInfo info;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(info.cta_rank));
    asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(info.cluster_id));

    // Derive cluster dimensions from compilation constants
    info.cluster_m = blockDim.y; // Assuming blockDim.y is cluster_m
    info.cluster_n = blockDim.z; // Assuming blockDim.z is cluster_n

    // Calculate rank position within cluster
    info.rank_m = info.cta_rank / info.cluster_n;
    info.rank_n = info.cta_rank % info.cluster_n;

    return info;
  }

  __device__ static void syncCluster() {
    asm volatile("barrier.cluster.arrive;\n" : :);
    asm volatile("barrier.cluster.wait;\n" : :);
  }
};

// WGMMA Operations
// Base template for WGMMA operations
template <int N, int ScaleD, int ScaleA, int ScaleB, int TransformA,
          int TransformB>
__device__ __forceinline__ void wgmma_impl(float (*d)[8], bf16 *sA, bf16 *sB);

// Specialization for N=256
template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void
wgmma_impl<256, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(float (*d)[8],
                                                                bf16 *sA,
                                                                bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(sA);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(sB);

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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

// Specialization for N=192
template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void
wgmma_impl<192, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(float (*d)[8],
                                                                bf16 *sA,
                                                                bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(sA);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(sB);

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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

// Specialization for N=128
template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void
wgmma_impl<128, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(float (*d)[8],
                                                                bf16 *sA,
                                                                bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(sA);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(sB);

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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

// Specialization for N=64
template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void
wgmma_impl<64, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(float (*d)[8],
                                                               bf16 *sA,
                                                               bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(sA);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(sB);

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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

// Specialization for N=32
template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void
wgmma_impl<32, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(float (*d)[8],
                                                               bf16 *sA,
                                                               bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(sA);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(sB);

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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void wgmma256(float d[16][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(&sB[0]);
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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void wgmma192(float d[12][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(&sB[0]);
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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void wgmma128(float d[8][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(&sB[0]);
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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(&sB[0]);
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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
__device__ __forceinline__ void wgmma32(float d[2][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(&sB[0]);
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
                 "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

template <int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransformA,
          int TransformB>
__device__ __forceinline__ void wgmma(float d[WGMMA_N / 16][8], bf16 *sA,
                                      bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::makeSharedDesc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::makeSharedDesc(&sB[0]);

  if constexpr (WGMMA_N == 256) {
    asm volatile(
        "{\n"
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
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransformA)),
          "n"(int32_t(TransformB)));
  } else if constexpr (WGMMA_N == 192) {
    asm volatile(
        "{\n"
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
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransformA)),
          "n"(int32_t(TransformB)));
  } else if constexpr (WGMMA_N == 128) {
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
                   "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
  } else if constexpr (WGMMA_N == 64) {
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
                   "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
  } else if constexpr (WGMMA_N == 32) {
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
                   "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
  }

} // end wgmma

// Generic WGMMA dispatch

// Main dispatch function
template <int Size, int ScaleD = 1, int ScaleA = 1, int ScaleB = 1,
          bool TransformA = false, bool TransformB = false>
__device__ __forceinline__ void wgmma_dispatch(void *d, bf16 *sA, bf16 *sB) {
  static_assert(Size == 32 || Size == 64 || Size == 128 || Size == 192 ||
                    Size == 256,
                "Invalid WGMMA size");

  using ArrayType = float[Size / 16][8];
  float(*d_array)[8] = reinterpret_cast<float(*)[8]>(d);
  wgmma_impl<Size, ScaleD, ScaleA, ScaleB, TransformA ? 1 : 0,
             TransformB ? 1 : 0>(d_array, sA, sB);
}

// Convenience wrappers for direct size-specific calls
template <int ScaleD = 1, int ScaleA = 1, int ScaleB = 1,
          bool TransformA = false, bool TransformB = false>
__device__ __forceinline__ void wgmma256(float d[16][8], bf16 *sA, bf16 *sB) {
  wgmma_dispatch<256, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(d, sA,
                                                                      sB);
}

template <int ScaleD = 1, int ScaleA = 1, int ScaleB = 1,
          bool TransformA = false, bool TransformB = false>
__device__ __forceinline__ void wgmma192(float d[12][8], bf16 *sA, bf16 *sB) {
  wgmma_dispatch<192, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(d, sA,
                                                                      sB);
}

template <int ScaleD = 1, int ScaleA = 1, int ScaleB = 1,
          bool TransformA = false, bool TransformB = false>
__device__ __forceinline__ void wgmma128(float d[8][8], bf16 *sA, bf16 *sB) {
  wgmma_dispatch<128, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(d, sA,
                                                                      sB);
}

template <int ScaleD = 1, int ScaleA = 1, int ScaleB = 1,
          bool TransformA = false, bool TransformB = false>
__device__ __forceinline__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB) {
  wgmma_dispatch<64, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(d, sA, sB);
}

template <int ScaleD = 1, int ScaleA = 1, int ScaleB = 1,
          bool TransformA = false, bool TransformB = false>
__device__ __forceinline__ void wgmma32(float d[2][8], bf16 *sA, bf16 *sB) {
  wgmma_dispatch<32, ScaleD, ScaleA, ScaleB, TransformA, TransformB>(d, sA, sB);
}

} // namespace wgmma_utils
