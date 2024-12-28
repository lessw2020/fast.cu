#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace wgmma_utils {
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Forward declare dispatch function
template <int N>
__device__ void wgmma_dispatch(float d[N / 16][8], bf16 *sA, bf16 *sB);

// Configuration validation
template <typename Config> struct ValidateWGMMAConfig {
  static_assert(Config::ScaleD >= 0 && Config::ScaleD <= 4,
                "ScaleD must be in range [0,4]");
  static_assert(Config::ScaleA >= 0 && Config::ScaleA <= 4,
                "ScaleA must be in range [0,4]");
  static_assert(Config::ScaleB >= 0 && Config::ScaleB <= 4,
                "ScaleB must be in range [0,4]");
  static_assert(Config::TransformA == 0 || Config::TransformA == 1,
                "TransformA must be 0 or 1");
  static_assert(Config::TransformB == 0 || Config::TransformB == 1,
                "TransformB must be 0 or 1");
  static constexpr bool IsValid = true;
};

// Default configuration
struct DefaultConfig {
  static constexpr int ScaleD = 1;
  static constexpr int ScaleA = 1;
  static constexpr int ScaleB = 1;
  static constexpr int TransformA = 0;
  static constexpr int TransformB = 0;
  static_assert(ValidateWGMMAConfig<DefaultConfig>::IsValid,
                "Invalid DefaultConfig parameters");
};

// Parameter validation

template <typename T, int BM, int BN, int BK, int QSIZE>
struct ValidateParameters {
  static_assert(BM > 0 && BM % 64 == 0,
                "BM must be positive and aligned to 64");
  static_assert(BN > 0 && BN % 16 == 0,
                "BN must be positive and aligned to 16");
  static_assert(BK > 0 && BK % 16 == 0,
                "BK must be positive and aligned to 16");
  static_assert(QSIZE > 0 && QSIZE <= 8, "QSIZE must be between 1 and 8");
  static_assert(std::is_same_v<T, bf16>, "Only bf16 data type is supported");
};

// Forward declarations
template <typename Config>
__device__ __forceinline__ void wgmma256(float d[16][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma192(float d[12][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma128(float d[8][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma32(float d[2][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma16(float d[1][8], bf16 *sA, bf16 *sB);

class BarrierSystem {
private:
  barrier full;
  barrier empty;

public:
  __device__ BarrierSystem(unsigned int count) : full(count), empty(count) {}

  __device__ static void init(BarrierSystem *barriers, unsigned int count) {
    if (threadIdx.x == 0) {
      new (barriers) BarrierSystem(count);
      cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
  }

  __device__ void arrive_and_wait_empty() {
    auto arrival_token = empty.arrive();
    empty.wait(std::move(arrival_token));
  }

  __device__ void arrive_and_wait_full() {
    auto arrival_token = full.arrive();
    full.wait(std::move(arrival_token));
  }

  __device__ barrier::arrival_token arrive_full() { return full.arrive(); }

  __device__ barrier::arrival_token arrive_empty() { return empty.arrive(); }
};

template <typename T, int BM, int BN, int BK, int QSIZE> struct CircularBuffer {
  ValidateParameters<T, BM, BN, BK, QSIZE> validate;

  struct alignas(128) BufferEntry {
    T A[BM * BK];
    alignas(128) T B[BK * BN];
  };

  alignas(128) BufferEntry entries[QSIZE];
  alignas(128) BarrierSystem barriers[QSIZE];

  __device__ static void init(CircularBuffer *buffer,
                              unsigned int consumer_count) {
    if (threadIdx.x == 0) {
      for (int i = 0; i < QSIZE; ++i) {
        BarrierSystem::init(&buffer->barriers[i], consumer_count);
      }
      cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
  }

  __device__ void produce_begin(int &qidx, int block_k) {
    qidx = block_k % QSIZE;
    barriers[qidx].arrive_and_wait_empty();
  }

  // Use proper barrier_arrive_tx with both token parameters
  // cuda::device::barrier_arrive_tx(barriers[qidx].arrive_full(), token);
  __device__ void produce_end(int qidx, barrier::arrival_token token) {
    cuda::device::barrier_arrive_tx<cuda::thread_scope_block>(
        barriers[qidx].arrive_full(), std::move(token));
  }

  __device__ void consume_begin(int &qidx, int block_k) {
    qidx = block_k % QSIZE;
    barriers[qidx].arrive_and_wait_full();
  }

  __device__ void consume_end(int qidx) { barriers[qidx].arrive_empty(); }

  __device__ T *get_A(int qidx) { return entries[qidx].A; }
  __device__ T *get_B(int qidx) { return entries[qidx].B; }
};
} // namespace wgmma_utils

// Template specializations outside of the namespace
template <>
__device__ void wgmma_utils::wgmma_dispatch<256>(float d[16][8], bf16 *sA,
                                                 bf16 *sB) {
  wgmma256<DefaultConfig>(d, sA, sB);
}

template <>
__device__ void wgmma_utils::wgmma_dispatch<128>(float d[8][8], bf16 *sA,
                                                 bf16 *sB) {
  wgmma128<DefaultConfig>(d, sA, sB);
}

template <>
__device__ void wgmma_utils::wgmma_dispatch<64>(float d[4][8], bf16 *sA,
                                                bf16 *sB) {
  wgmma64<DefaultConfig>(d, sA, sB);
}

template <>
__device__ void wgmma_utils::wgmma_dispatch<32>(float d[2][8], bf16 *sA,
                                                bf16 *sB) {
  wgmma32<DefaultConfig>(d, sA, sB);
}

template <>
__device__ void wgmma_utils::wgmma_dispatch<16>(float d[1][8], bf16 *sA,
                                                bf16 *sB) {
  wgmma16<DefaultConfig>(d, sA, sB);
}

namespace wgmma_utils {
// Shared memory descriptor handling
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

// WGMMA Barrier Operations
struct SyncOps {
  __device__ static inline void wg_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  }

  __device__ static inline void wg_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  }

  template <int N> __device__ static inline void wg_wait_group() {
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

template <typename T, int BM, int BN, int BK, int QSIZE>
struct ProducerConsumerSystem {
  using Buffer = CircularBuffer<T, BM, BN, BK, QSIZE>;

  struct ProducerState {
    const CUtensorMap *tensorMapA;
    const CUtensorMap *tensorMapB;
    int num_blocks_k;
    int block_m;
    int block_n;
  };

  __device__ static void run_producer(Buffer *buffer,
                                      const ProducerState &state) {
    if (threadIdx.x == 0) {
      int qidx;
      for (int block_k = 0; block_k < state.num_blocks_k; ++block_k) {
        buffer->produce_begin(qidx, block_k);

        barrier::arrival_token token;
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            buffer->get_A(qidx), state.tensorMapA, block_k * BK,
            state.block_m * BM);

        cde::cp_async_bulk_tensor_2d_global_to_shared(
            buffer->get_B(qidx), state.tensorMapB, block_k * BK,
            state.block_n * BN);

        buffer->produce_end(qidx, token);
      }
    }
  }

  template <int WGMMA_M = 64, int WGMMA_N = BN, int WGMMA_K = 16>
  __device__ static void run_consumer(Buffer *buffer,
                                      float output[][WGMMA_N / 16][8],
                                      int num_blocks_k) {
    static_assert(WGMMA_M == 64, "WGMMA_M must be 64");
    static_assert(WGMMA_K == 16, "WGMMA_K must be 16");
    static_assert(WGMMA_N % 16 == 0, "WGMMA_N must be multiple of 16");

    int qidx;
    for (int block_k = 0; block_k < num_blocks_k; ++block_k) {
      buffer->consume_begin(qidx, block_k);

      SyncOps::wg_arrive();

#pragma unroll
      for (int m_it = 0; m_it < BM / WGMMA_M; ++m_it) {
        T *wgmma_sA = buffer->get_A(qidx) + BK * m_it * WGMMA_M;

#pragma unroll
        for (int k_it = 0; k_it < BK / WGMMA_K; ++k_it) {
          wgmma_dispatch<WGMMA_N>(output[m_it], &wgmma_sA[k_it * WGMMA_K],
                                  &buffer->get_B(qidx)[k_it * WGMMA_K]);
        }
      }

      SyncOps::wg_commit_group();
      SyncOps::wg_wait_group<0>();

      buffer->consume_end(qidx);
    }
  }
};

struct DefaultConfig {
  static constexpr int ScaleD = 1;
  static constexpr int ScaleA = 1;
  static constexpr int ScaleB = 1;
  static constexpr int TransformA = 0;
  static constexpr int TransformB = 0;

  // Force validation at compile time
  static_assert(ValidateWGMMAConfig<DefaultConfig>::IsValid,
                "Invalid DefaultConfig parameters");
};

// Type trait to validate WGMMA_N values with better error messages
template <int N> struct is_valid_wgmma_n : std::false_type {
  static_assert(N == 16 || N == 32 || N == 64 || N == 128 || N == 192 ||
                    N == 256,
                "WGMMA_N must be one of: 16, 32, 64, 128, 192, or 256");
};

// Forward declarations of all WGMMA functions
template <typename Config>
__device__ __forceinline__ void wgmma256(float d[16][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma192(float d[12][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma128(float d[8][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma32(float d[2][8], bf16 *sA, bf16 *sB);

template <typename Config>
__device__ __forceinline__ void wgmma16(float d[1][8], bf16 *sA, bf16 *sB);

template <> struct is_valid_wgmma_n<16> : std::true_type {};
template <> struct is_valid_wgmma_n<32> : std::true_type {};
template <> struct is_valid_wgmma_n<64> : std::true_type {};
template <> struct is_valid_wgmma_n<128> : std::true_type {};
template <> struct is_valid_wgmma_n<192> : std::true_type {};
template <> struct is_valid_wgmma_n<256> : std::true_type {};

// Main dispatch function updated to include wgmma16
template <int WGMMA_N, typename Config = DefaultConfig>
__device__ __forceinline__ void wgmma_dispatch(float d[WGMMA_N / 16][8],
                                               bf16 *sA, bf16 *sB) {
  static_assert(is_valid_wgmma_n<WGMMA_N>::value, "Invalid WGMMA_N value");
  static_assert(ValidateWGMMAConfig<Config>::IsValid,
                "Invalid WGMMA configuration");

  if constexpr (WGMMA_N == 256) {
    wgmma256<Config>(d, sA, sB);
  } else if constexpr (WGMMA_N == 192) {
    wgmma192<Config>(d, sA, sB);
  } else if constexpr (WGMMA_N == 128) {
    wgmma128<Config>(d, sA, sB);
  } else if constexpr (WGMMA_N == 64) {
    wgmma64<Config>(d, sA, sB);
  } else if constexpr (WGMMA_N == 32) {
    wgmma32<Config>(d, sA, sB);
  } else if constexpr (WGMMA_N == 16) {
    wgmma16<Config>(d, sA, sB);
  }
}
// suite of wgmma ptx calls

// template <int ScaleD, int ScaleA, int ScaleB, int TransformA, int TransformB>
template <typename Config = DefaultConfig>
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
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(Config::ScaleD)),
                 "n"(int32_t(Config::ScaleA)), "n"(int32_t(Config::ScaleB)),
                 "n"(int32_t(Config::TransformA)),
                 "n"(int32_t(Config::TransformB)));

  //: "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
  //  "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
  //  "n"(int32_t(TransformA)), "n"(int32_t(TransformB)));
}

template <typename Config = DefaultConfig>
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
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(Config::ScaleD)),
                 "n"(int32_t(Config::ScaleA)), "n"(int32_t(Config::ScaleB)),
                 "n"(int32_t(Config::TransformA)),
                 "n"(int32_t(Config::TransformB)));
}

template <typename Config = DefaultConfig>
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
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(Config::ScaleD)),
                 "n"(int32_t(Config::ScaleA)), "n"(int32_t(Config::ScaleB)),
                 "n"(int32_t(Config::TransformA)),
                 "n"(int32_t(Config::TransformB)));
}

template <typename Config = DefaultConfig>
__device__ void wgmma64(float d[4][8], bf16 *sA, bf16 *sB) {
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
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(Config::ScaleD)),
                 "n"(int32_t(Config::ScaleA)), "n"(int32_t(Config::ScaleB)),
                 "n"(int32_t(Config::TransformA)),
                 "n"(int32_t(Config::TransformB)));
}

template <typename Config = DefaultConfig>
__device__ void wgmma32(float d[2][8], bf16 *sA, bf16 *sB) {
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
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(Config::ScaleD)),
                 "n"(int32_t(Config::ScaleA)), "n"(int32_t(Config::ScaleB)),
                 "n"(int32_t(Config::TransformA)),
                 "n"(int32_t(Config::TransformB)));
}

template <typename Config = DefaultConfig>
__device__ void wgmma16(float d[1][8], bf16 *sA, bf16 *sB) {
  uint64_t desc_a = WGMMADescriptor::make_smem_desc(&sA[0]);
  uint64_t desc_b = WGMMADescriptor::make_smem_desc(&sB[0]);
  asm volatile("{\n"
               "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
               " %8,"
               " %9,"
               " %10, %11, %12, %13, %14;\n"
               "}\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                 "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7])
               : "l"(desc_a), "l"(desc_b), "n"(int32_t(Config::ScaleD)),
                 "n"(int32_t(Config::ScaleA)), "n"(int32_t(Config::ScaleB)),
                 "n"(int32_t(Config::TransformA)),
                 "n"(int32_t(Config::TransformB)));
}

template <>
__device__ void wgmma_dispatch<256>(float d[16][8], bf16 *sA, bf16 *sB) {
  wgmma256<DefaultConfig>(d, sA, sB);
}

template <>
__device__ void wgmma_dispatch<128>(float d[8][8], bf16 *sA, bf16 *sB) {
  wgmma128<DefaultConfig>(d, sA, sB);
}

template <>
__device__ void wgmma_dispatch<64>(float d[4][8], bf16 *sA, bf16 *sB) {
  wgmma64<DefaultConfig>(d, sA, sB);
}

template <>
__device__ void wgmma_dispatch<32>(float d[2][8], bf16 *sA, bf16 *sB) {
  wgmma32<DefaultConfig>(d, sA, sB);
}

template <>
__device__ void wgmma_dispatch<16>(float d[1][8], bf16 *sA, bf16 *sB) {
  wgmma16<DefaultConfig>(d, sA, sB);
}

} // namespace wgmma_utils
