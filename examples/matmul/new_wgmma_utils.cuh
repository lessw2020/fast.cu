#pragma once
// #include "register_manager.hpp"
#include <cuda_bf16.h>

namespace cuda {
namespace wgmma {

// WGMMA operation modes
enum class AccumulateMode {
  Init,      // First multiplication in sequence (d = a * b)
  Accumulate // Subsequent multiplications (d += a * b)
};

class WGMMABase {
protected:
  static constexpr uint64_t SWIZZLE_BITS_128 = 1llu << 62;

  __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
  }

  __device__ uint64_t make_smem_desc(bf16 *ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= SWIZZLE_BITS_128; // 128B swizzle
    return desc;
  }

public:
  __device__ static void fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  }

  __device__ static void commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  }

  __device__ static void wait_batch(int N = 0) {
    // static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
  }
};

// Main WGMMA class with extended functionality
template <int M, int N, int K> class WGMMA : public WGMMABase {
  static_assert(M == 64, "Currently only M=64 is supported");
  static_assert(K == 16, "Currently only K=16 is supported");
  static_assert(N == 16 || N == 32 || N == 64 || N == 128 || N == 192 ||
                    N == 256,
                "N must be one of: 16, 32, 64, 128, 192, 256");

public:
  static constexpr int NumRows = N / 16;
  static constexpr int NumCols = 8;

  // Standard multiply with optional accumulation mode
  template <int ScaleD = 1, int ScaleA = 1, int ScaleB = 1, int TransA = 0,
            int TransB = 0>
  __device__ void multiply(float d[NumRows][NumCols], bf16 *sA, bf16 *sB,
                           AccumulateMode mode = AccumulateMode::Accumulate) {
    // Adjust ScaleD based on accumulation mode
    constexpr int effectiveScaleD = (mode == AccumulateMode::Init) ? 0 : ScaleD;

    uint64_t desc_a = make_smem_desc(sA);
    uint64_t desc_b = make_smem_desc(sB);

    if constexpr (N == 256) {
      multiply_256<effectiveScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a,
                                                                    desc_b);
    } else if constexpr (N == 192) {
      multiply_192<effectiveScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a,
                                                                    desc_b);
    } else if constexpr (N == 128) {
      multiply_128<effectiveScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a,
                                                                    desc_b);
    } else if constexpr (N == 64) {
      multiply_64<effectiveScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a,
                                                                   desc_b);
    } else if constexpr (N == 32) {
      multiply_32<effectiveScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a,
                                                                   desc_b);
    } else if constexpr (N == 16) {
      multiply_16<effectiveScaleD, ScaleA, ScaleB, TransA, TransB>(d, desc_a,
                                                                   desc_b);
    }
  }

  // Batch multiply - execute multiple multiplications in sequence
  template <int ScaleD = 1, int ScaleA = 1, int ScaleB = 1, int TransA = 0,
            int TransB = 0>
  __device__ void multiply_batch(float d[NumRows][NumCols], bf16 *sA, bf16 *sB,
                                 int batch_size) {
    fence();

    // First multiplication initializes
    multiply<0, ScaleA, ScaleB, TransA, TransB>(d, &sA[0], &sB[0],
                                                AccumulateMode::Init);

    // Subsequent multiplications accumulate
    for (int i = 1; i < batch_size; i++) {
      multiply<1, ScaleA, ScaleB, TransA, TransB>(
          d, &sA[i * M * K], &sB[i * K * N], AccumulateMode::Accumulate);
    }

    commit_batch();
    wait_batch<0>();
  }

private:
  // Implementation functions for different sizes
  template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
  __device__ void multiply_256(float d[16][8], uint64_t desc_a,
                               uint64_t desc_b) {
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
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
  }

  template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
  __device__ void multiply_192(float d[12][8], uint64_t desc_a,
                               uint64_t desc_b) {
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
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
  }

  template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
  __device__ void multiply_128(float d[8][8], uint64_t desc_a,
                               uint64_t desc_b) {
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
  __device__ void multiply_64(float d[4][8], uint64_t desc_a, uint64_t desc_b) {
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
  __device__ void multiply_32(float d[2][8], uint64_t desc_a, uint64_t desc_b) {
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

  template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
  __device__ void multiply_16(float d[1][8], uint64_t desc_a, uint64_t desc_b) {
    asm volatile("{\n"
                 "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
                 "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
                 " %8,"
                 " %9,"
                 " %10, %11, %12, %13, %14;\n"
                 "}\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
                   "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7])
                 : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)),
                   "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)),
                   "n"(int32_t(TransA)), "n"(int32_t(TransB)));
  }
};

} // namespace wgmma
} // namespace cuda
