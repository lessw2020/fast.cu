#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

namespace wgmma_utils {

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
} // namespace wgmma_utils
