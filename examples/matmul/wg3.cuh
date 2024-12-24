namespace wgmma {

// Unified configuration structures
struct ScaleConfig {
  int D, A, B;
  constexpr ScaleConfig(int d = 1, int a = 1, int b = 1) : D(d), A(a), B(b) {}
};

struct TransposeConfig {
  int A, B;
  constexpr TransposeConfig(int a = 0, int b = 0) : A(a), B(b) {}
};

// Single descriptor class
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
    uint64_t desc = 0;

    // Encode parameters with named constants for clarity
    static constexpr uint64_t STRIDE = 16;
    static constexpr uint64_t BLOCK_SIZE = 1024;

    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode(STRIDE) << 16;
    desc |= matrix_descriptor_encode(BLOCK_SIZE) << 32;
    desc |= SWIZZLE_BITS_128;

    return desc;
  }
};

// WGMMA operation wrapper class
template <int M, int N, int K> class WGMMAOp {
  static_assert(K == 16, "Only K=16 is supported currently");
  static_assert(M == 64, "Only M=64 is supported currently");
  static_assert(N >= 16 && N <= 256 && (N & (N - 1)) == 0,
                "N must be power of 2 between 16 and 256");

public:
  template <typename ScaleConfig, typename TransposeConfig>
  __device__ static void
  multiply(float *d, bf16 *sA, bf16 *sB,
           const ScaleConfig &scale = ScaleConfig{},
           const TransposeConfig &trans = TransposeConfig{}) {
    uint64_t desc_a = WGMMADescriptor::make_smem_desc(sA);
    uint64_t desc_b = WGMMADescriptor::make_smem_desc(sB);

    if constexpr (N == 256) {
      // Existing wgmma256 implementation
    } else if constexpr (N == 192) {
      // Existing wgmma192 implementation
    }
    // ... other sizes
  }
};

// Error checking wrapper
template <typename T> struct WGMMAErrorChecker {
  static constexpr bool is_supported_type =
      std::is_same_v<T, bf16> || std::is_same_v<T, float>;

  static_assert(is_supported_type,
                "WGMMA only supports bf16 input and float output");
};

} // namespace wgmma
