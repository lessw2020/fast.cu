// group_matmul.cuh
#pragma once

#include <cassert>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

namespace GroupM10 {

using bf16 = __nv_bfloat16;

struct GEMMParams {
  int M;
  int N;
  int K;
  bf16 *A;
  bf16 *B;
  bf16 *C;

  bool isValid() const {
    return M > 0 && N > 0 && K > 0 && A != nullptr && B != nullptr &&
           C != nullptr && M % 128 == 0 && N % 256 == 0 && K % 64 == 0;
  }
};

void runGroupMatmul(const std::vector<GEMMParams> &params);

} // namespace GroupM10
