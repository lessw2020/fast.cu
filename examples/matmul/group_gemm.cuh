// group_matmul.cu
#include "group_matmul.cuh"
#include "matmul_10.cuh"
#include <stdexcept>

namespace GroupM10 {

using namespace M10;
using namespace wgmma_utils;

// Helper macro for CUDA error checking
#define cudaCheck(err)                                                         \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess)                                                   \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err_) + " at " + __FILE__ +  \
                               ":" + std::to_string(__LINE__));                \
  }

class TMADescriptorGroup {
private:
  std::vector<CUtensorMap> maps_A;
  std::vector<CUtensorMap> maps_B;
  std::vector<CUtensorMap> maps_C;
  std::vector<GEMMParams> prev_params;

public:
  void ensureCapacity(size_t size) {
    if (maps_A.size() < size) {
      maps_A.resize(size);
      maps_B.resize(size);
      maps_C.resize(size);
      prev_params.resize(size);
    }
  }

  void updateMaps(const std::vector<GEMMParams> &params) {
    ensureCapacity(params.size());

    for (size_t i = 0; i < params.size(); i++) {
      if (!params[i].isValid()) {
        throw std::runtime_error("Invalid GEMM parameters");
      }

      if (prev_params[i].M != params[i].M || prev_params[i].N != params[i].N ||
          prev_params[i].K != params[i].K || prev_params[i].A != params[i].A ||
          prev_params[i].B != params[i].B || prev_params[i].C != params[i].C) {

        maps_A[i] = TensorMapManager::create_tensor_map<128, 64>(
            params[i].A, params[i].M, params[i].K);
        maps_B[i] = TensorMapManager::create_tensor_map<256, 64>(
            params[i].B, params[i].N, params[i].K);
        maps_C[i] = TensorMapManager::create_tensor_map<256, 128, false>(
            params[i].C, params[i].N, params[i].M);

        prev_params[i] = params[i];
      }
    }
  }

  const CUtensorMap *A_maps() const { return maps_A.data(); }
  const CUtensorMap *B_maps() const { return maps_B.data(); }
  const CUtensorMap *C_maps() const { return maps_C.data(); }
};

static TMADescriptorGroup tma_descriptors;

template <int BM, int BN, int BK, int NUM_THREADS, int QSIZE, int NUM_SM,
          int CLUSTER_M, int CLUSTER_N>
__global__
__launch_bounds__(NUM_THREADS) void __cluster_dims__(CLUSTER_M *CLUSTER_N, 1, 1)
    groupMatmulKernel(const int group_size,
                      const __grid_constant__ CUtensorMap *tensorMapC,
                      const __grid_constant__ CUtensorMap *tensorMapA,
                      const __grid_constant__ CUtensorMap *tensorMapB,
                      const int *M_array, const int *N_array,
                      const int *K_array) {
  // Calculate which group this block is processing
  int group_idx = blockIdx.x / (NUM_SM / (CLUSTER_M * CLUSTER_N));
  int local_block_idx = blockIdx.x % (NUM_SM / (CLUSTER_M * CLUSTER_N));

  // Get dimensions for this group
  int M = M_array[group_idx];
  int N = N_array[group_idx];
  int K = K_array[group_idx];
  const int num_blocks_k = K / BK;

  extern __shared__ __align__(128) uint8_t smem[];
  SharedMemoryLayout<BM, BN, BK, QSIZE> &s =
      *reinterpret_cast<SharedMemoryLayout<BM, BN, BK, QSIZE> *>(smem);

  bf16 *sA = s.A;
  bf16 *sB = s.B;
  bf16 *sC = s.C;
  uint64_t *full = s.full;
  uint64_t *empty = s.empty;

  constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N = BN;
  constexpr int num_consumers = (NUM_THREADS / 128) - 1;
  constexpr int B_WG_M = BM / num_consumers;
  constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

  if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
      PTXBarrier::init_barrier(&full[i], 0, 1);
      PTXBarrier::init_barrier(&empty[i], 0, num_consumers * CLUSTERS);
    }
  }

  ClusterOps::cluster_sync();

  uint32_t cluster_id = ClusterOps::get_cluster_id();
  uint32_t cluster_rank = ClusterOps::get_cluster_rank();
  uint32_t rank_m = cluster_rank / CLUSTER_N;
  uint32_t rank_n = cluster_rank % CLUSTER_N;

  Schedule<1, NUM_SM / CLUSTERS, BM * CLUSTER_M, BN * CLUSTER_N, 16 / CLUSTER_M,
           8 / CLUSTER_N>
      schedule(M, N, local_block_idx);

  int wg_idx = threadIdx.x / 128;
  int tid = threadIdx.x % 128;

  // Producer thread
  if (wg_idx == 0) {
    constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
    RegisterManager::warpgroup_reg_dealloc<num_regs>();

    if (tid == 0) {
      int p = 0;
      int qidx = 0;
      uint32_t col_mask = 0;
      for (int i = 0; i < CLUSTER_M; ++i) {
        col_mask |= (1 << (i * CLUSTER_N));
      }

      int num_block_m, num_block_n;
      while (schedule.next(num_block_m, num_block_n)) {
        num_block_n = num_block_n * CLUSTER_N + rank_n;
        num_block_m = num_block_m * CLUSTER_M + rank_m;

        for (int block_k_iter = 0; block_k_iter < num_blocks_k;
             ++block_k_iter, ++qidx) {
          if (qidx == QSIZE) {
            qidx = 0;
            p ^= 1;
          }
          PTXBarrier::wait(&empty[qidx], p);

          PTXBarrier::expect_bytes_tx(&full[qidx],
                                      (BK * BN + BK * BM) * sizeof(bf16));

          if constexpr (CLUSTER_N > 1) {
            uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
            if (rank_n == 0) {
              TMAOps::load_async_multicast(
                  &sA[qidx * BK * BM], &tensorMapA[group_idx], &full[qidx],
                  block_k_iter * BK, num_block_m * BM, mask);
            }
          } else {
            TMAOps::load_async(&sA[qidx * BK * BM], &tensorMapA[group_idx],
                               &full[qidx], block_k_iter * BK,
                               num_block_m * BM);
          }

          if constexpr (CLUSTER_M > 1) {
            if (rank_m == 0) {
              TMAOps::load_async_multicast(
                  &sB[qidx * BK * BN], &tensorMapB[group_idx], &full[qidx],
                  block_k_iter * BK, num_block_n * BN, col_mask << rank_n);
            }
          } else {
            TMAOps::load_async(&sB[qidx * BK * BN], &tensorMapB[group_idx],
                               &full[qidx], block_k_iter * BK,
                               num_block_n * BN);
          }
        }
      }
    }
  } else {
    // Consumer threads
    constexpr int num_regs =
        (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
    RegisterManager::warpgroup_reg_alloc<num_regs>();
    float d[B_WG_M / WGMMA_M][WGMMA_N / 16][8];
    --wg_idx;

    for (int qidx = 0; qidx < QSIZE; ++qidx) {
      if (tid < CLUSTERS)
        PTXBarrier::arrive_cluster(&empty[qidx], tid);
    }

    WGMMAOutputHandler<bf16, B_WG_M, WGMMA_M, WGMMA_N> output_handler(
        sC, threadIdx.x, wg_idx);

    int p = 0;
    int qidx = 0;
    int num_block_m, num_block_n;

    while (schedule.next(num_block_m, num_block_n)) {
      num_block_n = num_block_n * CLUSTER_N + rank_n;
      num_block_m = num_block_m * CLUSTER_M + rank_m;

      {
        if (qidx == QSIZE) {
          qidx = 0;
          p ^= 1;
        };
        PTXBarrier::wait(&full[qidx], p);
        WGMMASyncOps::warpgroup_arrive();

        // Compute for each sub-block
        for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
          bf16 *wgmma_sA = sA + qidx * BK * BM +
                           64 * (m_it + wg_idx * B_WG_M / WGMMA_M) * WGMMA_M;
          bf16 *wgmma_sB = sB + qidx * BK * BN;
          {
            // Initial WGMMA compute
            wgmma<WGMMA_N, 0, 1, 1, 0, 0>(d[m_it], &wgmma_sA[0], &wgmma_sB[0]);

            // Remaining iterations within 64 elem boundary
            for (int k_it = 1; k_it < 64 / WGMMA_K; ++k_it) {
              wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
                                            &wgmma_sB[k_it * WGMMA_K]);
            }
            wgmma_sA += 64 * BM;
            wgmma_sB += 64 * BN;
          }

          // Process remaining blocks
          for (int bk = 64; bk < BK; bk += 64) {
            for (int k_it = 0; k_it < 64 / WGMMA_K; ++k_it) {
              wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it * WGMMA_K],
                                            &wgmma_sB[k_it * WGMMA_K]);
            }
            wgmma_sA += 64 * BM;
            wgmma_sB += 64 * BN;
          }
        }
        WGMMASyncOps::warpgroup_commit_batch();
        WGMMASyncOps::warpgroup_wait<0>();
        if (tid < CLUSTERS)
          PTXBarrier::arrive_cluster(&empty[qidx], tid);
        ++qidx;
      }

      // Store results
      WGMMAGlobalStore::wait_previous();

      int lane = tid % 32, warp = tid / 32;
      int row = warp * 16 + lane / 4;

      bf16 *block_sC = sC + wg_idx * B_WG_M * BN;
      for (int m_it = 0; m_it < B_WG_M / WGMMA_M; ++m_it) {
        int yo = m_it * WGMMA_M;
        for (int w = 0; w < WGMMA_N; w += 16) {
          int col = w + 2 * (tid % 4);
#define ST(i, j, v) block_sC[(j) * B_WG_M + (i) + yo] = v

          ST(row, col, d[m_it][w / 16][0]);
          /////////////////
          ST(row + 8, col, d[m_it][w / 16][2]);

          ST(row, col + 1, d[m_it][w / 16][1]);
          ST(row + 8, col + 1, d[m_it][w / 16][3]);

          ST(row, col + 8, d[m_it][w / 16][4]);
          ST(row + 8, col + 8, d[m_it][w / 16][6]);

          ST(row, col + 9, d[m_it][w / 16][5]);
          ST(row + 8, col + 9, d[m_it][w / 16][7]);
#undef ST
        }
      }

      WGMMAGlobalStore::sync_threads();
      if (threadIdx.x == 128) {
        TMAOps::store_async(&tensorMapC[group_idx], (bf16 *)&sC[0],
                            num_block_m * BM, num_block_n * BN);
        WGMMAGlobalStore::commit_group();
      }
    }
  }
}

void runGroupMatmul(const std::vector<GEMMParams> &params) {
  if (params.empty()) {
    return;
  }

  constexpr int BM = 128;
  constexpr int BN = 256;
  constexpr int BK = 64;
  constexpr int NUM_THREADS = 128 * 3;
  constexpr int QSIZE = 3;
  constexpr int CLUSTER_M = 2;
  constexpr int CLUSTER_N = 1;
  constexpr int NUM_SM = 128;
  static_assert(NUM_SM % (CLUSTER_M * CLUSTER_N) == 0);

  try {
    // Update TMA descriptors
    tma_descriptors.updateMaps(params);

    // Prepare dimension arrays
    std::vector<int> M_array(params.size());
    std::vector<int> N_array(params.size());
    std::vector<int> K_array(params.size());

    for (size_t i = 0; i < params.size(); i++) {
      M_array[i] = params[i].M;
      N_array[i] = params[i].N;
      K_array[i] = params[i].K;
    }

    // Device memory for dimensions
    int *d_M_array = nullptr, *d_N_array = nullptr, *d_K_array = nullptr;
    cudaCheck(cudaMalloc(&d_M_array, params.size() * sizeof(int)));
    cudaCheck(cudaMalloc(&d_N_array, params.size() * sizeof(int)));
    cudaCheck(cudaMalloc(&d_K_array, params.size() * sizeof(int)));

    // RAII cleanup helper
    struct DeviceArrays {
      int *M, *N, *K;
      DeviceArrays(int *m, int *n, int *k) : M(m), N(n), K(k) {}
      ~DeviceArrays() {
        if (M)
          cudaFree(M);
        if (N)
          cudaFree(N);
        if (K)
          cudaFree(K);
      }
    } arrays(d_M_array, d_N_array, d_K_array);

    // Copy dimension data
    cudaCheck(cudaMemcpy(d_M_array, M_array.data(), params.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_N_array, N_array.data(), params.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_K_array, K_array.data(), params.size() * sizeof(int),
                         cudaMemcpyHostToDevice));

    // Kernel configuration
    auto *kernel = groupMatmulKernel<BM, BN, BK, NUM_THREADS, QSIZE, NUM_SM,
                                     CLUSTER_M, CLUSTER_N>;

    constexpr size_t sMemSize = sizeof(SharedMemoryLayout<BM, BN, BK, QSIZE>);
    static_assert(sMemSize < 256 * 1024);

    cudaCheck(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    // Calculate grid size
    int blocks_per_group = NUM_SM / (CLUSTER_M * CLUSTER_N);
    int total_blocks = blocks_per_group * params.size();

    // Launch kernel
    kernel<<<total_blocks, NUM_THREADS, sMemSize>>>(
        params.size(), tma_descriptors.C_maps(), tma_descriptors.A_maps(),
        tma_descriptors.B_maps(), d_M_array, d_N_array, d_K_array);

    cudaCheck(cudaGetLastError());

  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Group GEMM error: ") + e.what());
  }
}

} // namespace GroupM10
