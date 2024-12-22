namespace M10 {

class CUDABarrier {
private:
    uint64_t* barrier_ptr;
    uint32_t smem_addr;

    __device__ __forceinline__ static uint32_t get_smem_ptr(uint64_t* ptr) {
        return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    }

public:
    __device__ __forceinline__ CUDABarrier(uint64_t* ptr) : barrier_ptr(ptr) {
        smem_addr = get_smem_ptr(ptr);
    }

    // Initialize barrier with thread count and transaction count
    __device__ __forceinline__ void init(int thread_count, int transaction_count) {
        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(smem_addr), "r"(thread_count + transaction_count)
        );
    }

    // Set expected bytes for transactions
    __device__ __forceinline__ void expect_tx(uint32_t bytes) {
        asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(smem_addr), "r"(bytes));
    }

    // Wait for barrier with phase bit
    __device__ __forceinline__ void wait(int phase_bit) {
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(smem_addr), "r"(phase_bit)
        );
    }

    // Wait for barrier with phase bit (cluster-wide)
    __device__ __forceinline__ void wait_cluster(int phase_bit) {
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(smem_addr), "r"(phase_bit)
        );
    }

    // Signal barrier arrival
    __device__ __forceinline__ void arrive(uint32_t count = 1) {
        asm volatile (
            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
            :: "r"(smem_addr), "r"(count)
            : "memory"
        );
    }

    // Signal barrier arrival (cluster-wide)
    __device__ __forceinline__ void arrive_cluster(uint32_t cta_id, uint32_t count = 1) {
        asm volatile(
            "{\n\t"
            ".reg .b32 remAddr32;\n\t"
            "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
            "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n\t"
            "}"
            :: "r"(smem_addr), "r"(cta_id), "r"(count)
        );
    }
};

} // namespace M10 
