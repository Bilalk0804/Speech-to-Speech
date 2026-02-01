/**
 * ARM Cortex-X925 Optimization Module
 * Phase 3 Task 8: Hardware-specific performance tuning
 * 
 * This module provides optimization for ARM Cortex-X925 SoC
 * with focus on SME2 (Scalable Matrix Extension), advanced prefetching,
 * and memory hierarchy optimization.
 */

#ifndef S2S_CORTEX_X925_H
#define S2S_CORTEX_X925_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================
   CPU Feature Detection
   ============================================ */

/**
 * CPU feature flags for Cortex-X925
 */
typedef enum {
    CPU_FEATURE_SVE = 0x01,        // Scalable Vector Extension
    CPU_FEATURE_SVE2 = 0x02,       // SVE2
    CPU_FEATURE_SME = 0x04,        // Scalable Matrix Extension
    CPU_FEATURE_SME2 = 0x08,       // SME2
    CPU_FEATURE_FP16 = 0x10,       // Half-precision float
    CPU_FEATURE_BF16 = 0x20,       // Bfloat16
    CPU_FEATURE_INT8 = 0x40,       // INT8 SIMD
    CPU_FEATURE_DOTPROD = 0x80,    // UDOT/SDOT instructions
} cpu_feature_t;

/**
 * CPU information structure
 */
typedef struct {
    uint32_t features;              // Enabled features
    uint32_t sve_vector_bits;       // SVE vector length (128-2048)
    uint32_t sme_vector_bits;       // SME tile size
    uint32_t num_cores;             // Number of cores
    uint32_t l1_cache_size;         // L1 data cache (KB)
    uint32_t l2_cache_size;         // L2 cache (KB)
    uint32_t l3_cache_size;         // L3 cache (KB)
    float clock_ghz;                // CPU frequency
    char model_name[64];            // e.g., "ARM Cortex-X925"
} cpu_info_t;

/**
 * Detect CPU features
 * @return Bitmask of detected features
 */
uint32_t cortex_x925_detect_features(void);

/**
 * Get detailed CPU information
 * @param info Pointer to cpu_info_t structure
 * @return 0 on success, -1 on error
 */
int cortex_x925_get_cpu_info(cpu_info_t *info);

/* ============================================
   SME2 Optimizations
   ============================================ */

/**
 * Matrix multiply using SME2
 * Performs C = A * B with SME2 tile operations
 * 
 * @param A Input matrix A [m x k]
 * @param B Input matrix B [k x n]
 * @param C Output matrix C [m x n]
 * @param m Rows of A and C
 * @param k Columns of A and rows of B
 * @param n Columns of B and C
 * @return 0 on success, -1 on error
 */
int cortex_x925_sme2_matmul_f32(const float *A, const float *B, float *C,
                               uint32_t m, uint32_t k, uint32_t n);

/**
 * Matrix multiply with INT8 quantization using SME2
 * Performs C = A * B with 8-bit integer types
 * 
 * @param A Input matrix A (INT8) [m x k]
 * @param B Input matrix B (INT8) [k x n]
 * @param C Output matrix C (INT32) [m x n]
 * @param m Rows
 * @param k Inner dimension
 * @param n Columns
 * @param scale Scaling factor for output
 * @return 0 on success, -1 on error
 */
int cortex_x925_sme2_matmul_i8(const int8_t *A, const int8_t *B, int32_t *C,
                              uint32_t m, uint32_t k, uint32_t n, float scale);

/**
 * Batch matrix multiply with SME2
 * Process multiple matrices efficiently
 * 
 * @param A_batch Array of pointers to matrices
 * @param B_batch Array of pointers to matrices
 * @param C_batch Array of pointers to output matrices
 * @param batch_size Number of matrices
 * @param m Rows per matrix
 * @param k Inner dimension
 * @param n Columns per matrix
 * @return 0 on success, -1 on error
 */
int cortex_x925_sme2_batch_matmul(const float *const *A_batch,
                                 const float *const *B_batch,
                                 float *const *C_batch,
                                 uint32_t batch_size,
                                 uint32_t m, uint32_t k, uint32_t n);

/* ============================================
   SVE Optimizations
   ============================================ */

/**
 * Vector convolution using SVE
 * Performs 1D or 2D convolution with scalable vectors
 * 
 * @param input Input data
 * @param kernel Convolution kernel
 * @param output Output data
 * @param input_size Input size
 * @param kernel_size Kernel size
 * @param stride Stride
 * @return 0 on success, -1 on error
 */
int cortex_x925_sve_convolve(const float *input, const float *kernel,
                            float *output, uint32_t input_size,
                            uint32_t kernel_size, uint32_t stride);

/**
 * Element-wise operations using SVE
 * 
 * @param input Input vector
 * @param output Output vector
 * @param size Vector size
 * @param op Operation: 0=ReLU, 1=Sigmoid, 2=Tanh
 * @return 0 on success, -1 on error
 */
int cortex_x925_sve_elementwise(const float *input, float *output,
                               uint32_t size, int op);

/**
 * Reduction operations using SVE
 * 
 * @param input Input vector
 * @param size Vector size
 * @param op Operation: 0=Sum, 1=Max, 2=Min, 3=Mean
 * @return Reduction result
 */
float cortex_x925_sve_reduce(const float *input, uint32_t size, int op);

/* ============================================
   Memory Hierarchy Optimization
   ============================================ */

/**
 * Memory hierarchy levels
 */
typedef enum {
    MEM_L1_CACHE,      // L1 data cache (~64KB)
    MEM_L2_CACHE,      // L2 cache (~512KB-1MB)
    MEM_L3_CACHE,      // L3 cache (~4-8MB)
    MEM_MAIN,          // Main memory
} memory_level_t;

/**
 * Cache line size (typically 64 bytes for Cortex-X925)
 */
#define CACHE_LINE_SIZE 64

/**
 * Prefetch policy for memory access patterns
 */
typedef enum {
    PREFETCH_TEMPORAL_L1,   // Prefetch to L1 cache
    PREFETCH_TEMPORAL_L2,   // Prefetch to L2 cache
    PREFETCH_NON_TEMPORAL,  // Prefetch for one-time use
} prefetch_policy_t;

/**
 * Configure prefetch for sequential access pattern
 * 
 * @param base_addr Base address of data
 * @param size Total size
 * @param stride Access stride
 * @param policy Prefetch policy
 * @return 0 on success, -1 on error
 */
int cortex_x925_prefetch_sequential(void *base_addr, size_t size,
                                   uint32_t stride, prefetch_policy_t policy);

/**
 * Configure prefetch for random access pattern
 * 
 * @param addresses Array of addresses to prefetch
 * @param num_addresses Number of addresses
 * @return 0 on success, -1 on error
 */
int cortex_x925_prefetch_random(const void *const *addresses,
                               uint32_t num_addresses);

/**
 * Memory barrier for cache coherence
 * 
 * @param level Cache level to synchronize
 */
void cortex_x925_cache_barrier(memory_level_t level);

/* ============================================
   Performance Monitoring
   ============================================ */

/**
 * Performance event types
 */
typedef enum {
    PERF_CYCLES,               // CPU cycles
    PERF_INSTRUCTIONS,         // Executed instructions
    PERF_L1_HITS,             // L1 cache hits
    PERF_L1_MISSES,           // L1 cache misses
    PERF_L2_HITS,             // L2 cache hits
    PERF_L2_MISSES,           // L2 cache misses
    PERF_BRANCH_PRED_HITS,    // Branch prediction hits
    PERF_BRANCH_MISSES,       // Branch prediction misses
    PERF_TLB_MISSES,          // TLB misses
    PERF_SVE_OPERATIONS,      // SVE operations
    PERF_SME_OPERATIONS,      // SME operations
} perf_event_t;

/**
 * Performance counter handle
 */
typedef struct cortex_x925_perfcounter *perfcounter_t;

/**
 * Create performance counter
 * 
 * @param event Event type to monitor
 * @return Counter handle, NULL on error
 */
perfcounter_t cortex_x925_perfcounter_create(perf_event_t event);

/**
 * Start performance counter
 * 
 * @param counter Counter handle
 * @return 0 on success, -1 on error
 */
int cortex_x925_perfcounter_start(perfcounter_t counter);

/**
 * Stop performance counter and get value
 * 
 * @param counter Counter handle
 * @param value Output value
 * @return 0 on success, -1 on error
 */
int cortex_x925_perfcounter_stop(perfcounter_t counter, uint64_t *value);

/**
 * Reset counter
 * 
 * @param counter Counter handle
 * @return 0 on success, -1 on error
 */
int cortex_x925_perfcounter_reset(perfcounter_t counter);

/**
 * Destroy counter
 * 
 * @param counter Counter handle
 */
void cortex_x925_perfcounter_destroy(perfcounter_t counter);

/**
 * Get cache line hits/misses
 * 
 * @param addr Memory address
 * @param access_size Size of access
 * @return Cache efficiency percentage (0-100)
 */
float cortex_x925_get_cache_efficiency(const void *addr, size_t access_size);

/* ============================================
   NEON (Advanced) Optimizations
   ============================================ */

/**
 * NEON vector multiply-accumulate (FP32)
 * C += A * B for vectors
 * 
 * @param A Vector A
 * @param B Vector B
 * @param C Accumulator vector
 * @param size Vector size (must be multiple of 4)
 */
void cortex_x925_neon_fma_f32(const float *A, const float *B, float *C,
                             uint32_t size);

/**
 * NEON quantization (FP32 to INT8)
 * 
 * @param input FP32 input
 * @param output INT8 output
 * @param size Array size
 * @param scale Quantization scale
 */
void cortex_x925_neon_quantize_f32_i8(const float *input, int8_t *output,
                                     uint32_t size, float scale);

/**
 * NEON dequantization (INT8 to FP32)
 * 
 * @param input INT8 input
 * @param output FP32 output
 * @param size Array size
 * @param scale Dequantization scale
 */
void cortex_x925_neon_dequantize_i8_f32(const int8_t *input, float *output,
                                       uint32_t size, float scale);

/* ============================================
   Optimization Guidelines
   ============================================ */

/**
 * Optimization profile structure
 */
typedef struct {
    char name[32];
    uint32_t use_sme2;         // 1 = enable SME2 optimizations
    uint32_t use_sve;          // 1 = enable SVE optimizations
    uint32_t use_neon;         // 1 = enable NEON optimizations
    uint32_t cache_aware;      // 1 = optimize for cache hierarchy
    uint32_t prefetch_enabled; // 1 = enable prefetching
    uint32_t use_fp16;         // 1 = use FP16 where possible
    uint32_t use_int8;         // 1 = use INT8 quantization
} optimization_profile_t;

/**
 * Get recommended optimization profile
 * 
 * @param task Task type: 0=ASR, 1=MT, 2=TTS, 3=VideoSync
 * @return Optimization profile
 */
optimization_profile_t cortex_x925_get_optimization_profile(int task);

/**
 * Set optimization profile
 * 
 * @param profile Profile to apply
 * @return 0 on success, -1 on error
 */
int cortex_x925_set_optimization_profile(const optimization_profile_t *profile);

/**
 * Benchmark operation on Cortex-X925
 * 
 * @param operation_name Name of operation
 * @param iterations Number of iterations
 * @param operation_func Pointer to operation function
 * @param arg Operation argument
 * @return Execution time in milliseconds
 */
float cortex_x925_benchmark(const char *operation_name, uint32_t iterations,
                           void (*operation_func)(void *), void *arg);

#ifdef __cplusplus
}
#endif

#endif // S2S_CORTEX_X925_H
