#include "../include/s2s/matrix_ops.h"
#include <cstring>
#include <cmath>
#include <algorithm>

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

#ifdef ENABLE_SME2
// SME2 intrinsics would be included here once available
#endif

// ============================================================================
// Reference Implementations (portable C code)
// ============================================================================

/**
 * Reference GEMM implementation - basic matrix multiplication
 * C = alpha * A @ B + beta * C
 */
static void matops_gemm_reference(
    int transA, int transB,
    int32_t M, int32_t N, int32_t K,
    float alpha,
    const float* A, int32_t ldA,
    const float* B, int32_t ldB,
    float beta,
    float* C, int32_t ldC
) {
    for (int32_t i = 0; i < M; i++) {
        for (int32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            
            if (!transA && !transB) {
                // C[i,j] = A[i,:] @ B[:,j]
                for (int32_t k = 0; k < K; k++) {
                    sum += A[i * ldA + k] * B[k * ldB + j];
                }
            } else if (transA && !transB) {
                // C[i,j] = A[:,i] @ B[:,j]
                for (int32_t k = 0; k < K; k++) {
                    sum += A[k * ldA + i] * B[k * ldB + j];
                }
            } else if (!transA && transB) {
                // C[i,j] = A[i,:] @ B[j,:]
                for (int32_t k = 0; k < K; k++) {
                    sum += A[i * ldA + k] * B[j * ldB + k];
                }
            } else {
                // C[i,j] = A[:,i] @ B[j,:]
                for (int32_t k = 0; k < K; k++) {
                    sum += A[k * ldA + i] * B[j * ldB + k];
                }
            }
            
            C[i * ldC + j] = alpha * sum + beta * C[i * ldC + j];
        }
    }
}

/**
 * Reference ReLU activation
 */
static void matops_relu_reference(float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
}

/**
 * Reference GELU activation
 * GELU(x) = x * Phi(x), where Phi(x) is CDF of standard normal distribution
 * Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
static void matops_gelu_reference(float* data, size_t size) {
    static const float CONST_SQRT_2_PI = 0.7978845608f;
    static const float CONST_A = 0.044715f;
    
    for (size_t i = 0; i < size; i++) {
        float x = data[i];
        float x_cubed = x * x * x;
        float tanh_arg = CONST_SQRT_2_PI * (x + CONST_A * x_cubed);
        float gelu_val = 0.5f * x * (1.0f + tanh(tanh_arg));
        data[i] = gelu_val;
    }
}

/**
 * Reference Softmax
 * softmax(x_i) = exp(x_i) / sum(exp(x_j))
 */
static void matops_softmax_reference(float* data, int32_t rows, int32_t cols) {
    for (int32_t i = 0; i < rows; i++) {
        float* row = data + i * cols;
        
        // Find max for numerical stability
        float max_val = row[0];
        for (int32_t j = 1; j < cols; j++) {
            max_val = std::max(max_val, row[j]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int32_t j = 0; j < cols; j++) {
            row[j] = expf(row[j] - max_val);
            sum_exp += row[j];
        }
        
        // Normalize
        for (int32_t j = 0; j < cols; j++) {
            row[j] /= sum_exp;
        }
    }
}

/**
 * Reference Layer Normalization
 * y = gamma * (x - mean) / sqrt(var + eps) + beta
 */
static void matops_layer_norm_reference(
    float* x,
    const float* gamma,
    const float* beta,
    int32_t batch_size,
    int32_t hidden_dim,
    float eps
) {
    for (int32_t b = 0; b < batch_size; b++) {
        float* x_batch = x + b * hidden_dim;
        
        // Compute mean
        float mean = 0.0f;
        for (int32_t i = 0; i < hidden_dim; i++) {
            mean += x_batch[i];
        }
        mean /= hidden_dim;
        
        // Compute variance
        float var = 0.0f;
        for (int32_t i = 0; i < hidden_dim; i++) {
            float diff = x_batch[i] - mean;
            var += diff * diff;
        }
        var /= hidden_dim;
        
        // Normalize and scale
        float std_dev = sqrtf(var + eps);
        for (int32_t i = 0; i < hidden_dim; i++) {
            x_batch[i] = gamma[i] * (x_batch[i] - mean) / std_dev + beta[i];
        }
    }
}

// ============================================================================
// NEON Optimized Implementations (ARM NEON)
// ============================================================================

#ifdef ENABLE_NEON

/**
 * NEON-optimized GEMM for small matrices
 * Processes 4 floats at a time using NEON
 */
static void matops_gemm_neon(
    int transA, int transB,
    int32_t M, int32_t N, int32_t K,
    float alpha,
    const float* A, int32_t ldA,
    const float* B, int32_t ldB,
    float beta,
    float* C, int32_t ldC
) {
    // For simplicity, use reference implementation
    // Full NEON optimization would require specialized kernels for each transpose combination
    matops_gemm_reference(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

/**
 * NEON-optimized ReLU
 * Process 4 floats per iteration
 */
static void matops_relu_neon(float* data, size_t size) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    size_t i = 0;
    
    // Process 4 elements at a time
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t result = vmaxq_f32(v, zero);
        vst1q_f32(data + i, result);
    }
    
    // Process remaining elements
    for (; i < size; i++) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
}

#endif // ENABLE_NEON

// ============================================================================
// SME2 Optimized Implementations (ARM SVE/SME2)
// ============================================================================

#ifdef ENABLE_SME2

/**
 * SME2-optimized GEMM
 * Uses SVE registers for wider vectorization
 */
static void matops_gemm_sme2(
    int transA, int transB,
    int32_t M, int32_t N, int32_t K,
    float alpha,
    const float* A, int32_t ldA,
    const float* B, int32_t ldB,
    float beta,
    float* C, int32_t ldC
) {
    // SME2 allows much larger vectorization
    // For now, fall back to reference
    // Full SME2 implementation would use SPMM or custom assembly
    matops_gemm_reference(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

#endif // ENABLE_SME2

// ============================================================================
// C Interface Implementation
// ============================================================================

static MatrixOps::Backend g_backend = MatrixOps::AUTO;
static bool g_initialized = false;

int matops_init() {
    if (g_initialized) return 0;
    
#ifdef ENABLE_SME2
    g_backend = MatrixOps::SME2;
#elif ENABLE_NEON
    g_backend = MatrixOps::NEON;
#else
    g_backend = MatrixOps::REFERENCE;
#endif
    
    g_initialized = true;
    return 0;
}

void matops_cleanup() {
    g_initialized = false;
}

void matops_gemm(
    int transA, int transB,
    int32_t M, int32_t N, int32_t K,
    float alpha,
    const float* A, int32_t ldA,
    const float* B, int32_t ldB,
    float beta,
    float* C, int32_t ldC
) {
    if (!g_initialized) matops_init();
    
#ifdef ENABLE_SME2
    if (g_backend == MatrixOps::SME2) {
        matops_gemm_sme2(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
#endif
    
#ifdef ENABLE_NEON
    if (g_backend == MatrixOps::NEON) {
        matops_gemm_neon(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
#endif
    
    matops_gemm_reference(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void matops_batched_gemm(
    int32_t batch_size,
    int32_t M, int32_t N, int32_t K,
    const float* A,
    const float* B,
    float* C
) {
    size_t a_batch_stride = M * K;
    size_t b_batch_stride = K * N;
    size_t c_batch_stride = M * N;
    
    for (int32_t b = 0; b < batch_size; b++) {
        matops_gemm(
            0, 0,  // No transpose
            M, N, K,
            1.0f,
            A + b * a_batch_stride, K,
            B + b * b_batch_stride, N,
            0.0f,
            C + b * c_batch_stride, N
        );
    }
}

void matops_gemv(
    int transA,
    int32_t M, int32_t N,
    float alpha,
    const float* A, int32_t ldA,
    const float* x, int32_t incx,
    float beta,
    float* y, int32_t incy
) {
    // Matrix-vector: y = alpha * A @ x + beta * y
    if (!transA) {
        // y[i] = A[i,:] @ x[:]
        for (int32_t i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int32_t j = 0; j < N; j++) {
                sum += A[i * ldA + j] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    } else {
        // y[j] = A[:,j] @ x[:]
        for (int32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int32_t i = 0; i < M; i++) {
                sum += A[i * ldA + j] * x[i * incx];
            }
            y[j * incy] = alpha * sum + beta * y[j * incy];
        }
    }
}

void matops_relu(float* data, size_t size) {
    if (!g_initialized) matops_init();
    
#ifdef ENABLE_NEON
    if (g_backend == MatrixOps::NEON) {
        matops_relu_neon(data, size);
        return;
    }
#endif
    
    matops_relu_reference(data, size);
}

void matops_gelu(float* data, size_t size) {
    matops_gelu_reference(data, size);
}

void matops_softmax(float* data, int32_t rows, int32_t cols) {
    matops_softmax_reference(data, rows, cols);
}

void matops_layer_norm(
    float* x,
    const float* gamma,
    const float* beta,
    int32_t batch_size,
    int32_t hidden_dim,
    float eps
) {
    matops_layer_norm_reference(x, gamma, beta, batch_size, hidden_dim, eps);
}

void matops_convert_nchw_to_nhwc(
    const float* src, float* dst,
    int32_t N, int32_t C, int32_t H, int32_t W
) {
    for (int32_t n = 0; n < N; n++) {
        for (int32_t h = 0; h < H; h++) {
            for (int32_t w = 0; w < W; w++) {
                for (int32_t c = 0; c < C; c++) {
                    int src_idx = n * C * H * W + c * H * W + h * W + w;
                    int dst_idx = n * H * W * C + h * W * C + w * C + c;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

// ============================================================================
// C++ Interface Implementation
// ============================================================================

namespace s2s {

MatrixOps& MatrixOps::instance() {
    static MatrixOps m_instance;
    return m_instance;
}

int MatrixOps::init(int backend) {
    if (initialized) return 0;
    
    if (backend == 0) {  // AUTO
#ifdef ENABLE_SME2
        current_backend = SME2;
#elif ENABLE_NEON
        current_backend = NEON;
#else
        current_backend = REFERENCE;
#endif
    } else {
        current_backend = static_cast<Backend>(backend);
    }
    
    initialized = true;
    return matops_init();
}

void MatrixOps::gemm(
    bool transA, bool transB,
    int32_t M, int32_t N, int32_t K,
    float alpha,
    const float* A, int32_t ldA,
    const float* B, int32_t ldB,
    float beta,
    float* C, int32_t ldC
) {
    matops_gemm(transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void MatrixOps::batched_gemm(
    int32_t batch_size,
    int32_t M, int32_t N, int32_t K,
    const float* A,
    const float* B,
    float* C
) {
    matops_batched_gemm(batch_size, M, N, K, A, B, C);
}

void MatrixOps::relu(float* data, size_t size) {
    matops_relu(data, size);
}

void MatrixOps::gelu(float* data, size_t size) {
    matops_gelu(data, size);
}

void MatrixOps::softmax(float* data, int32_t rows, int32_t cols) {
    matops_softmax(data, rows, cols);
}

void MatrixOps::layer_norm(
    float* x,
    const float* gamma,
    const float* beta,
    int32_t batch_size,
    int32_t hidden_dim,
    float eps
) {
    matops_layer_norm(x, gamma, beta, batch_size, hidden_dim, eps);
}

} // namespace s2s
