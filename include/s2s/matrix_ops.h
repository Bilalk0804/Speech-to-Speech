#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Basic matrix operations interface
 * Supports multiple backend implementations (NEON, SME2, reference)
 */

/**
 * Initialize matrix operations library
 * Detects CPU capabilities and sets up optimal kernels
 */
int matops_init();

/**
 * Cleanup resources
 */
void matops_cleanup();

/**
 * Matrix multiplication: C = alpha * A @ B + beta * C
 * 
 * @param transA Whether to transpose A
 * @param transB Whether to transpose B
 * @param M Number of rows in A
 * @param N Number of columns in B
 * @param K Number of columns in A / rows in B
 * @param alpha Scalar multiplier
 * @param A Input matrix A [M x K] or [K x M]
 * @param ldA Leading dimension of A
 * @param B Input matrix B [K x N] or [N x K]
 * @param ldB Leading dimension of B
 * @param beta Scalar for accumulation
 * @param C Output matrix [M x N]
 * @param ldC Leading dimension of C
 */
void matops_gemm(
    int transA, int transB,
    int32_t M, int32_t N, int32_t K,
    float alpha,
    const float* A, int32_t ldA,
    const float* B, int32_t ldB,
    float beta,
    float* C, int32_t ldC
);

/**
 * Batch matrix multiplication for sequence processing
 * Useful for attention mechanisms
 * 
 * @param batch_size Number of matrices to multiply
 * @param M,N,K Matrix dimensions
 * @param A Batch of A matrices [batch_size x M x K]
 * @param B Batch of B matrices [batch_size x K x N]
 * @param C Output batch [batch_size x M x N]
 */
void matops_batched_gemm(
    int32_t batch_size,
    int32_t M, int32_t N, int32_t K,
    const float* A,
    const float* B,
    float* C
);

/**
 * Matrix-Vector multiplication: y = alpha * A @ x + beta * y
 */
void matops_gemv(
    int transA,
    int32_t M, int32_t N,
    float alpha,
    const float* A, int32_t ldA,
    const float* x, int32_t incx,
    float beta,
    float* y, int32_t incy
);

/**
 * Element-wise activation functions
 */
void matops_relu(float* data, size_t size);
void matops_gelu(float* data, size_t size);
void matops_softmax(float* data, int32_t rows, int32_t cols);

/**
 * Normalization operations (for layer norm)
 */
void matops_layer_norm(
    float* x,
    const float* gamma,
    const float* beta,
    int32_t batch_size,
    int32_t hidden_dim,
    float eps
);

/**
 * Memory layout conversion utilities
 */
void matops_convert_nchw_to_nhwc(
    const float* src, float* dst,
    int32_t N, int32_t C, int32_t H, int32_t W
);

#ifdef __cplusplus
}

namespace s2s {

/**
 * C++ wrapper for matrix operations
 * Provides higher-level interface with RAII
 */
class MatrixOps {
public:
    static MatrixOps& instance();

    /**
     * Initialize with specified backend
     * @param backend 0=auto, 1=reference, 2=neon, 3=sme2
     */
    int init(int backend = 0);

    void gemm(
        bool transA, bool transB,
        int32_t M, int32_t N, int32_t K,
        float alpha,
        const float* A, int32_t ldA,
        const float* B, int32_t ldB,
        float beta,
        float* C, int32_t ldC
    );

    void batched_gemm(
        int32_t batch_size,
        int32_t M, int32_t N, int32_t K,
        const float* A,
        const float* B,
        float* C
    );

    void relu(float* data, size_t size);
    void gelu(float* data, size_t size);
    void softmax(float* data, int32_t rows, int32_t cols);
    void layer_norm(
        float* x,
        const float* gamma,
        const float* beta,
        int32_t batch_size,
        int32_t hidden_dim,
        float eps
    );

    enum Backend {
        AUTO = 0,
        REFERENCE = 1,
        NEON = 2,
        SME2 = 3
    };

    Backend get_backend() const { return current_backend; }

private:
    MatrixOps() = default;
    Backend current_backend;
    bool initialized;
};

} // namespace s2s

#endif // __cplusplus

#endif // MATRIX_OPS_H
