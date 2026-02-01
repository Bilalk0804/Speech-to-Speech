#ifndef S2S_QUANTIZATION_H
#define S2S_QUANTIZATION_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Quantization data types and utilities for INT8, FP16, and dynamic quantization
 */

/* Quantization schemes */
typedef enum {
    QT_NONE = 0,        // No quantization
    QT_INT8 = 1,        // 8-bit integer
    QT_UINT8 = 2,       // Unsigned 8-bit integer
    QT_FP16 = 3,        // 16-bit float (half precision)
    QT_FP8 = 4,         // 8-bit float (E4M3)
    QT_DYNAMIC = 5      // Dynamic per-channel quantization
} quantization_type_t;

/* Quantization statistics */
typedef struct {
    float min_val;
    float max_val;
    float mean_val;
    float std_val;
    float scale;
    int32_t zero_point;
    float per_channel_scale[1024];  // For per-channel quantization
} quant_stats_t;

/* Calibration sample context */
typedef struct {
    float* data;
    size_t num_samples;
    size_t sample_size;
    quant_stats_t* stats;
    uint32_t max_samples;
} calibration_context_t;

/* Quantized model metadata */
typedef struct {
    quantization_type_t quant_type;
    uint32_t layer_count;
    float compression_ratio;
    float accuracy_loss_percent;
    size_t original_size_bytes;
    size_t quantized_size_bytes;
    char model_name[256];
    uint32_t calibration_samples_used;
} quant_metadata_t;

/* ============================================
   Quantization Functions
   ============================================ */

/**
 * Initialize quantization subsystem
 * @return 0 on success, non-zero on error
 */
int quant_init(void);

/**
 * Cleanup quantization subsystem
 */
void quant_cleanup(void);

/* INT8 Quantization */

/**
 * Quantize float32 array to int8
 * @param input Input float array
 * @param output Output int8 array
 * @param size Number of elements
 * @param stats Output quantization statistics
 * @return 0 on success
 */
int quant_float32_to_int8(
    const float* input,
    int8_t* output,
    size_t size,
    quant_stats_t* stats
);

/**
 * Dequantize int8 array to float32
 * @param input Quantized int8 array
 * @param output Output float array
 * @param size Number of elements
 * @param stats Quantization statistics from quantization step
 * @return 0 on success
 */
int quant_int8_to_float32(
    const int8_t* input,
    float* output,
    size_t size,
    const quant_stats_t* stats
);

/**
 * Per-channel quantization for layer weights
 * @param input Input weight matrix (row-major)
 * @param output Output quantized matrix
 * @param rows Number of output channels
 * @param cols Number of input channels
 * @param stats Output per-channel statistics
 * @return 0 on success
 */
int quant_weights_per_channel_int8(
    const float* input,
    int8_t* output,
    uint32_t rows,
    uint32_t cols,
    quant_stats_t* stats
);

/* FP16 Quantization */

/**
 * Quantize float32 to float16 (half precision)
 * @param input Input float32 array
 * @param output Output uint16 array (storing float16 bits)
 * @param size Number of elements
 * @return 0 on success
 */
int quant_float32_to_float16(
    const float* input,
    uint16_t* output,
    size_t size
);

/**
 * Dequantize float16 to float32
 * @param input Input uint16 array (storing float16 bits)
 * @param output Output float32 array
 * @param size Number of elements
 * @return 0 on success
 */
int quant_float16_to_float32(
    const uint16_t* input,
    float* output,
    size_t size
);

/* Calibration for Quantization */

/**
 * Create calibration context for collecting statistics
 * @param max_samples Maximum number of samples to collect
 * @param sample_size Size of each sample
 * @return Calibration context or NULL on error
 */
calibration_context_t* quant_calibration_create(
    uint32_t max_samples,
    size_t sample_size
);

/**
 * Add calibration data
 * @param ctx Calibration context
 * @param data Float data to add
 * @param num_elements Number of elements
 * @return 0 on success
 */
int quant_calibration_add_data(
    calibration_context_t* ctx,
    const float* data,
    size_t num_elements
);

/**
 * Compute statistics from calibration data
 * @param ctx Calibration context
 * @param percentile Percentile for clipping (default 99.9)
 * @return 0 on success
 */
int quant_calibration_compute_stats(
    calibration_context_t* ctx,
    float percentile
);

/**
 * Get computed statistics
 * @param ctx Calibration context
 * @return Pointer to statistics
 */
const quant_stats_t* quant_calibration_get_stats(
    calibration_context_t* ctx
);

/**
 * Destroy calibration context
 */
void quant_calibration_destroy(calibration_context_t* ctx);

/* Model Quantization */

/**
 * Quantize binary model file
 * @param input_model Input model path (binary format from Phase 2)
 * @param output_model Output quantized model path
 * @param quant_type Target quantization type
 * @param calibration_data Optional calibration data file
 * @return 0 on success
 */
int quant_quantize_model(
    const char* input_model,
    const char* output_model,
    quantization_type_t quant_type,
    const char* calibration_data
);

/**
 * Get quantization metadata
 * @param model_path Path to quantized model
 * @param metadata Output metadata structure
 * @return 0 on success
 */
int quant_get_metadata(
    const char* model_path,
    quant_metadata_t* metadata
);

/* Dynamic Quantization */

/**
 * Perform dynamic quantization without calibration
 * @param input Input model path
 * @param output Output model path
 * @param quant_type Target type
 * @return 0 on success
 */
int quant_dynamic_quantize_model(
    const char* input,
    const char* output,
    quantization_type_t quant_type
);

/* Accuracy Evaluation */

/**
 * Compare quantized vs original model outputs
 * @param original_output Reference output
 * @param quantized_output Quantized output
 * @param size Number of elements
 * @param mae_out Mean absolute error output
 * @param rmse_out Root mean square error output
 * @return 0 on success
 */
int quant_evaluate_accuracy(
    const float* original_output,
    const float* quantized_output,
    size_t size,
    float* mae_out,
    float* rmse_out
);

#ifdef __cplusplus
}
#endif

#endif // S2S_QUANTIZATION_H
