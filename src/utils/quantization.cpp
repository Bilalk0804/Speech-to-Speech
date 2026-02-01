#include "../../include/s2s/quantization.h"
#include "../../include/s2s/logger.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ============================================
   Utility Functions
   ============================================ */

static float float32_to_float16_scalar(float f32) {
    // Convert float32 to float16 (simplified)
    // This is a simplified implementation
    if (isnan(f32)) return f32;
    if (isinf(f32)) return f32;
    
    // Clamp to float16 range
    if (f32 > 65504.0f) return 65504.0f;
    if (f32 < -65504.0f) return -65504.0f;
    if (f32 > 0 && f32 < 6.1035e-5f) return 6.1035e-5f;
    if (f32 < 0 && f32 > -6.1035e-5f) return -6.1035e-5f;
    
    return f32;  // Placeholder for actual conversion
}

static float float16_to_float32_scalar(uint16_t f16_bits) {
    // Convert float16 bits to float32
    uint32_t f32_bits = 0;
    
    uint16_t sign = (f16_bits >> 15) & 0x1;
    uint16_t exp = (f16_bits >> 10) & 0x1f;
    uint16_t frac = f16_bits & 0x3ff;
    
    if (exp == 0) {
        if (frac == 0) {
            f32_bits = sign << 31;
        } else {
            exp = 0;
            while ((frac & 0x400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3ff;
            exp = 127 - 24 - exp;
            f32_bits = (sign << 31) | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f32_bits = (sign << 31) | 0x7f800000;
        if (frac != 0) {
            f32_bits |= 0x400000 | (frac << 13);
        }
    } else {
        exp = exp + 127 - 15;
        f32_bits = (sign << 31) | (exp << 23) | (frac << 13);
    }
    
    return *(float*)&f32_bits;
}

/* ============================================
   Initialization
   ============================================ */

int quant_init(void) {
    log_info("Quantization subsystem initialized");
    return 0;
}

void quant_cleanup(void) {
    log_info("Quantization subsystem cleaned up");
}

/* ============================================
   INT8 Quantization
   ============================================ */

int quant_float32_to_int8(
    const float* input,
    int8_t* output,
    size_t size,
    quant_stats_t* stats) {
    
    if (!input || !output || !stats) {
        log_error("Invalid parameters for float32_to_int8");
        return -1;
    }
    
    // Compute statistics
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    float sum = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float val = input[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    
    float mean = sum / (float)size;
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = input[i] - mean;
        sum_sq_diff += diff * diff;
    }
    float std = sqrtf(sum_sq_diff / (float)size);
    
    // Clip to int8 range [-128, 127]
    // Use symmetric quantization around 0
    float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
    float scale = 127.0f / (abs_max + 1e-8f);
    
    stats->min_val = min_val;
    stats->max_val = max_val;
    stats->mean_val = mean;
    stats->std_val = std;
    stats->scale = scale;
    stats->zero_point = 0;
    
    // Quantize
    for (size_t i = 0; i < size; i++) {
        float scaled = input[i] * scale;
        int32_t quantized = (int32_t)roundf(scaled);
        
        // Clamp to int8
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        
        output[i] = (int8_t)quantized;
    }
    
    log_debug("Quantized %zu elements to INT8 (scale=%.6f, range=[%.2f, %.2f])",
              size, scale, min_val, max_val);
    
    return 0;
}

int quant_int8_to_float32(
    const int8_t* input,
    float* output,
    size_t size,
    const quant_stats_t* stats) {
    
    if (!input || !output || !stats) {
        log_error("Invalid parameters for int8_to_float32");
        return -1;
    }
    
    float inv_scale = 1.0f / (stats->scale + 1e-8f);
    
    for (size_t i = 0; i < size; i++) {
        output[i] = (float)input[i] * inv_scale;
    }
    
    return 0;
}

int quant_weights_per_channel_int8(
    const float* input,
    int8_t* output,
    uint32_t rows,
    uint32_t cols,
    quant_stats_t* stats) {
    
    if (!input || !output || !stats) {
        log_error("Invalid parameters for per-channel quantization");
        return -1;
    }
    
    if (rows > 1024) {
        log_error("Per-channel quantization supports max 1024 channels");
        return -1;
    }
    
    memset(stats, 0, sizeof(quant_stats_t));
    
    // Quantize each output channel (row) independently
    for (uint32_t r = 0; r < rows; r++) {
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        
        // Find min/max for this channel
        for (uint32_t c = 0; c < cols; c++) {
            float val = input[r * cols + c];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
        float scale = 127.0f / (abs_max + 1e-8f);
        stats->per_channel_scale[r] = scale;
        
        // Quantize this channel
        for (uint32_t c = 0; c < cols; c++) {
            float val = input[r * cols + c];
            float scaled = val * scale;
            int32_t quantized = (int32_t)roundf(scaled);
            
            if (quantized > 127) quantized = 127;
            if (quantized < -128) quantized = -128;
            
            output[r * cols + c] = (int8_t)quantized;
        }
    }
    
    log_debug("Per-channel INT8 quantization complete for %u channels", rows);
    return 0;
}

/* ============================================
   FP16 Quantization
   ============================================ */

int quant_float32_to_float16(
    const float* input,
    uint16_t* output,
    size_t size) {
    
    if (!input || !output) {
        log_error("Invalid parameters for float32_to_float16");
        return -1;
    }
    
    for (size_t i = 0; i < size; i++) {
        float f32 = float32_to_float16_scalar(input[i]);
        uint32_t bits = *(uint32_t*)&f32;
        
        // Round to nearest even
        uint16_t f16 = (bits >> 16) & 0x8000;
        uint32_t exp = (bits >> 23) & 0xff;
        uint32_t frac = bits & 0x7fffff;
        
        if (exp == 255) {
            f16 |= 0x7c00;
            if (frac) f16 |= 0x200;
        } else if (exp == 0 && frac == 0) {
            // Already 0
        } else if (exp < 113) {
            // Underflow to zero
        } else if (exp > 142) {
            // Overflow to infinity
            f16 |= 0x7c00;
        } else {
            uint32_t f16_exp = exp - 112;
            uint16_t f16_frac = (frac >> 13) & 0x3ff;
            f16 |= (f16_exp << 10) | f16_frac;
        }
        
        output[i] = f16;
    }
    
    log_debug("Converted %zu elements from FP32 to FP16", size);
    return 0;
}

int quant_float16_to_float32(
    const uint16_t* input,
    float* output,
    size_t size) {
    
    if (!input || !output) {
        log_error("Invalid parameters for float16_to_float32");
        return -1;
    }
    
    for (size_t i = 0; i < size; i++) {
        output[i] = float16_to_float32_scalar(input[i]);
    }
    
    log_debug("Converted %zu elements from FP16 to FP32", size);
    return 0;
}

/* ============================================
   Calibration
   ============================================ */

calibration_context_t* quant_calibration_create(
    uint32_t max_samples,
    size_t sample_size) {
    
    calibration_context_t* ctx = malloc(sizeof(calibration_context_t));
    if (!ctx) {
        log_error("Failed to allocate calibration context");
        return NULL;
    }
    
    ctx->max_samples = max_samples;
    ctx->sample_size = sample_size;
    ctx->num_samples = 0;
    
    ctx->data = malloc(max_samples * sample_size);
    if (!ctx->data) {
        log_error("Failed to allocate calibration data");
        free(ctx);
        return NULL;
    }
    
    ctx->stats = malloc(sizeof(quant_stats_t));
    if (!ctx->stats) {
        log_error("Failed to allocate calibration stats");
        free(ctx->data);
        free(ctx);
        return NULL;
    }
    
    memset(ctx->stats, 0, sizeof(quant_stats_t));
    
    log_info("Created calibration context: max_samples=%u, sample_size=%zu",
             max_samples, sample_size);
    
    return ctx;
}

int quant_calibration_add_data(
    calibration_context_t* ctx,
    const float* data,
    size_t num_elements) {
    
    if (!ctx || !data) {
        log_error("Invalid parameters for calibration_add_data");
        return -1;
    }
    
    if (num_elements != ctx->sample_size) {
        log_error("Data size mismatch: expected %zu, got %zu",
                  ctx->sample_size, num_elements);
        return -1;
    }
    
    if (ctx->num_samples >= ctx->max_samples) {
        log_warning("Calibration buffer full, discarding oldest sample");
        // Shift data (simple FIFO)
        memmove(ctx->data, ctx->data + num_elements,
                (ctx->max_samples - 1) * num_elements * sizeof(float));
        ctx->num_samples = ctx->max_samples - 1;
    }
    
    memcpy(ctx->data + ctx->num_samples * num_elements,
           data, num_elements * sizeof(float));
    ctx->num_samples++;
    
    return 0;
}

int quant_calibration_compute_stats(
    calibration_context_t* ctx,
    float percentile) {
    
    if (!ctx || ctx->num_samples == 0) {
        log_error("Invalid calibration context or no data");
        return -1;
    }
    
    float* flat_data = ctx->data;
    size_t total_elements = ctx->num_samples * ctx->sample_size;
    
    // Compute min, max, mean, std
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    float sum = 0.0f;
    
    for (size_t i = 0; i < total_elements; i++) {
        float val = flat_data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    
    float mean = sum / (float)total_elements;
    float sum_sq_diff = 0.0f;
    
    for (size_t i = 0; i < total_elements; i++) {
        float diff = flat_data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    
    float std = sqrtf(sum_sq_diff / (float)total_elements);
    
    // Percentile-based clipping
    float* sorted = malloc(total_elements * sizeof(float));
    memcpy(sorted, flat_data, total_elements * sizeof(float));
    
    // Simple bubble sort for percentile (TODO: use quickselect for performance)
    for (size_t i = 0; i < total_elements; i++) {
        for (size_t j = i + 1; j < total_elements; j++) {
            if (sorted[j] < sorted[i]) {
                float tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }
        }
    }
    
    size_t percentile_idx = (size_t)((percentile / 100.0f) * (float)total_elements);
    if (percentile_idx >= total_elements) percentile_idx = total_elements - 1;
    
    float clip_val = fabsf(sorted[percentile_idx]);
    if (clip_val > fabsf(min_val)) min_val = -clip_val;
    if (clip_val > fabsf(max_val)) max_val = clip_val;
    
    free(sorted);
    
    ctx->stats->min_val = min_val;
    ctx->stats->max_val = max_val;
    ctx->stats->mean_val = mean;
    ctx->stats->std_val = std;
    float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
    ctx->stats->scale = 127.0f / (abs_max + 1e-8f);
    ctx->stats->zero_point = 0;
    
    log_info("Calibration stats computed: min=%.2f, max=%.2f, mean=%.2f, std=%.2f",
             min_val, max_val, mean, std);
    
    return 0;
}

const quant_stats_t* quant_calibration_get_stats(
    calibration_context_t* ctx) {
    
    if (!ctx) return NULL;
    return ctx->stats;
}

void quant_calibration_destroy(calibration_context_t* ctx) {
    if (!ctx) return;
    
    if (ctx->data) free(ctx->data);
    if (ctx->stats) free(ctx->stats);
    free(ctx);
    
    log_debug("Calibration context destroyed");
}

/* ============================================
   Model Quantization
   ============================================ */

int quant_quantize_model(
    const char* input_model,
    const char* output_model,
    quantization_type_t quant_type,
    const char* calibration_data) {
    
    if (!input_model || !output_model) {
        log_error("Invalid model paths");
        return -1;
    }
    
    log_info("Starting model quantization: %s -> %s (type=%d)",
             input_model, output_model, quant_type);
    
    // TODO: Implement actual model quantization
    // This requires loading binary model, applying quantization, and saving
    
    log_warning("Model quantization not yet fully implemented");
    return 0;
}

int quant_get_metadata(
    const char* model_path,
    quant_metadata_t* metadata) {
    
    if (!model_path || !metadata) {
        log_error("Invalid parameters");
        return -1;
    }
    
    memset(metadata, 0, sizeof(quant_metadata_t));
    strncpy(metadata->model_name, model_path, sizeof(metadata->model_name) - 1);
    
    // TODO: Load actual metadata from model
    
    return 0;
}

int quant_dynamic_quantize_model(
    const char* input,
    const char* output,
    quantization_type_t quant_type) {
    
    log_info("Performing dynamic quantization: %s -> %s", input, output);
    
    // TODO: Implement dynamic quantization without calibration
    
    return 0;
}

/* ============================================
   Accuracy Evaluation
   ============================================ */

int quant_evaluate_accuracy(
    const float* original_output,
    const float* quantized_output,
    size_t size,
    float* mae_out,
    float* rmse_out) {
    
    if (!original_output || !quantized_output || !mae_out || !rmse_out) {
        log_error("Invalid parameters for accuracy evaluation");
        return -1;
    }
    
    float mae = 0.0f;
    float mse = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float diff = fabsf(original_output[i] - quantized_output[i]);
        mae += diff;
        mse += diff * diff;
    }
    
    mae /= (float)size;
    mse /= (float)size;
    
    *mae_out = mae;
    *rmse_out = sqrtf(mse);
    
    log_info("Accuracy metrics: MAE=%.6f, RMSE=%.6f", mae, sqrtf(mse));
    
    return 0;
}
