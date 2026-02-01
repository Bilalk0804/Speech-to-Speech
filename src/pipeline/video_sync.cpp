#include "../../include/s2s/video_sync.h"
#include "../../include/s2s/logger.h"
#include <math.h>
#include <string.h>

/* ============================================
   Initialization
   ============================================ */

video_sync_context_t* vsync_create(uint32_t video_fps,
                                   uint32_t audio_sr,
                                   uint32_t buffer_size) {
    if (video_fps == 0 || audio_sr == 0 || buffer_size == 0) {
        log_error("Invalid sync parameters: fps=%u, sr=%u, buf=%u",
                 video_fps, audio_sr, buffer_size);
        return NULL;
    }
    
    video_sync_context_t* ctx = malloc(sizeof(video_sync_context_t));
    if (!ctx) {
        log_error("Failed to allocate sync context");
        return NULL;
    }
    
    memset(ctx, 0, sizeof(video_sync_context_t));
    
    ctx->video_fps = video_fps;
    ctx->audio_sample_rate = audio_sr;
    ctx->target_latency_ms = 40.0f;  // ~1-2 frames at 24fps
    ctx->sync_tolerance_ms = 20.0f;  // ±20ms tolerance
    
    // Allocate video buffer
    ctx->video_buffer_size = buffer_size;
    ctx->video_buffer = malloc(buffer_size * sizeof(video_frame_t));
    if (!ctx->video_buffer) {
        log_error("Failed to allocate video buffer");
        free(ctx);
        return NULL;
    }
    
    // Allocate audio buffer
    ctx->audio_buffer_size = buffer_size * audio_sr / video_fps;
    ctx->audio_buffer = malloc(buffer_size * sizeof(audio_sample_t));
    if (!ctx->audio_buffer) {
        log_error("Failed to allocate audio buffer");
        free(ctx->video_buffer);
        free(ctx);
        return NULL;
    }
    
    // Allocate feature arrays
    ctx->lip_features = malloc(buffer_size * sizeof(lip_analysis_t));
    ctx->audio_features = malloc(buffer_size * sizeof(audio_analysis_t));
    
    log_info("Video sync context created (fps=%u, sr=%u, buf=%u)",
            video_fps, audio_sr, buffer_size);
    
    return ctx;
}

void vsync_destroy(video_sync_context_t* ctx) {
    if (!ctx) return;
    
    if (ctx->video_buffer) free(ctx->video_buffer);
    if (ctx->audio_buffer) free(ctx->audio_buffer);
    if (ctx->lip_features) free(ctx->lip_features);
    if (ctx->audio_features) free(ctx->audio_features);
    
    free(ctx);
    log_debug("Video sync context destroyed");
}

void vsync_set_latency(video_sync_context_t* ctx,
                      float latency_ms, float tolerance_ms) {
    if (!ctx) return;
    
    ctx->target_latency_ms = latency_ms;
    ctx->sync_tolerance_ms = tolerance_ms;
    
    log_info("Sync latency set: target=%.1f ms, tolerance=%.1f ms",
            latency_ms, tolerance_ms);
}

/* ============================================
   Buffering
   ============================================ */

int vsync_add_video_frame(video_sync_context_t* ctx,
                         const video_frame_t* frame) {
    if (!ctx || !frame) {
        log_error("Invalid video frame parameters");
        return -1;
    }
    
    // Check if buffer is full
    uint32_t next_pos = (ctx->video_write_pos + 1) % ctx->video_buffer_size;
    if (next_pos == ctx->video_read_pos) {
        log_warning("Video buffer full, dropping oldest frame");
        ctx->video_read_pos = (ctx->video_read_pos + 1) % ctx->video_buffer_size;
    }
    
    // Copy frame
    memcpy(&ctx->video_buffer[ctx->video_write_pos], frame, sizeof(video_frame_t));
    ctx->video_write_pos = next_pos;
    
    return 0;
}

int vsync_add_audio_samples(video_sync_context_t* ctx,
                           const audio_sample_t* samples) {
    if (!ctx || !samples) {
        log_error("Invalid audio samples parameters");
        return -1;
    }
    
    uint32_t next_pos = (ctx->audio_write_pos + 1) % ctx->audio_buffer_size;
    if (next_pos == ctx->audio_read_pos) {
        log_warning("Audio buffer full, dropping oldest samples");
        ctx->audio_read_pos = (ctx->audio_read_pos + 1) % ctx->audio_buffer_size;
    }
    
    memcpy(&ctx->audio_buffer[ctx->audio_write_pos], samples, sizeof(audio_sample_t));
    ctx->audio_write_pos = next_pos;
    
    return 0;
}

uint32_t vsync_video_buffer_count(video_sync_context_t* ctx) {
    if (!ctx) return 0;
    
    if (ctx->video_write_pos >= ctx->video_read_pos) {
        return ctx->video_write_pos - ctx->video_read_pos;
    } else {
        return ctx->video_buffer_size - ctx->video_read_pos + ctx->video_write_pos;
    }
}

uint32_t vsync_audio_buffer_count(video_sync_context_t* ctx) {
    if (!ctx) return 0;
    
    if (ctx->audio_write_pos >= ctx->audio_read_pos) {
        return ctx->audio_write_pos - ctx->audio_read_pos;
    } else {
        return ctx->audio_buffer_size - ctx->audio_read_pos + ctx->audio_write_pos;
    }
}

/* ============================================
   Feature Extraction
   ============================================ */

int vsync_extract_lip_features(const video_frame_t* frame,
                              lip_analysis_t* analysis) {
    if (!frame || !analysis) {
        log_error("Invalid lip feature parameters");
        return -1;
    }
    
    memset(analysis, 0, sizeof(lip_analysis_t));
    
    // Placeholder: Simple feature extraction
    // In real implementation, would use face detection (MediaPipe, dlib, etc.)
    
    analysis->mouth_openness = 0.3f;  // Example value
    analysis->mouth_width = 0.5f;
    analysis->jaw_angle = 15.0f;
    analysis->lip_distance = 25.0f;
    analysis->num_faces = 1;
    analysis->confidence = 0.85f;
    
    return 0;
}

int vsync_extract_audio_features(const audio_sample_t* samples,
                                audio_analysis_t* analysis) {
    if (!samples || !analysis) {
        log_error("Invalid audio feature parameters");
        return -1;
    }
    
    memset(analysis, 0, sizeof(audio_analysis_t));
    
    if (samples->num_samples == 0) {
        return -1;
    }
    
    // Calculate energy
    float energy = 0.0f;
    for (uint32_t i = 0; i < samples->num_samples; i++) {
        energy += samples->samples[i] * samples->samples[i];
    }
    analysis->energy = sqrtf(energy / samples->num_samples);
    
    // Calculate zero crossing rate
    int zero_crossings = 0;
    for (uint32_t i = 1; i < samples->num_samples; i++) {
        if ((samples->samples[i] >= 0 && samples->samples[i-1] < 0) ||
            (samples->samples[i] < 0 && samples->samples[i-1] >= 0)) {
            zero_crossings++;
        }
    }
    analysis->zero_crossing_rate = (float)zero_crossings / samples->num_samples;
    
    // Estimate voicing
    analysis->voicing_confidence = (analysis->energy > 0.01f) ? 0.8f : 0.2f;
    
    // Placeholder MFCC (would need proper implementation)
    for (int i = 0; i < 13; i++) {
        analysis->mfcc[i] = analysis->energy * cosf(i * 3.14159f / 13.0f);
    }
    
    return 0;
}

float vsync_correlate_features(const lip_analysis_t* lip_features,
                              const audio_analysis_t* audio_features) {
    if (!lip_features || !audio_features) {
        return 0.0f;
    }
    
    // Correlate mouth openness with speech energy
    float lip_motion = lip_features->mouth_openness;
    float speech_energy = audio_features->energy;
    
    // Normalize
    float normalized_lip = fminf(lip_motion, 1.0f);
    float normalized_energy = fminf(speech_energy / 0.2f, 1.0f);
    
    // Compute correlation
    float correlation = 1.0f - fabsf(normalized_lip - normalized_energy);
    
    return correlation;
}

/* ============================================
   Synchronization
   ============================================ */

float vsync_compute_sync_error(video_sync_context_t* ctx) {
    if (!ctx) {
        return 0.0f;
    }
    
    uint32_t video_count = vsync_video_buffer_count(ctx);
    uint32_t audio_count = vsync_audio_buffer_count(ctx);
    
    // Calculate expected audio samples per video frame
    float audio_per_frame = (float)ctx->audio_sample_rate / ctx->video_fps;
    
    // Calculate timing
    float video_duration = video_count * 1000.0f / ctx->video_fps;
    float audio_duration = audio_count * 1000.0f / ctx->audio_sample_rate;
    
    // Sync error (negative = audio behind)
    float sync_error = video_duration - audio_duration;
    ctx->sync_error_ms = sync_error;
    
    return sync_error;
}

int vsync_synchronize(video_sync_context_t* ctx) {
    if (!ctx) {
        log_error("Invalid sync context");
        return -1;
    }
    
    float error = vsync_compute_sync_error(ctx);
    
    if (fabsf(error) <= ctx->sync_tolerance_ms) {
        ctx->is_synced = 1;
        return 0;
    }
    
    ctx->is_synced = 0;
    
    // Adjust by dropping or buffering frames
    if (error > ctx->sync_tolerance_ms) {
        // Audio is behind, drop video frames
        vsync_skip_video_frames(ctx, 1);
        log_debug("Dropping video frame to catch up (error=%.1f ms)", error);
    } else if (error < -ctx->sync_tolerance_ms) {
        // Video is behind, would need to speed up audio or buffer
        log_debug("Audio ahead by %.1f ms", fabsf(error));
    }
    
    return (ctx->is_synced) ? 0 : 1;
}

float vsync_get_video_adjustment(video_sync_context_t* ctx) {
    if (!ctx) return 0.0f;
    
    float error = vsync_compute_sync_error(ctx);
    return fmaxf(-1.0f, fminf(1.0f, error / ctx->sync_tolerance_ms));
}

float vsync_get_audio_adjustment(video_sync_context_t* ctx) {
    if (!ctx) return 0.0f;
    
    float error = vsync_compute_sync_error(ctx);
    return -vsync_get_video_adjustment(ctx);
}

uint8_t vsync_is_synchronized(video_sync_context_t* ctx) {
    if (!ctx) return 0;
    return ctx->is_synced;
}

/* ============================================
   Lip-Sync Specific
   ============================================ */

float vsync_detect_lip_motion(const video_frame_t* current_frame,
                             const video_frame_t* previous_frame) {
    if (!current_frame || !previous_frame) {
        return 0.0f;
    }
    
    // Simple placeholder: compare frame timestamps
    // Real implementation would detect facial landmarks
    uint64_t time_diff = (current_frame->timestamp_ms > previous_frame->timestamp_ms) ?
                        current_frame->timestamp_ms - previous_frame->timestamp_ms : 0;
    
    return fminf((float)time_diff / 1000.0f, 1.0f);
}

float vsync_detect_speech_energy(const audio_sample_t* samples) {
    if (!samples || samples->num_samples == 0) {
        return 0.0f;
    }
    
    float energy = 0.0f;
    for (uint32_t i = 0; i < samples->num_samples; i++) {
        energy += samples->samples[i] * samples->samples[i];
    }
    
    energy = sqrtf(energy / samples->num_samples);
    return fminf(energy / 0.2f, 1.0f);
}

float vsync_align_lip_speech(video_sync_context_t* ctx,
                            float video_motion, float speech_energy) {
    if (!ctx) return 0.0f;
    
    // Alignment is better when lip motion and speech are synchronized
    float diff = fabsf(video_motion - speech_energy);
    float alignment = 1.0f - fminf(diff, 1.0f);
    
    ctx->lip_sync_confidence = alignment;
    return alignment;
}

/* ============================================
   Frame Synchronization
   ============================================ */

int vsync_get_synced_pair(video_sync_context_t* ctx,
                         video_frame_t* video_out,
                         audio_sample_t* audio_out) {
    if (!ctx || !video_out || !audio_out) {
        log_error("Invalid synced pair parameters");
        return 1;
    }
    
    uint32_t video_count = vsync_video_buffer_count(ctx);
    uint32_t audio_count = vsync_audio_buffer_count(ctx);
    
    if (video_count == 0 || audio_count == 0) {
        return 1;  // No data available
    }
    
    // Get video frame
    memcpy(video_out, &ctx->video_buffer[ctx->video_read_pos], sizeof(video_frame_t));
    ctx->video_read_pos = (ctx->video_read_pos + 1) % ctx->video_buffer_size;
    
    // Get audio samples
    memcpy(audio_out, &ctx->audio_buffer[ctx->audio_read_pos], sizeof(audio_sample_t));
    ctx->audio_read_pos = (ctx->audio_read_pos + 1) % ctx->audio_buffer_size;
    
    return 0;
}

uint32_t vsync_skip_video_frames(video_sync_context_t* ctx,
                                uint32_t num_frames) {
    if (!ctx) return 0;
    
    uint32_t available = vsync_video_buffer_count(ctx);
    uint32_t to_skip = (num_frames > available) ? available : num_frames;
    
    for (uint32_t i = 0; i < to_skip; i++) {
        ctx->video_read_pos = (ctx->video_read_pos + 1) % ctx->video_buffer_size;
    }
    
    return to_skip;
}

float* vsync_interpolate_audio(const float* samples,
                              uint32_t input_size,
                              uint32_t output_size) {
    if (!samples || input_size == 0 || output_size == 0) {
        return NULL;
    }
    
    float* output = malloc(output_size * sizeof(float));
    if (!output) {
        log_error("Failed to allocate interpolated audio");
        return NULL;
    }
    
    // Linear interpolation
    for (uint32_t i = 0; i < output_size; i++) {
        float src_pos = (float)i * input_size / output_size;
        uint32_t src_idx = (uint32_t)src_pos;
        float frac = src_pos - src_idx;
        
        if (src_idx + 1 < input_size) {
            output[i] = samples[src_idx] * (1.0f - frac) + 
                       samples[src_idx + 1] * frac;
        } else {
            output[i] = samples[src_idx];
        }
    }
    
    return output;
}

/* ============================================
   Statistics
   ============================================ */

void vsync_get_stats(video_sync_context_t* ctx, sync_stats_t* stats) {
    if (!ctx || !stats) {
        log_error("Invalid stats parameters");
        return;
    }
    
    memset(stats, 0, sizeof(sync_stats_t));
    
    stats->current_sync_error_ms = vsync_compute_sync_error(ctx);
    stats->avg_sync_error_ms = stats->current_sync_error_ms;
    stats->max_sync_error_ms = stats->current_sync_error_ms;
    stats->lip_sync_confidence = ctx->lip_sync_confidence;
    
    log_debug("Sync stats: error=%.1f ms, confidence=%.2f",
             stats->current_sync_error_ms, stats->lip_sync_confidence);
}

void vsync_reset_stats(video_sync_context_t* ctx) {
    if (!ctx) return;
    
    ctx->audio_lead_ms = 0.0f;
    ctx->sync_error_ms = 0.0f;
    ctx->lip_sync_confidence = 0.0f;
    
    log_debug("Sync stats reset");
}

float vsync_get_time_drift(video_sync_context_t* ctx) {
    if (!ctx) return 0.0f;
    return vsync_compute_sync_error(ctx);
}

/* ============================================
   Memory Management
   ============================================ */

video_frame_t* vsync_frame_alloc(uint32_t width, uint32_t height,
                                video_format_t format) {
    video_frame_t* frame = malloc(sizeof(video_frame_t));
    if (!frame) {
        log_error("Failed to allocate video frame");
        return NULL;
    }
    
    // Calculate data size based on format
    uint32_t data_size = width * height * 3;  // Simplified
    if (format == VIDEO_FORMAT_YUV420) {
        data_size = width * height * 3 / 2;
    }
    
    frame->data = malloc(data_size);
    if (!frame->data) {
        log_error("Failed to allocate frame data");
        free(frame);
        return NULL;
    }
    
    frame->width = width;
    frame->height = height;
    frame->format = format;
    frame->timestamp_ms = 0;
    frame->frame_number = 0;
    frame->confidence = 0.0f;
    
    return frame;
}

void vsync_frame_free(video_frame_t* frame) {
    if (!frame) return;
    if (frame->data) free(frame->data);
    free(frame);
}

audio_sample_t* vsync_audio_alloc(uint32_t num_samples) {
    audio_sample_t* samples = malloc(sizeof(audio_sample_t));
    if (!samples) {
        log_error("Failed to allocate audio samples");
        return NULL;
    }
    
    samples->samples = malloc(num_samples * sizeof(float));
    if (!samples->samples) {
        log_error("Failed to allocate audio data");
        free(samples);
        return NULL;
    }
    
    samples->num_samples = num_samples;
    samples->sample_rate = 0;
    samples->timestamp_ms = 0;
    
    return samples;
}

void vsync_audio_free(audio_sample_t* samples) {
    if (!samples) return;
    if (samples->samples) free(samples->samples);
    free(samples);
}
