#ifndef S2S_VIDEO_SYNC_H
#define S2S_VIDEO_SYNC_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Video-Audio Synchronization for Lip-Sync in S2S Applications
 */

/* Video Frame Format */
typedef enum {
    VIDEO_FORMAT_RGB = 0,
    VIDEO_FORMAT_BGR = 1,
    VIDEO_FORMAT_YUV420 = 2,
    VIDEO_FORMAT_NV12 = 3
} video_format_t;

/* Video Frame */
typedef struct {
    uint8_t* data;
    uint32_t width;
    uint32_t height;
    video_format_t format;
    uint64_t timestamp_ms;    // Frame timestamp
    uint32_t frame_number;
    float confidence;         // Lip-sync confidence (0-1)
} video_frame_t;

/* Audio Sample */
typedef struct {
    float* samples;
    uint32_t num_samples;
    uint32_t sample_rate;
    uint64_t timestamp_ms;    // Audio chunk timestamp
} audio_sample_t;

/* Lip-Sync Analysis */
typedef struct {
    float mouth_openness;     // 0-1 scale
    float mouth_width;        // 0-1 scale
    float jaw_angle;          // Degrees
    float lip_distance;       // Pixels
    uint32_t num_faces;       // Number of faces detected
    float confidence;         // Detection confidence
} lip_analysis_t;

/* Audio Analysis */
typedef struct {
    float energy;             // Overall energy level
    float spectral_centroid;  // Frequency center
    float zero_crossing_rate; // ZCR
    float mfcc[13];           // MFCC coefficients
    float voicing_confidence; // Likelihood of voice
} audio_analysis_t;

/* Sync Context */
typedef struct {
    uint32_t video_fps;
    uint32_t audio_sample_rate;
    float target_latency_ms;  // Desired sync latency
    float sync_tolerance_ms;  // Acceptable sync window
    
    // Circular buffers
    video_frame_t* video_buffer;
    uint32_t video_buffer_size;
    uint32_t video_write_pos;
    uint32_t video_read_pos;
    
    audio_sample_t* audio_buffer;
    uint32_t audio_buffer_size;
    uint32_t audio_write_pos;
    uint32_t audio_read_pos;
    
    // Sync metrics
    float audio_lead_ms;      // How much audio is ahead (+) or behind (-)
    float sync_error_ms;      // Current synchronization error
    uint8_t is_synced;        // 1 if currently synchronized
    
    // Lip-sync specific
    lip_analysis_t* lip_features;
    audio_analysis_t* audio_features;
    float lip_sync_confidence;
} video_sync_context_t;

/* ============================================
   Initialization & Configuration
   ============================================ */

/**
 * Create video sync context
 * @param video_fps Video frame rate
 * @param audio_sr Audio sample rate
 * @param buffer_size Circular buffer size
 * @return Context or NULL on error
 */
video_sync_context_t* vsync_create(uint32_t video_fps, 
                                   uint32_t audio_sr,
                                   uint32_t buffer_size);

/**
 * Destroy video sync context
 */
void vsync_destroy(video_sync_context_t* ctx);

/**
 * Set target latency and tolerance
 * @param ctx Sync context
 * @param latency_ms Target sync latency (milliseconds)
 * @param tolerance_ms Sync window tolerance
 */
void vsync_set_latency(video_sync_context_t* ctx,
                      float latency_ms, float tolerance_ms);

/* ============================================
   Frame Input & Buffering
   ============================================ */

/**
 * Add video frame to buffer
 * @param ctx Sync context
 * @param frame Video frame
 * @return 0 on success
 */
int vsync_add_video_frame(video_sync_context_t* ctx,
                         const video_frame_t* frame);

/**
 * Add audio samples to buffer
 * @param ctx Sync context
 * @param samples Audio samples
 * @return 0 on success
 */
int vsync_add_audio_samples(video_sync_context_t* ctx,
                           const audio_sample_t* samples);

/**
 * Get video frame count in buffer
 */
uint32_t vsync_video_buffer_count(video_sync_context_t* ctx);

/**
 * Get audio sample count in buffer
 */
uint32_t vsync_audio_buffer_count(video_sync_context_t* ctx);

/* ============================================
   Feature Extraction
   ============================================ */

/**
 * Analyze lip/facial features from video frame
 * @param frame Video frame
 * @param analysis Output lip analysis
 * @return 0 on success
 */
int vsync_extract_lip_features(const video_frame_t* frame,
                              lip_analysis_t* analysis);

/**
 * Analyze audio characteristics
 * @param samples Audio samples
 * @param analysis Output audio analysis
 * @return 0 on success
 */
int vsync_extract_audio_features(const audio_sample_t* samples,
                                audio_analysis_t* analysis);

/**
 * Correlate lip features with audio features
 * @param lip_features Lip analysis
 * @param audio_features Audio analysis
 * @return Correlation score (0-1)
 */
float vsync_correlate_features(const lip_analysis_t* lip_features,
                              const audio_analysis_t* audio_features);

/* ============================================
   Synchronization & Alignment
   ============================================ */

/**
 * Compute current sync error
 * @param ctx Sync context
 * @return Sync error in milliseconds (negative = audio behind)
 */
float vsync_compute_sync_error(video_sync_context_t* ctx);

/**
 * Synchronize video and audio
 * @param ctx Sync context
 * @return 0 if synced, non-zero if adjustment needed
 */
int vsync_synchronize(video_sync_context_t* ctx);

/**
 * Get time adjustment for video frame
 * @param ctx Sync context
 * @return Time adjustment in milliseconds
 */
float vsync_get_video_adjustment(video_sync_context_t* ctx);

/**
 * Get time adjustment for audio
 * @param ctx Sync context
 * @return Time adjustment in milliseconds
 */
float vsync_get_audio_adjustment(video_sync_context_t* ctx);

/**
 * Check if video and audio are synchronized
 * @return 1 if synced, 0 otherwise
 */
uint8_t vsync_is_synchronized(video_sync_context_t* ctx);

/* ============================================
   Lip-Sync Specific
   ============================================ */

/**
 * Detect lip motion
 * @param current_frame Current video frame
 * @param previous_frame Previous video frame
 * @return Lip motion magnitude (0-1)
 */
float vsync_detect_lip_motion(const video_frame_t* current_frame,
                             const video_frame_t* previous_frame);

/**
 * Detect speech energy in audio
 * @param samples Audio samples
 * @return Speech energy level (0-1)
 */
float vsync_detect_speech_energy(const audio_sample_t* samples);

/**
 * Align lip motion with speech
 * @param ctx Sync context
 * @param video_motion Lip motion from video
 * @param speech_energy Speech energy from audio
 * @return Alignment score (0-1, 1=perfect alignment)
 */
float vsync_align_lip_speech(video_sync_context_t* ctx,
                            float video_motion, float speech_energy);

/* ============================================
   Frame Synchronization
   ============================================ */

/**
 * Get next synchronized frame pair
 * @param ctx Sync context
 * @param video_out Output video frame
 * @param audio_out Output audio samples
 * @return 0 on success, 1 if no data available
 */
int vsync_get_synced_pair(video_sync_context_t* ctx,
                         video_frame_t* video_out,
                         audio_sample_t* audio_out);

/**
 * Skip frames for catch-up
 * @param ctx Sync context
 * @param num_frames Number of frames to skip
 * @return Frames actually skipped
 */
uint32_t vsync_skip_video_frames(video_sync_context_t* ctx,
                                uint32_t num_frames);

/**
 * Interpolate audio for frame rate matching
 * @param samples Input audio
 * @param output_size Desired output size
 * @return Interpolated audio
 */
float* vsync_interpolate_audio(const float* samples,
                              uint32_t input_size,
                              uint32_t output_size);

/* ============================================
   Timing & Statistics
   ============================================ */

/**
 * Get synchronization statistics
 */
typedef struct {
    float current_sync_error_ms;
    float avg_sync_error_ms;
    float max_sync_error_ms;
    uint32_t sync_adjustments;
    float lip_sync_confidence;
    uint64_t total_frames_processed;
} sync_stats_t;

/**
 * Get sync statistics
 * @param ctx Sync context
 * @param stats Output statistics
 */
void vsync_get_stats(video_sync_context_t* ctx, sync_stats_t* stats);

/**
 * Reset sync statistics
 */
void vsync_reset_stats(video_sync_context_t* ctx);

/**
 * Get time drift between video and audio
 * @param ctx Sync context
 * @return Drift in milliseconds
 */
float vsync_get_time_drift(video_sync_context_t* ctx);

/* ============================================
   Memory Management
   ============================================ */

/**
 * Allocate video frame
 * @param width Frame width
 * @param height Frame height
 * @param format Video format
 * @return Allocated frame
 */
video_frame_t* vsync_frame_alloc(uint32_t width, uint32_t height,
                                video_format_t format);

/**
 * Free video frame
 */
void vsync_frame_free(video_frame_t* frame);

/**
 * Allocate audio samples
 * @param num_samples Number of samples
 * @return Allocated audio
 */
audio_sample_t* vsync_audio_alloc(uint32_t num_samples);

/**
 * Free audio samples
 */
void vsync_audio_free(audio_sample_t* samples);

#ifdef __cplusplus
}
#endif

#endif // S2S_VIDEO_SYNC_H
