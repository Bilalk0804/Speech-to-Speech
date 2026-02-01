#ifndef S2S_STREAMING_TTS_H
#define S2S_STREAMING_TTS_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Streaming Text-to-Speech (TTS) API
 * Supports real-time synthesis with low latency
 */

/* TTS Configuration */
typedef struct {
    uint32_t sample_rate;        // 22050, 44100, 48000
    uint32_t mel_bins;           // Mel spectrogram channels (80-128)
    float f_min;                 // Min frequency (Hz)
    float f_max;                 // Max frequency (Hz)
    uint32_t n_fft;              // FFT size
    uint32_t hop_length;         // Hop length for mel extraction
    uint32_t win_length;         // Window length
    uint32_t max_batch_size;     // For inference
    char vocoder_model[256];     // Path to vocoder model
    char glow_tts_model[256];    // Path to Glow-TTS model
} tts_config_t;

/* Mel Spectrogram */
typedef struct {
    float* data;                 // [time_steps, mel_bins]
    uint32_t time_steps;
    uint32_t mel_bins;
    float duration_ms;
} mel_spectrogram_t;

/* TTS Context */
typedef struct tts_context_s {
    tts_config_t config;
    void* glow_tts_model;        // Glow-TTS model instance
    void* vocoder_model;         // Vocoder model instance
    void* text_encoder;          // Text tokenizer
    float* mel_buffer;           // Streaming mel buffer
    uint32_t mel_buffer_size;
    uint32_t mel_buffer_pos;
    float* audio_buffer;         // Output audio buffer
    uint32_t audio_buffer_size;
    uint32_t audio_buffer_pos;
    uint8_t is_streaming;        // Streaming mode flag
} tts_context_t;

/* Streaming Audio Chunk */
typedef struct {
    float* audio_data;
    uint32_t num_samples;
    uint32_t sample_rate;
    float duration_ms;
    uint8_t is_final;            // Last chunk indicator
} audio_chunk_t;

/* ============================================
   Initialization & Configuration
   ============================================ */

/**
 * Create TTS context
 * @param config Configuration structure
 * @return TTS context or NULL on error
 */
tts_context_t* tts_create(const tts_config_t* config);

/**
 * Destroy TTS context
 */
void tts_destroy(tts_context_t* ctx);

/**
 * Load models
 * @return 0 on success
 */
int tts_load_models(tts_context_t* ctx);

/* ============================================
   Text Processing
   ============================================ */

/**
 * Tokenize text for synthesis
 * @param ctx TTS context
 * @param text Input text
 * @param tokens Output token IDs
 * @param max_tokens Maximum token count
 * @return Number of tokens
 */
int tts_tokenize(tts_context_t* ctx, const char* text, 
                 int32_t* tokens, uint32_t max_tokens);

/**
 * Estimate duration from tokens
 * @param ctx TTS context
 * @param tokens Token array
 * @param num_tokens Number of tokens
 * @return Estimated duration in milliseconds
 */
float tts_estimate_duration(tts_context_t* ctx, 
                           const int32_t* tokens, uint32_t num_tokens);

/* ============================================
   Mel-Spectrogram Generation
   ============================================ */

/**
 * Generate mel-spectrogram from text tokens
 * @param ctx TTS context
 * @param tokens Token IDs
 * @param num_tokens Number of tokens
 * @param mel Output mel-spectrogram
 * @return 0 on success
 */
int tts_generate_mel(tts_context_t* ctx, 
                     const int32_t* tokens, uint32_t num_tokens,
                     mel_spectrogram_t* mel);

/**
 * Get partial mel-spectrogram for streaming
 * @param ctx TTS context
 * @param mel Output mel chunk
 * @param frame_count Frames to generate
 * @return Actual frames generated
 */
uint32_t tts_get_mel_chunk(tts_context_t* ctx, 
                          mel_spectrogram_t* mel,
                          uint32_t frame_count);

/* ============================================
   Vocoder (Mel → Waveform)
   ============================================ */

/**
 * Generate audio from mel-spectrogram
 * @param ctx TTS context
 * @param mel Input mel-spectrogram
 * @param audio Output audio chunk
 * @return 0 on success
 */
int tts_vocoder_infer(tts_context_t* ctx,
                     const mel_spectrogram_t* mel,
                     audio_chunk_t* audio);

/**
 * Generate audio in streaming mode
 * @param ctx TTS context
 * @param mel_frame Single mel frame (1, mel_bins)
 * @param audio Output audio chunk (hop_length samples)
 * @return 0 on success
 */
int tts_vocoder_streaming(tts_context_t* ctx,
                         const float* mel_frame,
                         audio_chunk_t* audio);

/* ============================================
   Streaming Synthesis
   ============================================ */

/**
 * Start streaming synthesis
 * @param ctx TTS context
 * @param text Input text
 * @return 0 on success
 */
int tts_streaming_start(tts_context_t* ctx, const char* text);

/**
 * Get next audio chunk (non-blocking)
 * @param ctx TTS context
 * @param chunk Output audio chunk
 * @return Number of samples in chunk (0 if not ready)
 */
uint32_t tts_streaming_get_chunk(tts_context_t* ctx, audio_chunk_t* chunk);

/**
 * Finish streaming synthesis
 * @param ctx TTS context
 * @return 0 on success
 */
int tts_streaming_finish(tts_context_t* ctx);

/**
 * Check if synthesis is complete
 * @param ctx TTS context
 * @return 1 if complete, 0 otherwise
 */
uint8_t tts_streaming_is_complete(tts_context_t* ctx);

/* ============================================
   Batch Synthesis
   ============================================ */

/**
 * Synthesize full audio from text
 * @param ctx TTS context
 * @param text Input text
 * @param audio Output audio buffer
 * @param max_samples Maximum output samples
 * @param num_samples Output: actual samples generated
 * @return 0 on success
 */
int tts_synthesize(tts_context_t* ctx,
                  const char* text,
                  float* audio,
                  uint32_t max_samples,
                  uint32_t* num_samples);

/* ============================================
   Audio Processing
   ============================================ */

/**
 * Apply normalization to audio
 * @param audio Audio samples
 * @param num_samples Number of samples
 * @param target_rms Target RMS level (default 0.1)
 */
void tts_normalize_audio(float* audio, uint32_t num_samples, 
                        float target_rms);

/**
 * Apply fade in/out
 * @param audio Audio samples
 * @param num_samples Number of samples
 * @param fade_samples Number of samples to fade
 * @param is_fade_in 1 for fade in, 0 for fade out
 */
void tts_apply_fade(float* audio, uint32_t num_samples,
                   uint32_t fade_samples, uint8_t is_fade_in);

/**
 * Remove silence from audio
 * @param audio Audio samples
 * @param num_samples Number of samples
 * @param threshold Silence threshold (RMS level)
 * @param trimmed_samples Output: samples after trimming
 * @return Trimmed audio or NULL
 */
float* tts_trim_silence(const float* audio, uint32_t num_samples,
                       float threshold, uint32_t* trimmed_samples);

/* ============================================
   Performance & Monitoring
   ============================================ */

/**
 * Get synthesis statistics
 */
typedef struct {
    float total_time_ms;
    float mel_generation_ms;
    float vocoding_ms;
    float audio_processing_ms;
    uint32_t total_samples;
    float real_time_factor;      // Synthesis time / audio duration
} tts_stats_t;

/**
 * Get synthesis statistics
 * @param ctx TTS context
 * @param stats Output statistics
 */
void tts_get_stats(tts_context_t* ctx, tts_stats_t* stats);

/**
 * Reset statistics
 */
void tts_reset_stats(tts_context_t* ctx);

/* ============================================
   Memory Management
   ============================================ */

/**
 * Allocate mel-spectrogram
 * @param time_steps Time dimension
 * @param mel_bins Frequency dimension
 * @return Allocated mel-spectrogram
 */
mel_spectrogram_t* tts_mel_alloc(uint32_t time_steps, uint32_t mel_bins);

/**
 * Free mel-spectrogram
 */
void tts_mel_free(mel_spectrogram_t* mel);

/**
 * Allocate audio chunk
 * @param num_samples Number of samples
 * @return Allocated audio chunk
 */
audio_chunk_t* tts_audio_alloc(uint32_t num_samples);

/**
 * Free audio chunk
 */
void tts_audio_free(audio_chunk_t* chunk);

#ifdef __cplusplus
}
#endif

#endif // S2S_STREAMING_TTS_H
