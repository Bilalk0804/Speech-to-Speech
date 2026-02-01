#ifndef ASR_H
#define ASR_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Kaldi/Vosk-style ASR Context
 * Real-time streaming speech recognition
 */
typedef struct asr_context_s asr_context_t;

/**
 * Feature extraction configuration
 * Mimics Kaldi MFCC/fbank extraction
 */
typedef struct {
    uint32_t sample_rate;           // Audio sample rate (16000 Hz typical)
    uint32_t num_mel_bins;          // Number of mel-frequency bins (40 typical)
    uint32_t frame_length_ms;       // Frame length in milliseconds (25 ms)
    uint32_t frame_shift_ms;        // Frame shift in milliseconds (10 ms)
    uint32_t num_cepstral_coeffs;   // Number of MFCC coefficients (13 typical)
    float energy_floor;             // Minimum energy level
} asr_feature_config_t;

/**
 * Segmenter configuration for streaming
 * Detects speech boundaries
 */
typedef struct {
    float vad_threshold;            // Voice activity detection threshold
    uint32_t silence_duration_ms;   // Silence duration for utterance end
    uint32_t min_utterance_duration_ms;  // Minimum utterance length
    uint32_t max_utterance_duration_ms;  // Maximum utterance length
} asr_segmenter_config_t;

// Create ASR context with model path
asr_context_t* asr_create(const char* model_path);

// Destroy ASR context
void asr_destroy(asr_context_t* ctx);

// Set feature extraction configuration
int asr_set_feature_config(asr_context_t* ctx, const asr_feature_config_t* config);

// Set segmenter configuration
int asr_set_segmenter_config(asr_context_t* ctx, const asr_segmenter_config_t* config);

/**
 * Process audio chunk for streaming recognition
 * @param ctx ASR context
 * @param audio Raw audio samples (float, normalized to [-1, 1])
 * @param num_samples Number of samples in this chunk
 * @param output_text Recognition result (can be partial)
 * @param is_final Whether this chunk ends an utterance
 * @return 0 on success, <0 on error, >0 if new result available
 */
int asr_process(asr_context_t* ctx, const float* audio, int num_samples, 
                char* output_text, int* is_final);

/**
 * Reset ASR state for next utterance
 */
void asr_reset(asr_context_t* ctx);

/**
 * Flush buffered audio and get final result
 * @param output_text Final recognition result
 * @return 0 on success
 */
int asr_flush(asr_context_t* ctx, char* output_text);

/**
 * Get confidence score for last recognition
 * @return Confidence score [0, 1]
 */
float asr_get_confidence(asr_context_t* ctx);

/**
 * Get segmenter state (1=speaking, 0=silence)
 */
int asr_get_segmenter_state(asr_context_t* ctx);

#ifdef __cplusplus
}

namespace s2s {

/**
 * Feature Extraction Module
 * Kaldi-compatible MFCC and Mel-Filterbank extraction
 */
class FeatureExtractor {
public:
    FeatureExtractor(const asr_feature_config_t& config);
    ~FeatureExtractor();

    /**
     * Extract features from audio frame
     * @param audio_frame [sample_rate * frame_length_ms / 1000] samples
     * @param features Output feature vector [num_mel_bins or num_cepstral_coeffs]
     */
    int extract_mfcc(const float* audio_frame, float* features);
    int extract_mel_spectrogram(const float* audio_frame, float* features);

    /**
     * Add delta (first derivative) features
     * Used to capture spectral dynamics
     */
    int add_delta_features(const float* features, float* delta_features);

    uint32_t get_num_features() const { return config.num_cepstral_coeffs; }

private:
    asr_feature_config_t config;
    float* mel_filterbank;
    uint32_t num_filters;
    
    int compute_mel_filterbank();
    int fft_forward(const float* input, float* real_part, float* imag_part);
    float mel_scale(float freq_hz);
};

/**
 * Voice Activity Detection & Segmentation
 * Vosk-style streaming segmenter
 */
class StreamingSegmenter {
public:
    StreamingSegmenter(const asr_segmenter_config_t& config);
    ~StreamingSegmenter();

    /**
     * Process audio frame and detect boundaries
     * @return 1 if speech detected, 0 if silence
     */
    int process_frame(const float* audio_frame, uint32_t frame_len);

    /**
     * Check if utterance is complete
     * @return 1 if utterance ended, 0 otherwise
     */
    int is_utterance_complete();

    /**
     * Reset segmenter state
     */
    void reset();

    /**
     * Get current energy level
     */
    float get_energy_level() const;

private:
    asr_segmenter_config_t config;
    float current_energy;
    uint32_t silence_frames;
    uint32_t utterance_frames;
    int in_speech;
    
    float compute_energy(const float* frame, uint32_t frame_len);
};

/**
 * Acoustic Model Interface
 * Wrapper for neural network inference
 */
class AcousticModel {
public:
    AcousticModel(const char* model_path);
    ~AcousticModel();

    /**
     * Forward pass through acoustic model
     * @param features Input features [num_mel_bins]
     * @param logits Output logits [num_output_classes]
     * @param num_output_classes Number of output classes (phones + blank)
     */
    int forward(const float* features, float* logits, uint32_t num_output_classes);

    /**
     * Get model state (for RNN-based models)
     */
    float* get_state();
    void reset_state();

private:
    bool model_loaded;
};

/**
 * CTC Decoder (Connectionist Temporal Classification)
 * Decodes acoustic model outputs to text
 */
class CTCDecoder {
public:
    CTCDecoder();
    ~CTCDecoder();

    /**
     * Decode logit sequence to text
     * @param logits Sequence of logits [time_steps x num_classes]
     * @param time_steps Number of time steps
     * @param num_classes Number of output classes
     * @param output_text Decoded text result
     */
    int decode(const float* logits, uint32_t time_steps, uint32_t num_classes,
               char* output_text);

    /**
     * Greedy CTC decoding (fastest)
     */
    int greedy_decode(const float* logits, uint32_t time_steps, uint32_t num_classes,
                      char* output_text);

    /**
     * Beam search CTC decoding (more accurate)
     */
    int beam_search_decode(const float* logits, uint32_t time_steps, uint32_t num_classes,
                          char* output_text, uint32_t beam_width = 10);

private:
    char* class_to_char(uint32_t class_idx);
};

} // namespace s2s

#endif // __cplusplus

#endif // ASR_H
