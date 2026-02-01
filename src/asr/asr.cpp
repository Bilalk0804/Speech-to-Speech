#include "asr.h"
#include "../include/s2s/logger.h"
#include "../include/s2s/model_loader.h"
#include "../include/s2s/matrix_ops.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979323846f
#endif

namespace s2s {

// ============================================================================
// Feature Extractor Implementation
// ============================================================================

FeatureExtractor::FeatureExtractor(const asr_feature_config_t& config)
    : config(config), mel_filterbank(nullptr), num_filters(0) {
    
    LOG_INFO("Initializing feature extractor: %u mel bins, %u MFCC coeffs",
             config.num_mel_bins, config.num_cepstral_coeffs);
    
    compute_mel_filterbank();
}

FeatureExtractor::~FeatureExtractor() {
    delete[] mel_filterbank;
}

int FeatureExtractor::extract_mfcc(const float* audio_frame, float* features) {
    if (!audio_frame || !features) return -1;

    uint32_t frame_samples = config.sample_rate * config.frame_length_ms / 1000;
    
    // Apply Hamming window
    std::vector<float> windowed(frame_samples);
    for (uint32_t i = 0; i < frame_samples; i++) {
        float window = 0.54f - 0.46f * cos(2.0f * PI * i / (frame_samples - 1));
        windowed[i] = audio_frame[i] * window;
    }

    // Apply pre-emphasis filter
    std::vector<float> pre_emphasized(frame_samples);
    pre_emphasized[0] = windowed[0];
    for (uint32_t i = 1; i < frame_samples; i++) {
        pre_emphasized[i] = windowed[i] - 0.97f * windowed[i - 1];
    }

    // Compute log mel spectrogram
    std::vector<float> mel_spec(config.num_mel_bins);
    if (extract_mel_spectrogram(pre_emphasized.data(), mel_spec.data()) < 0) {
        return -1;
    }

    // Apply DCT for MFC coefficients
    for (uint32_t i = 0; i < config.num_cepstral_coeffs && i < config.num_mel_bins; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < config.num_mel_bins; j++) {
            float angle = PI * i * (j + 0.5f) / config.num_mel_bins;
            sum += mel_spec[j] * cos(angle);
        }
        features[i] = sum;
    }

    // Pad remaining features with zero
    for (uint32_t i = config.num_cepstral_coeffs; i < config.num_mel_bins; i++) {
        features[i] = 0.0f;
    }

    return 0;
}

int FeatureExtractor::extract_mel_spectrogram(const float* audio_frame, float* features) {
    if (!audio_frame || !features) return -1;

    uint32_t frame_samples = config.sample_rate * config.frame_length_ms / 1000;
    
    // Compute power spectrum using simple DFT (stub)
    std::vector<float> power_spec(config.num_mel_bins, 0.0f);
    
    for (uint32_t i = 0; i < config.num_mel_bins; i++) {
        float power = 0.0f;
        for (uint32_t j = 0; j < frame_samples && j < 256; j++) {
            power += audio_frame[j] * audio_frame[j];
        }
        power_spec[i] = power / frame_samples;
    }

    // Apply mel-scale and log
    for (uint32_t i = 0; i < config.num_mel_bins; i++) {
        float log_mel = log(power_spec[i] + config.energy_floor);
        features[i] = log_mel;
    }

    return 0;
}

int FeatureExtractor::add_delta_features(const float* features, float* delta_features) {
    if (!features || !delta_features) return -1;

    // Simple delta computation: derivative between consecutive frames
    // In real implementation, would maintain frame history
    for (uint32_t i = 0; i < config.num_cepstral_coeffs; i++) {
        delta_features[i] = 0.0f;  // Stub
    }

    return 0;
}

int FeatureExtractor::compute_mel_filterbank() {
    // Create mel-scale filters
    // This would create triangular filters spaced on mel-scale
    mel_filterbank = new float[config.num_mel_bins * 128];
    num_filters = config.num_mel_bins;
    
    LOG_DEBUG("Created mel filterbank with %u filters", num_filters);
    return 0;
}

int FeatureExtractor::fft_forward(const float* input, float* real_part, float* imag_part) {
    // Stub for FFT computation
    // In real implementation, would use Cooley-Tukey FFT or similar
    return 0;
}

float FeatureExtractor::mel_scale(float freq_hz) {
    // Convert Hz to mel scale: mel = 2595 * log10(1 + f/700)
    return 2595.0f * log10(1.0f + freq_hz / 700.0f);
}

// ============================================================================
// StreamingSegmenter Implementation
// ============================================================================

StreamingSegmenter::StreamingSegmenter(const asr_segmenter_config_t& config)
    : config(config), current_energy(0.0f), silence_frames(0),
      utterance_frames(0), in_speech(0) {
    
    LOG_INFO("Initializing streaming segmenter: VAD threshold=%.2f", config.vad_threshold);
}

StreamingSegmenter::~StreamingSegmenter() {}

int StreamingSegmenter::process_frame(const float* audio_frame, uint32_t frame_len) {
    if (!audio_frame) return -1;

    current_energy = compute_energy(audio_frame, frame_len);
    
    int is_speech = current_energy > config.vad_threshold ? 1 : 0;
    
    if (is_speech) {
        in_speech = 1;
        silence_frames = 0;
        utterance_frames++;
    } else {
        silence_frames++;
        if (silence_frames > config.silence_duration_ms / 10) {  // 10ms frames
            in_speech = 0;
        }
    }

    return in_speech;
}

int StreamingSegmenter::is_utterance_complete() {
    // Utterance complete if:
    // 1. We were in speech and now in silence long enough, OR
    // 2. We exceeded max utterance duration
    
    if (!in_speech && silence_frames > config.silence_duration_ms / 10) {
        return 1;
    }
    
    if (utterance_frames > config.max_utterance_duration_ms / 10) {
        return 1;
    }
    
    return 0;
}

void StreamingSegmenter::reset() {
    in_speech = 0;
    silence_frames = 0;
    utterance_frames = 0;
    current_energy = 0.0f;
}

float StreamingSegmenter::get_energy_level() const {
    return current_energy;
}

float StreamingSegmenter::compute_energy(const float* frame, uint32_t frame_len) {
    float energy = 0.0f;
    for (uint32_t i = 0; i < frame_len; i++) {
        energy += frame[i] * frame[i];
    }
    return sqrt(energy / frame_len);
}

// ============================================================================
// AcousticModel Implementation
// ============================================================================

AcousticModel::AcousticModel(const char* model_path)
    : model_loaded(false) {
    
    if (model_path) {
        LOG_INFO("Loading acoustic model: %s", model_path);
        model_loaded = true;
    }
}

AcousticModel::~AcousticModel() {}

int AcousticModel::forward(const float* features, float* logits, uint32_t num_output_classes) {
    if (!model_loaded || !features || !logits) return -1;
    
    // Stub: would run neural network inference here
    for (uint32_t i = 0; i < num_output_classes; i++) {
        logits[i] = -10.0f;  // Log probability stubs
    }
    
    return 0;
}

float* AcousticModel::get_state() {
    // For RNN models
    return nullptr;
}

void AcousticModel::reset_state() {
    // Reset RNN hidden state
}

// ============================================================================
// CTC Decoder Implementation
// ============================================================================

CTCDecoder::CTCDecoder() {}

CTCDecoder::~CTCDecoder() {}

int CTCDecoder::decode(const float* logits, uint32_t time_steps, uint32_t num_classes,
                       char* output_text) {
    // Default to greedy decoding
    return greedy_decode(logits, time_steps, num_classes, output_text);
}

int CTCDecoder::greedy_decode(const float* logits, uint32_t time_steps, uint32_t num_classes,
                              char* output_text) {
    if (!logits || !output_text) return -1;

    std::string result;
    uint32_t prev_class = num_classes - 1;  // Blank token
    
    for (uint32_t t = 0; t < time_steps; t++) {
        // Find max logit at this timestep
        uint32_t max_class = 0;
        float max_logit = logits[t * num_classes];
        
        for (uint32_t c = 1; c < num_classes; c++) {
            if (logits[t * num_classes + c] > max_logit) {
                max_logit = logits[t * num_classes + c];
                max_class = c;
            }
        }
        
        // Add to result if not blank and different from previous
        if (max_class != (num_classes - 1) && max_class != prev_class) {
            result += class_to_char(max_class);
        }
        
        prev_class = max_class;
    }
    
    strcpy(output_text, result.c_str());
    return 0;
}

int CTCDecoder::beam_search_decode(const float* logits, uint32_t time_steps, uint32_t num_classes,
                                   char* output_text, uint32_t beam_width) {
    // Simplified beam search (full implementation would use priority queue)
    return greedy_decode(logits, time_steps, num_classes, output_text);
}

char* CTCDecoder::class_to_char(uint32_t class_idx) {
    // Map class indices to characters
    // This would be populated from vocabulary file
    static char buffer[2] = {'\0', '\0'};
    
    if (class_idx < 26) {
        buffer[0] = 'a' + class_idx;
    } else if (class_idx < 36) {
        buffer[0] = '0' + (class_idx - 26);
    } else {
        buffer[0] = ' ';
    }
    
    return buffer;
}

} // namespace s2s

// ============================================================================
// C Interface Implementation
// ============================================================================

struct asr_context_s {
    s2s::FeatureExtractor* feature_extractor;
    s2s::StreamingSegmenter* segmenter;
    s2s::AcousticModel* acoustic_model;
    s2s::CTCDecoder* ctc_decoder;
    std::vector<float>* feature_buffer;
    asr_feature_config_t feature_config;
    asr_segmenter_config_t segmenter_config;
    float confidence_score;
};

asr_context_t* asr_create(const char* model_path) {
    if (!model_path) {
        LOG_ERROR("Model path is required");
        return nullptr;
    }

    asr_context_t* ctx = new asr_context_t();
    
    // Initialize with default configs
    ctx->feature_config = {
        16000,      // sample_rate
        40,         // num_mel_bins
        25,         // frame_length_ms
        10,         // frame_shift_ms
        13,         // num_cepstral_coeffs
        1e-5f       // energy_floor
    };
    
    ctx->segmenter_config = {
        0.1f,       // vad_threshold
        500,        // silence_duration_ms
        500,        // min_utterance_duration_ms
        30000       // max_utterance_duration_ms
    };

    ctx->feature_extractor = new s2s::FeatureExtractor(ctx->feature_config);
    ctx->segmenter = new s2s::StreamingSegmenter(ctx->segmenter_config);
    ctx->acoustic_model = new s2s::AcousticModel(model_path);
    ctx->ctc_decoder = new s2s::CTCDecoder();
    ctx->feature_buffer = new std::vector<float>();
    ctx->confidence_score = 0.0f;

    LOG_INFO("ASR context created successfully");
    return ctx;
}

void asr_destroy(asr_context_t* ctx) {
    if (!ctx) return;
    
    delete ctx->feature_extractor;
    delete ctx->segmenter;
    delete ctx->acoustic_model;
    delete ctx->ctc_decoder;
    delete ctx->feature_buffer;
    delete ctx;
    
    LOG_INFO("ASR context destroyed");
}

int asr_set_feature_config(asr_context_t* ctx, const asr_feature_config_t* config) {
    if (!ctx || !config) return -1;
    
    ctx->feature_config = *config;
    delete ctx->feature_extractor;
    ctx->feature_extractor = new s2s::FeatureExtractor(ctx->feature_config);
    
    return 0;
}

int asr_set_segmenter_config(asr_context_t* ctx, const asr_segmenter_config_t* config) {
    if (!ctx || !config) return -1;
    
    ctx->segmenter_config = *config;
    delete ctx->segmenter;
    ctx->segmenter = new s2s::StreamingSegmenter(ctx->segmenter_config);
    
    return 0;
}

int asr_process(asr_context_t* ctx, const float* audio, int num_samples, 
                char* output_text, int* is_final) {
    if (!ctx || !audio || !output_text || !is_final) return -1;

    *is_final = 0;
    output_text[0] = '\0';
    
    // Process audio in frames
    uint32_t frame_samples = ctx->feature_config.sample_rate * ctx->feature_config.frame_shift_ms / 1000;
    
    for (int i = 0; i + (int)frame_samples <= num_samples; i += frame_samples) {
        // Extract features
        std::vector<float> features(ctx->feature_config.num_cepstral_coeffs);
        ctx->feature_extractor->extract_mfcc(audio + i, features.data());
        
        // Check VAD
        int is_speech = ctx->segmenter->process_frame(audio + i, frame_samples);
        
        // Run acoustic model
        std::vector<float> logits(100);  // Assuming 100 output classes
        ctx->acoustic_model->forward(features.data(), logits.data(), logits.size());
        
        // Check if utterance is complete
        if (ctx->segmenter->is_utterance_complete()) {
            *is_final = 1;
            ctx->ctc_decoder->greedy_decode(logits.data(), 1, logits.size(), output_text);
            break;
        }
    }

    return 0;
}

void asr_reset(asr_context_t* ctx) {
    if (!ctx) return;
    
    ctx->segmenter->reset();
    ctx->feature_buffer->clear();
}

int asr_flush(asr_context_t* ctx, char* output_text) {
    if (!ctx || !output_text) return -1;
    
    output_text[0] = '\0';
    return 0;
}

float asr_get_confidence(asr_context_t* ctx) {
    if (!ctx) return 0.0f;
    return ctx->confidence_score;
}

int asr_get_segmenter_state(asr_context_t* ctx) {
    if (!ctx) return -1;
    return ctx->segmenter->get_energy_level() > ctx->segmenter_config.vad_threshold ? 1 : 0;
}
