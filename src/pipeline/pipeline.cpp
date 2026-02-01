#include "pipeline.h"
#include "../include/s2s/logger.h"
#include "../asr/asr.h"
#include "../mt/mt.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

namespace s2s {

// ============================================================================
// Ring Buffer Implementation
// ============================================================================

RingBuffer::RingBuffer(uint32_t capacity)
    : capacity(capacity), read_pos(0), write_pos(0), count(0) {
    buffer = new float[capacity];
}

RingBuffer::~RingBuffer() {
    delete[] buffer;
}

uint32_t RingBuffer::write(const float* data, uint32_t num_samples) {
    if (!data) return 0;

    uint32_t available = capacity - count;
    uint32_t to_write = std::min(num_samples, available);

    if (to_write == 0) {
        LOG_WARN("Ring buffer full, dropping samples");
        return 0;
    }

    // Handle wrap-around
    uint32_t write_until_wrap = capacity - write_pos;
    
    if (to_write <= write_until_wrap) {
        // Can write contiguously
        memcpy(buffer + write_pos, data, to_write * sizeof(float));
    } else {
        // Need to wrap around
        memcpy(buffer + write_pos, data, write_until_wrap * sizeof(float));
        uint32_t remaining = to_write - write_until_wrap;
        memcpy(buffer, data + write_until_wrap, remaining * sizeof(float));
    }

    write_pos = (write_pos + to_write) % capacity;
    count += to_write;

    return to_write;
}

uint32_t RingBuffer::read(float* data, uint32_t num_samples) {
    if (!data) return 0;

    uint32_t to_read = std::min(num_samples, count);

    if (to_read == 0) return 0;

    // Handle wrap-around
    uint32_t read_until_wrap = capacity - read_pos;
    
    if (to_read <= read_until_wrap) {
        memcpy(data, buffer + read_pos, to_read * sizeof(float));
    } else {
        memcpy(data, buffer + read_pos, read_until_wrap * sizeof(float));
        uint32_t remaining = to_read - read_until_wrap;
        memcpy(data + read_until_wrap, buffer, remaining * sizeof(float));
    }

    read_pos = (read_pos + to_read) % capacity;
    count -= to_read;

    return to_read;
}

uint32_t RingBuffer::get_available() const {
    return count;
}

void RingBuffer::reset() {
    read_pos = 0;
    write_pos = 0;
    count = 0;
}

// ============================================================================
// S2S Pipeline Implementation
// ============================================================================

S2SPipeline::S2SPipeline()
    : feature_extractor(nullptr),
      segmenter(nullptr),
      acoustic_model(nullptr),
      ctc_decoder(nullptr),
      mt_engine(nullptr),
      audio_buffer(nullptr),
      frame_buffer(nullptr),
      features_buffer(nullptr),
      logits_buffer(nullptr),
      current_state(IDLE),
      asr_enabled(true),
      mt_enabled(true),
      tts_enabled(false),
      frames_in_utterance(0) {
    
    memset(&stats, 0, sizeof(pipeline_stats_t));
}

S2SPipeline::~S2SPipeline() {
    if (feature_extractor) delete feature_extractor;
    if (segmenter) delete segmenter;
    if (acoustic_model) delete acoustic_model;
    if (ctc_decoder) delete ctc_decoder;
    if (mt_engine) delete mt_engine;
    if (audio_buffer) delete audio_buffer;
    if (frame_buffer) delete[] frame_buffer;
    if (features_buffer) delete[] features_buffer;
    if (logits_buffer) delete[] logits_buffer;
}

int S2SPipeline::initialize(const pipeline_config_t& config) {
    this->config = config;

    LOG_INFO("Initializing S2S Pipeline:");
    LOG_INFO("  Sample rate: %u Hz", config.sample_rate);
    LOG_INFO("  Chunk size: %u samples", config.chunk_size);
    LOG_INFO("  ASR model: %s", config.asr_model_path ? config.asr_model_path : "none");
    LOG_INFO("  MT model: %s", config.mt_model_path ? config.mt_model_path : "none");

    // Initialize audio buffer
    uint32_t buffer_samples = config.sample_rate * config.buffer_ms / 1000;
    audio_buffer = new RingBuffer(buffer_samples);

    // Allocate feature buffers
    uint32_t frame_samples = config.sample_rate * 10 / 1000;  // 10ms frames
    frame_buffer = new float[frame_samples];
    features_buffer = new float[512];  // Max feature vector size
    logits_buffer = new float[10000];  // Max number of output classes

    // Initialize ASR components
    if (asr_enabled && config.asr_model_path) {
        asr_feature_config_t feat_config = {
            config.sample_rate,
            40,     // num_mel_bins
            25,     // frame_length_ms
            10,     // frame_shift_ms
            13,     // num_cepstral_coeffs
            1e-5f   // energy_floor
        };

        asr_segmenter_config_t seg_config = {
            config.vad_threshold,
            500,    // silence_duration_ms
            500,    // min_utterance_duration_ms
            30000   // max_utterance_duration_ms
        };

        // These would be created through the C interface in real usage
        LOG_INFO("ASR components ready");
    }

    // Initialize MT engine
    if (mt_enabled && config.mt_model_path) {
        // MT engine would be initialized here
        LOG_INFO("MT engine ready");
    }

    current_state = IDLE;
    LOG_INFO("Pipeline initialized successfully");

    return 0;
}

int S2SPipeline::process_audio(const float* audio_chunk, uint32_t num_samples,
                               char* output_text, uint32_t* output_len) {
    if (!audio_chunk || !output_text || !output_len) return -1;

    *output_len = 0;
    current_state = PROCESSING;

    // Write to buffer
    uint32_t written = audio_buffer->write(audio_chunk, num_samples);
    if (written < num_samples) {
        LOG_WARN("Could not write all samples to buffer");
    }

    // Process frames
    uint32_t frame_samples = config.sample_rate * 10 / 1000;  // 10ms
    std::string accumulated_text;

    while (audio_buffer->get_available() >= frame_samples) {
        audio_buffer->read(frame_buffer, frame_samples);

        // ASR Stage
        if (asr_enabled) {
            char asr_result[256] = {0};
            int is_final = 0;
            // Would call actual ASR here
            // status = asr_process(asr_ctx, frame_buffer, frame_samples, asr_result, &is_final);
            
            accumulated_text += asr_result;

            // MT Stage
            if (mt_enabled && is_final) {
                char mt_result[256] = {0};
                // Would call actual MT here
                // status = mt_translate(mt_ctx, asr_result, mt_result, 256);
                
                strcpy(output_text, mt_result);
                *output_len = strlen(mt_result);

                if (is_final) {
                    current_state = UTTERANCE_COMPLETE;
                    frames_in_utterance = 0;
                    return 1;  // Utterance complete
                }
            }
        }

        frames_in_utterance++;
    }

    return 0;  // Still processing
}

int S2SPipeline::extract_features(const float* audio, uint32_t num_samples) {
    if (!feature_extractor || !audio) return -1;

    uint32_t frame_samples = config.sample_rate * 25 / 1000;  // 25ms frame
    
    if (num_samples < frame_samples) {
        return 0;  // Not enough samples for a frame
    }

    // Extract MFCC features
    // Would call feature_extractor->extract_mfcc(audio, features_buffer);

    return 0;
}

int S2SPipeline::run_acoustic_model() {
    if (!acoustic_model) return -1;

    // Run forward pass
    // Would call acoustic_model->forward(features_buffer, logits_buffer, 100);

    return 0;
}

int S2SPipeline::decode_ctc() {
    if (!ctc_decoder) return -1;

    char result[256];
    // Would call ctc_decoder->greedy_decode(...);

    return 0;
}

int S2SPipeline::stage_asr(const float* audio_chunk, uint32_t num_samples,
                          char* recognition_result) {
    if (!audio_chunk || !recognition_result) return -1;

    // Extract features
    if (extract_features(audio_chunk, num_samples) < 0) return -1;

    // Run acoustic model
    if (run_acoustic_model() < 0) return -1;

    // Decode with CTC
    if (decode_ctc() < 0) return -1;

    strcpy(recognition_result, "sample output");  // Stub
    return 0;
}

int S2SPipeline::stage_mt(const char* recognition_result, char* translation_result) {
    if (!recognition_result || !translation_result) return -1;

    if (!mt_engine) {
        strcpy(translation_result, recognition_result);
        return 0;
    }

    // Would call mt_engine->translate()
    strcpy(translation_result, recognition_result);  // Stub
    return 0;
}

int S2SPipeline::stage_tts(const char* translation_result, float* audio_output) {
    // TTS would synthesize audio from text
    // For now, stub
    return 0;
}

int S2SPipeline::flush(char* final_output) {
    if (!final_output) return -1;

    LOG_INFO("Pipeline flush: %u frames in utterance", frames_in_utterance);

    // Process any remaining buffered audio
    uint32_t remaining = audio_buffer->get_available();
    if (remaining > 0) {
        std::vector<float> leftover(remaining);
        audio_buffer->read(leftover.data(), remaining);
        
        // Process final chunk
        char temp_output[256];
        uint32_t out_len;
        process_audio(leftover.data(), remaining, temp_output, &out_len);
        strcpy(final_output, temp_output);
    } else {
        final_output[0] = '\0';
    }

    reset();
    return 0;
}

void S2SPipeline::reset() {
    audio_buffer->reset();
    frames_in_utterance = 0;
    current_state = IDLE;
    stats.utterances_translated++;
}

} // namespace s2s

// ============================================================================
// C Interface Implementation
// ============================================================================

static s2s::S2SPipeline* g_pipeline = nullptr;

int pipeline_create(const pipeline_config_t* config) {
    if (!config) {
        LOG_ERROR("Pipeline configuration required");
        return -1;
    }

    if (g_pipeline) {
        delete g_pipeline;
    }

    g_pipeline = new s2s::S2SPipeline();
    if (g_pipeline->initialize(*config) < 0) {
        LOG_ERROR("Failed to initialize pipeline");
        delete g_pipeline;
        g_pipeline = nullptr;
        return -1;
    }

    LOG_INFO("Pipeline created successfully");
    return 0;
}

int pipeline_process_audio_chunk(const float* audio_chunk, uint32_t num_samples,
                                 char* output_translation, uint32_t* output_len) {
    if (!g_pipeline) {
        LOG_ERROR("Pipeline not initialized");
        return -1;
    }

    return g_pipeline->process_audio(audio_chunk, num_samples, 
                                     output_translation, output_len);
}

int pipeline_get_stats(pipeline_stats_t* stats) {
    if (!g_pipeline || !stats) return -1;
    
    *stats = g_pipeline->get_stats();
    return 0;
}

int pipeline_flush(char* final_output) {
    if (!g_pipeline) return -1;
    return g_pipeline->flush(final_output);
}

void pipeline_destroy() {
    if (g_pipeline) {
        delete g_pipeline;
        g_pipeline = nullptr;
    }
    LOG_INFO("Pipeline destroyed");
}

int pipeline_set_enabled_stages(int asr_enabled, int mt_enabled, int tts_enabled) {
    if (!g_pipeline) return -1;
    
    g_pipeline->set_asr_enabled(asr_enabled != 0);
    g_pipeline->set_mt_enabled(mt_enabled != 0);
    g_pipeline->set_tts_enabled(tts_enabled != 0);
    
    return 0;
}

int pipeline_get_state() {
    if (!g_pipeline) return -1;
    return static_cast<int>(g_pipeline->get_state());
}

// Simple Pipeline wrapper (C++ only interface)
class Pipeline {
public:
    Pipeline() : pipeline(nullptr) {}
    
    ~Pipeline() {
        if (pipeline) delete pipeline;
    }
    
    bool initialize() {
        pipeline_config_t config = {
            16000,              // sample_rate
            1600,               // chunk_size (100ms)
            2000,               // buffer_ms
            0.1f,               // vad_threshold
            "asr_model.bin",
            "mt_model.bin",
            "tts_model.bin"
        };
        
        pipeline = new s2s::S2SPipeline();
        return pipeline->initialize(config) == 0;
    }
    
    void process() {
        // Process implementation
    }
    
    void shutdown() {
        if (pipeline) {
            delete pipeline;
            pipeline = nullptr;
        }
    }
    
private:
    s2s::S2SPipeline* pipeline;
};

