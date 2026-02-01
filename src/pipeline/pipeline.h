#ifndef PIPELINE_H
#define PIPELINE_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Real-time speech-to-speech pipeline configuration
 */
typedef struct {
    uint32_t sample_rate;       // 16000 Hz typical
    uint32_t chunk_size;        // Audio chunk size (samples)
    uint32_t buffer_ms;         // Buffering duration (ms)
    float vad_threshold;        // Voice activity detection threshold
    const char* asr_model_path;
    const char* mt_model_path;
    const char* tts_model_path;
} pipeline_config_t;

/**
 * Pipeline statistics
 */
typedef struct {
    uint32_t frames_processed;
    uint32_t utterances_translated;
    float avg_latency_ms;
    float throughput_fps;
} pipeline_stats_t;

/**
 * Create and initialize pipeline
 */
int pipeline_create(const pipeline_config_t* config);

/**
 * Process audio stream
 * Streaming API: accept audio chunks and produce translations
 */
int pipeline_process_audio_chunk(const float* audio_chunk, uint32_t num_samples,
                                 char* output_translation, uint32_t* output_len);

/**
 * Get pipeline statistics
 */
int pipeline_get_stats(pipeline_stats_t* stats);

/**
 * Flush pipeline (finalize current utterance)
 */
int pipeline_flush(char* final_output);

/**
 * Destroy pipeline
 */
void pipeline_destroy();

/**
 * Enable/disable specific stages
 */
int pipeline_set_enabled_stages(int asr_enabled, int mt_enabled, int tts_enabled);

/**
 * Get pipeline state
 * @return 0=idle, 1=processing, 2=awaiting_flush
 */
int pipeline_get_state();

#ifdef __cplusplus
}

namespace s2s {

// Forward declarations
class FeatureExtractor;
class StreamingSegmenter;
class AcousticModel;
class CTCDecoder;
class IndicTrans2Engine;
class RingBuffer;

/**
 * Real-time Speech-to-Speech Pipeline
 * Orchestrates: ASR -> MT -> TTS
 */
class S2SPipeline {
public:
    S2SPipeline();
    ~S2SPipeline();

    /**
     * Initialize pipeline with configuration
     */
    int initialize(const pipeline_config_t& config);

    /**
     * Process audio chunk (streaming)
     * @param audio_chunk Audio samples [sample_rate * frame_shift_ms / 1000]
     * @param num_samples Number of samples
     * @param output_text Recognition/translation output
     * @param output_len Output length
     * @return 0 if processing, 1 if utterance complete
     */
    int process_audio(const float* audio_chunk, uint32_t num_samples,
                      char* output_text, uint32_t* output_len);

    /**
     * Finalize and get output
     */
    int flush(char* final_output);

    /**
     * Reset pipeline for new utterance
     */
    void reset();

    /**
     * Get current pipeline state
     */
    enum State {
        IDLE = 0,
        PROCESSING = 1,
        UTTERANCE_COMPLETE = 2,
        ERROR = -1
    };

    State get_state() const { return current_state; }

    /**
     * Enable/disable pipeline stages
     */
    void set_asr_enabled(bool enabled) { asr_enabled = enabled; }
    void set_mt_enabled(bool enabled) { mt_enabled = enabled; }
    void set_tts_enabled(bool enabled) { tts_enabled = enabled; }

    /**
     * Get pipeline statistics
     */
    const pipeline_stats_t& get_stats() const { return stats; }

private:
    // Pipeline components
    FeatureExtractor* feature_extractor;
    StreamingSegmenter* segmenter;
    AcousticModel* acoustic_model;
    CTCDecoder* ctc_decoder;
    IndicTrans2Engine* mt_engine;
    RingBuffer* audio_buffer;

    // Configuration
    pipeline_config_t config;
    pipeline_stats_t stats;
    State current_state;

    // Feature buffers
    float* frame_buffer;
    float* features_buffer;
    float* logits_buffer;

    // State tracking
    bool asr_enabled;
    bool mt_enabled;
    bool tts_enabled;
    uint32_t frames_in_utterance;

    // Helper methods
    int stage_asr(const float* audio_chunk, uint32_t num_samples, 
                  char* recognition_result);
    int stage_mt(const char* recognition_result, char* translation_result);
    int stage_tts(const char* translation_result, float* audio_output);

    int extract_features(const float* audio, uint32_t num_samples);
    int run_acoustic_model();
    int decode_ctc();
};

/**
 * Ring Buffer for audio streaming
 */
class RingBuffer {
public:
    RingBuffer(uint32_t capacity);
    ~RingBuffer();

    /**
     * Write samples to buffer
     */
    uint32_t write(const float* data, uint32_t num_samples);

    /**
     * Read samples from buffer
     */
    uint32_t read(float* data, uint32_t num_samples);

    /**
     * Get available samples
     */
    uint32_t get_available() const;

    /**
     * Reset buffer
     */
    void reset();

private:
    float* buffer;
    uint32_t capacity;
    uint32_t read_pos;
    uint32_t write_pos;
    uint32_t count;
};

} // namespace s2s

#endif // __cplusplus

#endif // PIPELINE_H
