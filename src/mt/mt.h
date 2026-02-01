#ifndef MT_H
#define MT_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Machine Translation Context for IndicTrans2
 * Encoder-Decoder architecture with multi-head attention
 */
typedef struct mt_context_s mt_context_t;

/**
 * IndicTrans2 model configuration
 */
typedef struct {
    uint32_t vocab_size;            // Source/target vocabulary size
    uint32_t hidden_dim;            // Embedding and hidden dimension (512)
    uint32_t num_encoder_layers;    // Number of encoder layers
    uint32_t num_decoder_layers;    // Number of decoder layers
    uint32_t num_attention_heads;   // Multi-head attention heads
    uint32_t intermediate_dim;      // FFN intermediate dimension
    float dropout_rate;             // Dropout probability
    uint32_t max_seq_length;        // Maximum sequence length
} mt_model_config_t;

/**
 * Create MT context with model configuration
 */
mt_context_t* mt_create(const char* model_path, const char* source_lang, const char* target_lang);

/**
 * Set model configuration
 */
int mt_set_config(mt_context_t* ctx, const mt_model_config_t* config);

/**
 * Destroy MT context
 */
void mt_destroy(mt_context_t* ctx);

/**
 * Load tokenizer vocabulary
 */
int mt_load_vocabulary(mt_context_t* ctx, const char* vocab_path);

/**
 * Tokenize input text
 * @param input_text Input sentence
 * @param token_ids Output token IDs
 * @param max_tokens Maximum number of tokens
 * @return Number of tokens generated
 */
int mt_tokenize(mt_context_t* ctx, const char* input_text, 
                uint32_t* token_ids, uint32_t max_tokens);

/**
 * Detokenize (convert token IDs back to text)
 */
int mt_detokenize(mt_context_t* ctx, const uint32_t* token_ids, uint32_t num_tokens,
                  char* output_text, int max_length);

/**
 * Translate text
 * @param input_text Source language text
 * @param output_text Target language translation
 * @param max_length Maximum output length
 * @return 0 on success, <0 on error
 */
int mt_translate(mt_context_t* ctx, const char* input_text, 
                 char* output_text, int max_length);

/**
 * Translate with token sequences (for streaming)
 * @param input_tokens Token IDs
 * @param num_tokens Number of input tokens
 * @param output_tokens Output token IDs
 * @param max_output_tokens Max output tokens
 * @return Number of output tokens
 */
int mt_translate_tokens(mt_context_t* ctx, const uint32_t* input_tokens, uint32_t num_tokens,
                       uint32_t* output_tokens, uint32_t max_output_tokens);

/**
 * Encode input (for two-pass decoding)
 * Returns encoder outputs for potential reuse
 */
int mt_encode(mt_context_t* ctx, const char* input_text, float* encoder_outputs);

/**
 * Decode with pre-computed encoder outputs
 */
int mt_decode(mt_context_t* ctx, const float* encoder_outputs, 
              char* output_text, int max_length);

/**
 * Get model statistics
 */
float mt_get_confidence(mt_context_t* ctx);
uint32_t mt_get_model_size(mt_context_t* ctx);

/**
 * Beam search parameters for decoding
 */
typedef struct {
    uint32_t beam_width;            // Number of hypotheses to track
    uint32_t max_length;            // Maximum output sequence length
    float length_penalty;           // Penalty for longer sequences
    float coverage_penalty;         // Coverage penalty for repeated translations
} mt_decode_params_t;

/**
 * Set decoding parameters
 */
int mt_set_decode_params(mt_context_t* ctx, const mt_decode_params_t* params);

#ifdef __cplusplus
}

namespace s2s {

/**
 * IndicTrans2 Tokenizer
 * BPE-based subword tokenization
 */
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    /**
     * Load vocabulary and BPE merges from file
     */
    int load(const char* vocab_path, const char* merges_path);

    /**
     * Encode text to token IDs
     */
    int encode(const char* text, uint32_t* token_ids, uint32_t max_tokens);

    /**
     * Decode token IDs back to text
     */
    int decode(const uint32_t* token_ids, uint32_t num_tokens, char* text);

    /**
     * Get vocabulary size
     */
    uint32_t get_vocab_size() const { return vocab_size; }

private:
    uint32_t vocab_size;
    std::vector<std::string> vocabulary;
};

/**
 * Transformer Encoder
 * Encodes input sequences
 */
class TransformerEncoder {
public:
    TransformerEncoder(const mt_model_config_t& config);
    ~TransformerEncoder();

    /**
     * Forward pass through encoder
     * @param input_ids Input token IDs [sequence_length]
     * @param num_tokens Number of tokens
     * @param output Output embeddings [sequence_length x hidden_dim]
     */
    int forward(const uint32_t* input_ids, uint32_t num_tokens, float* output);

    /**
     * Get output dimension
     */
    uint32_t get_hidden_dim() const { return config.hidden_dim; }

private:
    mt_model_config_t config;
    
    // Embedding matrices and weights would be loaded from model
    float* token_embedding;
    float* position_embedding;

    int embed_tokens(const uint32_t* token_ids, uint32_t num_tokens, float* embeddings);
    int apply_position_encoding(float* embeddings, uint32_t seq_len);
    int apply_attention_layers(float* embeddings, uint32_t seq_len);
};

/**
 * Transformer Decoder
 * Decodes output sequences with attention to encoder
 */
class TransformerDecoder {
public:
    TransformerDecoder(const mt_model_config_t& config);
    ~TransformerDecoder();

    /**
     * Forward pass (autoregressive)
     * @param input_ids Decoder input token IDs
     * @param encoder_output Encoder outputs
     * @param encoder_seq_len Length of encoder output
     * @param logits Output logits [vocab_size]
     */
    int forward(const uint32_t* input_ids, uint32_t input_len,
                const float* encoder_output, uint32_t encoder_seq_len,
                float* logits);

    /**
     * Reset hidden state between sequences
     */
    void reset_state();

private:
    mt_model_config_t config;
    float* token_embedding;
    
    int embed_tokens(const uint32_t* token_ids, uint32_t num_tokens, float* embeddings);
    int apply_cross_attention(float* decoder_hidden, const float* encoder_output,
                             uint32_t encoder_seq_len);
    int apply_self_attention(float* hidden, uint32_t seq_len);
};

/**
 * Beam Search Decoder
 * Generates translations using beam search
 */
class BeamSearchDecoder {
public:
    BeamSearchDecoder(const mt_decode_params_t& params, const Tokenizer& tokenizer);
    ~BeamSearchDecoder();

    /**
     * Decode encoder outputs to text
     * @param encoder_output Encoder outputs
     * @param output_text Decoded translation
     */
    int decode(const float* encoder_output, uint32_t encoder_seq_len,
               char* output_text, int max_length);

    /**
     * Greedy decoding (faster, less accurate)
     */
    int greedy_decode(const float* encoder_output, uint32_t encoder_seq_len,
                     char* output_text, int max_length);

private:
    mt_decode_params_t params;
    const Tokenizer& tokenizer;
    
    struct Hypothesis {
        std::vector<uint32_t> tokens;
        float score;
        float* decoder_state;
    };

    std::vector<Hypothesis> hypotheses;

    int sample_next_token(const float* logits, uint32_t vocab_size,
                         uint32_t top_k = 40);
};

/**
 * IndicTrans2 Machine Translation Engine
 */
class IndicTrans2Engine {
public:
    IndicTrans2Engine(const char* model_path, const char* source_lang, const char* target_lang);
    ~IndicTrans2Engine();

    /**
     * Initialize with model and tokenizers
     */
    int initialize();

    /**
     * Translate text
     */
    int translate(const char* input_text, char* output_text, int max_length);

    /**
     * Set decoding configuration
     */
    void set_decode_params(const mt_decode_params_t& params);

    /**
     * Get model memory footprint
     */
    uint64_t get_model_size() const;

    /**
     * Check if engine is ready
     */
    bool is_ready() const { return initialized; }

private:
    std::string model_path;
    std::string source_lang;
    std::string target_lang;
    bool initialized;

    mt_model_config_t config;
    Tokenizer* tokenizer;
    TransformerEncoder* encoder;
    TransformerDecoder* decoder;
    BeamSearchDecoder* beam_decoder;

    uint64_t model_size;
};

} // namespace s2s

#endif // __cplusplus

#endif // MT_H
