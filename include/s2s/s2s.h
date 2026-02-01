#ifndef S2S_H
#define S2S_H

// Public API for Speech-to-Speech on-device

#ifdef __cplusplus
extern "C" {
#endif

typedef struct s2s_context_s s2s_context_t;

// Initialize S2S pipeline
s2s_context_t* s2s_create();
void s2s_destroy(s2s_context_t* ctx);

// Load models
int s2s_load_asr_model(s2s_context_t* ctx, const char* model_path);
int s2s_load_mt_model(s2s_context_t* ctx, const char* model_path);
int s2s_load_tts_model(s2s_context_t* ctx, const char* model_path);

// Process audio
int s2s_process_audio(s2s_context_t* ctx, const float* input_audio, int num_samples, 
                      float* output_audio, int* output_samples);

#ifdef __cplusplus
}
#endif

#endif // S2S_H
