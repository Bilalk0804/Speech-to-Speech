#ifndef TTS_H
#define TTS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tts_context_s tts_context_t;

tts_context_t* tts_create(const char* model_path, const char* voice_id);
void tts_destroy(tts_context_t* ctx);

int tts_synthesize(tts_context_t* ctx, const char* text, float* audio_output, int* num_samples);

#ifdef __cplusplus
}
#endif

#endif // TTS_H
