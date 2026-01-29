#ifndef ASR_H
#define ASR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct asr_context_s asr_context_t;

asr_context_t* asr_create(const char* model_path);
void asr_destroy(asr_context_t* ctx);

int asr_process(asr_context_t* ctx, const float* audio, int num_samples, char* output);

#ifdef __cplusplus
}
#endif

#endif // ASR_H
