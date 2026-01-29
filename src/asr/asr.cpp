#include "asr.h"

struct asr_context_s {
    char* model_path;
    float* model_weights;
    int model_size;
};

asr_context_t* asr_create(const char* model_path) {
    asr_context_t* ctx = new asr_context_t();
    ctx->model_path = new char[256];
    return ctx;
}

void asr_destroy(asr_context_t* ctx) {
    if (ctx) {
        delete[] ctx->model_path;
        delete ctx;
    }
}

int asr_process(asr_context_t* ctx, const float* audio, int num_samples, char* output) {
    if (!ctx || !audio) return -1;
    return 0;
}
