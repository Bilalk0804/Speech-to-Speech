#include "mt.h"
#include <cstring>

struct mt_context_s {
    char* model_path;
    char* source_lang;
    char* target_lang;
};

mt_context_t* mt_create(const char* model_path, const char* source_lang, const char* target_lang) {
    mt_context_t* ctx = new mt_context_t();
    ctx->model_path = new char[256];
    ctx->source_lang = new char[16];
    ctx->target_lang = new char[16];
    return ctx;
}

void mt_destroy(mt_context_t* ctx) {
    if (ctx) {
        delete[] ctx->model_path;
        delete[] ctx->source_lang;
        delete[] ctx->target_lang;
        delete ctx;
    }
}

int mt_translate(mt_context_t* ctx, const char* input_text, char* output_text, int max_length) {
    if (!ctx || !input_text) return -1;
    return 0;
}
