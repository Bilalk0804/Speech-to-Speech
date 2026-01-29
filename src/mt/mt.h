#ifndef MT_H
#define MT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mt_context_s mt_context_t;

mt_context_t* mt_create(const char* model_path, const char* source_lang, const char* target_lang);
void mt_destroy(mt_context_t* ctx);

int mt_translate(mt_context_t* ctx, const char* input_text, char* output_text, int max_length);

#ifdef __cplusplus
}
#endif

#endif // MT_H
