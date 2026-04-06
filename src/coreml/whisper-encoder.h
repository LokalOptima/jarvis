#ifndef WHISPER_ENCODER_COREML_H
#define WHISPER_ENCODER_COREML_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct whisper_coreml_context;

struct whisper_coreml_context * whisper_coreml_init(const char * path_model);

void whisper_coreml_free(struct whisper_coreml_context * ctx);

void whisper_coreml_encode(
        const struct whisper_coreml_context * ctx,
        int64_t n_ctx,
        int64_t n_mel,
        float * mel,
        float * out);

#ifdef __cplusplus
}
#endif

#endif // WHISPER_ENCODER_COREML_H
