#include "whisper.h"

#include "ggml-cpu.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef WHISPER_USE_COREML
#include "coreml/whisper-encoder.h"
#endif

#ifdef WHISPER_USE_OPENVINO
#include "openvino/whisper-openvino-encoder.h"
#endif

#include <atomic>
#include <algorithm>
#include <cassert>
#include <cfloat>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <random>
#include <functional>
#include <codecvt>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if defined(GGML_BIG_ENDIAN)
#include <bit>

template<typename T>
static T byteswap(T value) {
    return std::byteswap(value);
}

template<>
float byteswap(float value) {
    return std::bit_cast<float>(byteswap(std::bit_cast<std::uint32_t>(value)));
}

template<typename T>
static void byteswap_tensor_data(ggml_tensor * tensor) {
    T * datum = reinterpret_cast<T *>(tensor->data);
    for (int i = 0; i < ggml_nelements(tensor); i++) {
        datum[i] = byteswap(datum[i]);
    }
}

static void byteswap_tensor(ggml_tensor * tensor) {
    switch (tensor->type) {
        case GGML_TYPE_I16: {
            byteswap_tensor_data<int16_t>(tensor);
            break;
        }
        case GGML_TYPE_F16: {
            byteswap_tensor_data<ggml_fp16_t>(tensor);
            break;
        }
        case GGML_TYPE_I32: {
            byteswap_tensor_data<int32_t>(tensor);
            break;
        }
        case GGML_TYPE_F32: {
            byteswap_tensor_data<float>(tensor);
            break;
        }
        default: { // GML_TYPE_I8
            break;
        }
    }
}

#define BYTESWAP_VALUE(d) d = byteswap(d)
#define BYTESWAP_FILTERS(f)            \
    do {                              \
        for (auto & datum : f.data) { \
            datum = byteswap(datum);  \
        }                             \
    } while (0)
#define BYTESWAP_TENSOR(t)       \
    do {                         \
        byteswap_tensor(t); \
    } while (0)
#else
#define BYTESWAP_VALUE(d) do {} while (0)
#define BYTESWAP_FILTERS(f) do {} while (0)
#define BYTESWAP_TENSOR(t) do {} while (0)
#endif

#ifdef __GNUC__
#ifdef __MINGW32__
#define WHISPER_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define WHISPER_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define WHISPER_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

WHISPER_ATTRIBUTE_FORMAT(2, 3)
static void whisper_log_internal        (ggml_log_level level, const char * format, ...);
static void whisper_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define WHISPER_LOG_ERROR(...) whisper_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define WHISPER_LOG_WARN(...)  whisper_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define WHISPER_LOG_INFO(...)  whisper_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)

// define this to enable verbose trace logging - useful for debugging purposes
//#define WHISPER_DEBUG

#if defined(WHISPER_DEBUG)
#define WHISPER_LOG_DEBUG(...) whisper_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define WHISPER_LOG_DEBUG(...)
#endif

#define WHISPER_ASSERT(x) \
    do { \
        if (!(x)) { \
            WHISPER_LOG_ERROR("WHISPER_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#define WHISPER_MAX_DECODERS 8
#define WHISPER_MAX_NODES 4096

//
// ggml helpers
//

static bool ggml_graph_compute_helper(
          struct ggml_cgraph * graph,
        std::vector<uint8_t> & buf,
                         int   n_threads,
         ggml_abort_callback   abort_callback,
                        void * abort_callback_data) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);

    plan.abort_callback      = abort_callback;
    plan.abort_callback_data = abort_callback_data;

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    return ggml_graph_compute(graph, &plan);
}

static bool ggml_graph_compute_helper(
      ggml_backend_sched_t   sched,
        struct ggml_cgraph * graph,
                       int   n_threads) {

    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

        auto * fn_set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn_set_n_threads) {
            fn_set_n_threads(backend, n_threads);
        }
    }

    bool t = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_reset(sched);
    return t;
}

// faster matrix multiplications for tensors that do not have dimension 0 divisible by "pad"
// the idea is to represent the original matrix multiplication:
//
//   Z = X @ Y
//
// with the sum of two matrix multiplications:
//
//   Z = (X_0 @ Y_0) + (X_1 @ Y_1)
//
// here X_0 and Y_0 are views of X and Y that have dimension 0 divisible by "pad"
// and X_1 and Y_1 are the remaining views. X_1 and Y_1 end up being small matrices that can be processed with more
// general-purpose kernels
//
static struct ggml_tensor * ggml_mul_mat_pad(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y, int pad = 32) {
    // use padding only if dimension 0 is at least 8 times larger than the padding
    // else we won't get much benefit from the optimization
    const int n_pad_req = 8;

    if (x->ne[0] % pad == 0 || x->ne[0] / pad < n_pad_req) {
        return ggml_mul_mat(ctx, x, y);
    }

    struct ggml_tensor * x_0 = ggml_view_3d(ctx, x, (x->ne[0]/pad)*pad, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    struct ggml_tensor * x_1 = ggml_view_3d(ctx, x,  x->ne[0]%pad,      x->ne[1], x->ne[2], x->nb[1], x->nb[2], x_0->ne[0]*x_0->nb[0]);

    struct ggml_tensor * y_0 = ggml_view_3d(ctx, y, (y->ne[0]/pad)*pad, y->ne[1], y->ne[2], y->nb[1], y->nb[2], 0);
    struct ggml_tensor * y_1 = ggml_view_3d(ctx, y,  y->ne[0]%pad,      y->ne[1], y->ne[2], y->nb[1], y->nb[2], y_0->ne[0]*y_0->nb[0]);

    return ggml_add(ctx,
            ggml_mul_mat(ctx, x_0, y_0),
            ggml_mul_mat(ctx, x_1, y_1));
}

// TODO: check if other platforms can benefit from this optimization
// TODO: CUDA is currently broken - seems ggml_mul_mat does not handle views correctly
#if defined(GGML_USE_METAL)
#define ggml_mul_mat ggml_mul_mat_pad
#endif

// available whisper models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_TINY,
    MODEL_BASE,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
};

static const std::map<e_model, std::string> g_model_name = {
    { MODEL_UNKNOWN,  "unknown"  },
    { MODEL_TINY,     "tiny"     },
    { MODEL_BASE,     "base"     },
    { MODEL_SMALL,    "small"    },
    { MODEL_MEDIUM,   "medium"   },
    { MODEL_LARGE,    "large"    },
};

static const std::map<std::string, std::pair<int, std::string>> g_lang = {
    { "en",  { 0,  "english",         } },
    { "zh",  { 1,  "chinese",         } },
    { "de",  { 2,  "german",          } },
    { "es",  { 3,  "spanish",         } },
    { "ru",  { 4,  "russian",         } },
    { "ko",  { 5,  "korean",          } },
    { "fr",  { 6,  "french",          } },
    { "ja",  { 7,  "japanese",        } },
    { "pt",  { 8,  "portuguese",      } },
    { "tr",  { 9,  "turkish",         } },
    { "pl",  { 10, "polish",          } },
    { "ca",  { 11,  "catalan",        } },
    { "nl",  { 12,  "dutch",          } },
    { "ar",  { 13,  "arabic",         } },
    { "sv",  { 14,  "swedish",        } },
    { "it",  { 15,  "italian",        } },
    { "id",  { 16,  "indonesian",     } },
    { "hi",  { 17,  "hindi",          } },
    { "fi",  { 18,  "finnish",        } },
    { "vi",  { 19,  "vietnamese",     } },
    { "he",  { 20,  "hebrew",         } },
    { "uk",  { 21,  "ukrainian",      } },
    { "el",  { 22,  "greek",          } },
    { "ms",  { 23,  "malay",          } },
    { "cs",  { 24,  "czech",          } },
    { "ro",  { 25,  "romanian",       } },
    { "da",  { 26,  "danish",         } },
    { "hu",  { 27,  "hungarian",      } },
    { "ta",  { 28,  "tamil",          } },
    { "no",  { 29,  "norwegian",      } },
    { "th",  { 30,  "thai",           } },
    { "ur",  { 31,  "urdu",           } },
    { "hr",  { 32,  "croatian",       } },
    { "bg",  { 33,  "bulgarian",      } },
    { "lt",  { 34,  "lithuanian",     } },
    { "la",  { 35,  "latin",          } },
    { "mi",  { 36,  "maori",          } },
    { "ml",  { 37,  "malayalam",      } },
    { "cy",  { 38,  "welsh",          } },
    { "sk",  { 39,  "slovak",         } },
    { "te",  { 40,  "telugu",         } },
    { "fa",  { 41,  "persian",        } },
    { "lv",  { 42,  "latvian",        } },
    { "bn",  { 43,  "bengali",        } },
    { "sr",  { 44,  "serbian",        } },
    { "az",  { 45,  "azerbaijani",    } },
    { "sl",  { 46,  "slovenian",      } },
    { "kn",  { 47,  "kannada",        } },
    { "et",  { 48,  "estonian",       } },
    { "mk",  { 49,  "macedonian",     } },
    { "br",  { 50,  "breton",         } },
    { "eu",  { 51,  "basque",         } },
    { "is",  { 52,  "icelandic",      } },
    { "hy",  { 53,  "armenian",       } },
    { "ne",  { 54,  "nepali",         } },
    { "mn",  { 55,  "mongolian",      } },
    { "bs",  { 56,  "bosnian",        } },
    { "kk",  { 57,  "kazakh",         } },
    { "sq",  { 58,  "albanian",       } },
    { "sw",  { 59,  "swahili",        } },
    { "gl",  { 60,  "galician",       } },
    { "mr",  { 61,  "marathi",        } },
    { "pa",  { 62,  "punjabi",        } },
    { "si",  { 63,  "sinhala",        } },
    { "km",  { 64,  "khmer",          } },
    { "sn",  { 65,  "shona",          } },
    { "yo",  { 66,  "yoruba",         } },
    { "so",  { 67,  "somali",         } },
    { "af",  { 68,  "afrikaans",      } },
    { "oc",  { 69,  "occitan",        } },
    { "ka",  { 70,  "georgian",       } },
    { "be",  { 71,  "belarusian",     } },
    { "tg",  { 72,  "tajik",          } },
    { "sd",  { 73,  "sindhi",         } },
    { "gu",  { 74,  "gujarati",       } },
    { "am",  { 75,  "amharic",        } },
    { "yi",  { 76,  "yiddish",        } },
    { "lo",  { 77,  "lao",            } },
    { "uz",  { 78,  "uzbek",          } },
    { "fo",  { 79,  "faroese",        } },
    { "ht",  { 80,  "haitian creole", } },
    { "ps",  { 81,  "pashto",         } },
    { "tk",  { 82,  "turkmen",        } },
    { "nn",  { 83,  "nynorsk",        } },
    { "mt",  { 84,  "maltese",        } },
    { "sa",  { 85,  "sanskrit",       } },
    { "lb",  { 86,  "luxembourgish",  } },
    { "my",  { 87,  "myanmar",        } },
    { "bo",  { 88,  "tibetan",        } },
    { "tl",  { 89,  "tagalog",        } },
    { "mg",  { 90,  "malagasy",       } },
    { "as",  { 91,  "assamese",       } },
    { "tt",  { 92,  "tatar",          } },
    { "haw", { 93,  "hawaiian",       } },
    { "ln",  { 94,  "lingala",        } },
    { "ha",  { 95,  "hausa",          } },
    { "ba",  { 96,  "bashkir",        } },
    { "jw",  { 97,  "javanese",       } },
    { "su",  { 98,  "sundanese",      } },
    { "yue", { 99,  "cantonese",      } },
};

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

struct whisper_vocab {
    using id    = int32_t;
    using token = std::string;

    int n_vocab = 51864;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    // reference: https://github.com/openai/whisper/blob/248b6cb124225dd263bb9bd32d060b6517e067f8/whisper/tokenizer.py#L334-L349
    id token_eot        = 50256;
    id token_sot        = 50257;
    // task tokens (used only for multilingual models)
    id token_translate  = 50357;
    id token_transcribe = 50358;
    // other special tokens
    id token_solm       = 50359; // [TDRZ] used by tinydiarize models to indicate speaker turn
    id token_prev       = 50360;
    id token_nosp       = 50361;
    id token_not        = 50362; // no timestamps
    id token_beg        = 50363; // begin timestamps

    bool is_multilingual() const {
        return n_vocab >= 51865;
    }

    int num_languages() const {
        return n_vocab - 51765 - (is_multilingual() ? 1 : 0);
    }
};

struct whisper_segment {
    int64_t t0;
    int64_t t1;

    std::string text;
    float no_speech_prob;

    std::vector<whisper_token_data> tokens;

    bool speaker_turn_next;
};

struct whisper_batch {
    int32_t n_tokens;

    whisper_token  *  token;
    whisper_pos    *  pos;
    int32_t        *  n_seq_id; // always 1, here for consistency with llama.cpp
    whisper_seq_id ** seq_id;   // null terminated
    int8_t         *  logits;
};

static struct whisper_batch whisper_batch_init(int32_t n_tokens, int32_t n_seq_max) {
    whisper_batch batch = { 0, nullptr, nullptr, nullptr, nullptr, nullptr, };

    batch.token    = (whisper_token *  ) malloc(sizeof(whisper_token)    * (n_tokens));
    batch.pos      = (whisper_pos *)     malloc(sizeof(whisper_pos)      * (n_tokens));
    batch.n_seq_id = (int32_t *)         malloc(sizeof(int32_t)          * (n_tokens));
    batch.seq_id   = (whisper_seq_id **) malloc(sizeof(whisper_seq_id *) * (n_tokens + 1));
    for (int i = 0; i < n_tokens; ++i) {
        batch.seq_id[i] = (whisper_seq_id *) malloc(sizeof(whisper_seq_id)   * n_seq_max);
    }
    batch.seq_id[n_tokens] = nullptr;
    batch.logits   = (int8_t *)          malloc(sizeof(int8_t)           * n_tokens);

    return batch;
}

static void whisper_batch_free(struct whisper_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i]; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}

// replace std::pair by using customized pair struct (reason: std::pair is very slow)
template<typename A, typename B>
struct whisper_pair {
    A first;
    B second;

    // Define a constructor that takes two arguments.
    whisper_pair(const A& a, const B& b) : first(a), second(b) {}
    // Define a constructor that takes no argument.
    whisper_pair() : first(A()), second(B()) {}
};

// ggml_backend_sched wrapper for whisper usage
struct whisper_sched {
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> meta;
};

static size_t whisper_sched_size(struct whisper_sched & allocr) {
    size_t size = allocr.meta.size();
    for (int i = 0; i < ggml_backend_sched_get_n_backends(allocr.sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(allocr.sched, i);
        size += ggml_backend_sched_get_buffer_size(allocr.sched, backend);
    }
    return size;
}

// measure the memory usage of a graph and prepare the allocr's internal data buffer
static bool whisper_sched_graph_init(struct whisper_sched & allocr, std::vector<ggml_backend_t> backends, std::function<struct ggml_cgraph *()> && get_graph) {
    auto & sched = allocr.sched;
    auto & meta  = allocr.meta;

    sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), WHISPER_MAX_NODES, false);

    meta.resize(ggml_tensor_overhead()*WHISPER_MAX_NODES + ggml_graph_overhead());

    // since there are dependencies between the different graphs,
    // we need to allocate them instead of only reserving to get the correct compute buffer size
    if (!ggml_backend_sched_alloc_graph(sched, get_graph())) {
        // failed to allocate the compute buffer
        WHISPER_LOG_ERROR("%s: failed to allocate the compute buffer\n", __func__);
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

// medium
// hparams: {
// 'n_mels': 80,
// 'n_vocab': 51864,
// 'n_audio_ctx': 1500,
// 'n_audio_state': 1024,
// 'n_audio_head': 16,
// 'n_audio_layer': 24,
// 'n_text_ctx': 448,
// 'n_text_state': 1024,
// 'n_text_head': 16,
// 'n_text_layer': 24
// }
//
// default hparams (Whisper tiny)
struct whisper_hparams {
    int32_t n_vocab       = 51864;
    int32_t n_audio_ctx   = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head  = 6;
    int32_t n_audio_layer = 4;
    int32_t n_text_ctx    = 448;
    int32_t n_text_state  = 384;
    int32_t n_text_head   = 6;
    int32_t n_text_layer  = 4;
    int32_t n_mels        = 80;
    int32_t ftype         = 1;
    float   eps           = 1e-5f;
};

// audio encoding layer
struct whisper_layer_encoder {
    // encoder.blocks.*.attn_ln
    struct ggml_tensor * attn_ln_0_w;
    struct ggml_tensor * attn_ln_0_b;

    // encoder.blocks.*.attn.out
    struct ggml_tensor * attn_ln_1_w;
    struct ggml_tensor * attn_ln_1_b;

    // encoder.blocks.*.attn.query
    struct ggml_tensor * attn_q_w;
    struct ggml_tensor * attn_q_b;

    // encoder.blocks.*.attn.key
    struct ggml_tensor * attn_k_w;

    // encoder.blocks.*.attn.value
    struct ggml_tensor * attn_v_w;
    struct ggml_tensor * attn_v_b;

    // encoder.blocks.*.mlp_ln
    struct ggml_tensor * mlp_ln_w;
    struct ggml_tensor * mlp_ln_b;

    // encoder.blocks.*.mlp.0
    struct ggml_tensor * mlp_0_w;
    struct ggml_tensor * mlp_0_b;

    // encoder.blocks.*.mlp.2
    struct ggml_tensor * mlp_1_w;
    struct ggml_tensor * mlp_1_b;
};

// token decoding layer (struct kept for binary compat with whisper_model)
struct whisper_layer_decoder {
    // decoder.blocks.*.attn_ln
    struct ggml_tensor * attn_ln_0_w;
    struct ggml_tensor * attn_ln_0_b;

    // decoder.blocks.*.attn.out
    struct ggml_tensor * attn_ln_1_w;
    struct ggml_tensor * attn_ln_1_b;

    // decoder.blocks.*.attn.query
    struct ggml_tensor * attn_q_w;
    struct ggml_tensor * attn_q_b;

    // decoder.blocks.*.attn.key
    struct ggml_tensor * attn_k_w;

    // decoder.blocks.*.attn.value
    struct ggml_tensor * attn_v_w;
    struct ggml_tensor * attn_v_b;

    // decoder.blocks.*.cross_attn_ln
    struct ggml_tensor * cross_attn_ln_0_w;
    struct ggml_tensor * cross_attn_ln_0_b;

    // decoder.blocks.*.cross_attn.out
    struct ggml_tensor * cross_attn_ln_1_w;
    struct ggml_tensor * cross_attn_ln_1_b;

    // decoder.blocks.*.cross_attn.query
    struct ggml_tensor * cross_attn_q_w;
    struct ggml_tensor * cross_attn_q_b;

    // decoder.blocks.*.cross_attn.key
    struct ggml_tensor * cross_attn_k_w;

    // decoder.blocks.*.cross_attn.value
    struct ggml_tensor * cross_attn_v_w;
    struct ggml_tensor * cross_attn_v_b;

    // decoder.blocks.*.mlp_ln
    struct ggml_tensor * mlp_ln_w;
    struct ggml_tensor * mlp_ln_b;

    // decoder.blocks.*.mlp.0
    struct ggml_tensor * mlp_0_w;
    struct ggml_tensor * mlp_0_b;

    // decoder.blocks.*.mlp.2
    struct ggml_tensor * mlp_1_w;
    struct ggml_tensor * mlp_1_b;
};

struct whisper_kv_cell {
    whisper_pos pos = -1;

    std::set<whisper_seq_id> seq_id;

    bool has_seq_id(const whisper_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }
};

struct whisper_kv_cache {
    uint32_t head = 0;
    uint32_t size = 0;

    // computed before each graph build
    uint32_t n = 0;

    std::vector<whisper_kv_cell> cells;

    struct ggml_tensor * k;
    struct ggml_tensor * v;

    ggml_backend_buffer_t buffer = nullptr;

    std::vector<uint8_t> ctx_buf;
};

struct whisper_model {
    e_model type = MODEL_UNKNOWN;

    whisper_hparams hparams;
    whisper_filters filters;

    // encoder.positional_embedding
    struct ggml_tensor * e_pe;

    // encoder.conv1
    struct ggml_tensor * e_conv_1_w;
    struct ggml_tensor * e_conv_1_b;

    // encoder.conv2
    struct ggml_tensor * e_conv_2_w;
    struct ggml_tensor * e_conv_2_b;

    // encoder.ln_post
    struct ggml_tensor * e_ln_w;
    struct ggml_tensor * e_ln_b;

    // decoder.positional_embedding
    struct ggml_tensor * d_pe;

    // decoder.token_embedding
    struct ggml_tensor * d_te;

    // decoder.ln
    struct ggml_tensor * d_ln_w;
    struct ggml_tensor * d_ln_b;

    std::vector<whisper_layer_encoder> layers_encoder;
    std::vector<whisper_layer_decoder> layers_decoder;

    // ggml context that contains all the meta information about the model tensors
    struct ggml_context * ctx = nullptr;

    // the model backend data is read-only and can be shared between processors
    ggml_backend_buffer_t buffer = nullptr;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Struct stubs kept for binary compat with whisper_state
struct whisper_partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct whisper_grammar {
    /*const*/ std::vector<std::vector<whisper_grammar_element>> rules;
    std::vector<std::vector<const whisper_grammar_element *>>   stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    whisper_partial_utf8 partial_utf8;
};

struct whisper_grammar_candidate {
    whisper_token          id;
    const uint32_t       * code_points;
    whisper_partial_utf8   partial_utf8;
};

struct whisper_sequence {
    std::vector<whisper_token_data> tokens;

    // the accumulated transcription in the current iteration (used to truncate the tokens array)
    int result_len;

    double sum_logprobs_all; // the sum of the log probabilities of the tokens
    double sum_logprobs;     // the sum of the log probabilities of the tokens (first result_len tokens)
    double avg_logprobs;     // the average log probability of the tokens
    double entropy;          // the entropy of the tokens
    double score;            // likelihood rank score
};

// TAGS: WHISPER_DECODER_INIT
struct whisper_decoder {
    // the currently generated sequence of tokens
    whisper_sequence sequence;

    // grammar parse state of generated sequence of tokens
    whisper_grammar  grammar;

    int i_batch;    // the index of the token in the current batch
    int seek_delta; // the window shift found so far based on the decoded timestamp tokens

    bool failed;    // has the current segment failed to decode?
    bool completed; // has the decoder completed the current segment?
    bool has_ts;    // have we already sampled a non-beg timestamp token for the current segment?

    // new token probs, logits and logprobs after the last whisper_decode (1-dimensional array: [n_vocab])
    std::vector<float> probs;
    std::vector<float> logits;
    std::vector<float> logprobs;

    // work container used to avoid memory allocations
    std::vector<whisper_pair<double, whisper_vocab::id>> logits_id;

    mutable std::mt19937 rng; // used for sampling at t > 0.0
};

// [EXPERIMENTAL] Token-level timestamps with DTW
struct whisper_aheads_masks {
    std::vector<struct ggml_tensor *> m;    // One mask per text layer.
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

struct whisper_state {
    int64_t t_sample_us = 0;
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int64_t t_batchd_us = 0;
    int64_t t_prompt_us = 0;
    int64_t t_mel_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_encode = 0; // number of encoder calls
    int32_t n_decode = 0; // number of decoder calls with n_tokens == 1  (text-generation)
    int32_t n_batchd = 0; // number of decoder calls with n_tokens <  16 (batch decoding)
    int32_t n_prompt = 0; // number of decoder calls with n_tokens >  1  (prompt encoding)
    int32_t n_fail_p = 0; // number of logprob threshold failures
    int32_t n_fail_h = 0; // number of entropy threshold failures

    // number of decoders for which we have constructed the KV cache
    int32_t kv_self_n_dec = 0;

    // unified self-attention KV cache for all decoders
    whisper_kv_cache kv_self;

    // cross-attention KV cache for the decoders
    // shared between all decoders
    whisper_kv_cache kv_cross;

    // padded buffer for flash-attention
    whisper_kv_cache kv_pad;

    whisper_mel mel;

    whisper_batch batch;

    whisper_decoder decoders[WHISPER_MAX_DECODERS];

    std::vector<ggml_backend_t> backends;

    // - stores meta info about the intermediate tensors into the `meta` buffers
    whisper_sched sched_conv;
    whisper_sched sched_encode;
    whisper_sched sched_cross;
    whisper_sched sched_decode;

    // result of the encoder
    struct ggml_tensor * embd_conv = nullptr;
    struct ggml_tensor * embd_enc  = nullptr;

    // helpers for GPU offloading
    std::vector<float> inp_mel;
    std::vector<float> inp_mask;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;

    std::vector<whisper_segment> result_all;
    std::vector<whisper_token>   prompt_past;

    int lang_id = 0; // english by default

    std::string path_model; // populated by whisper_init_from_file_with_params()

#ifdef WHISPER_USE_COREML
    whisper_coreml_context * ctx_coreml = nullptr;
#endif

#ifdef WHISPER_USE_OPENVINO
    whisper_openvino_context * ctx_openvino = nullptr;
#endif

    // [EXPERIMENTAL] token-level timestamps data
    int64_t t_beg  = 0;
    int64_t t_last = 0;

    whisper_token tid_last;

    std::vector<float> energy; // PCM signal energy
    float no_speech_prob = 0.0f;

    // [EXPERIMENTAL] Token-level timestamps with DTW
    whisper_aheads_masks aheads_masks;
    ggml_tensor * aheads_cross_QKs = nullptr;
    std::vector<float> aheads_cross_QKs_data;

    // [EXPERIMENTAL] speed-up techniques
    int32_t exp_n_audio_ctx = 0; // 0 - use default
    bool skip_cross = false;     // skip cross-attention KV pre-computation (encoder-only mode)

    // ggml-accelerated mel spectrogram
    whisper_sched sched_mel;
    struct ggml_tensor * mel_dft_cos = nullptr;  // (n_fft_bins, frame_size) DFT cosine basis
    struct ggml_tensor * mel_dft_sin = nullptr;  // (n_fft_bins, frame_size) DFT sine basis
    struct ggml_tensor * mel_filters_tensor = nullptr;  // (n_mel, n_fft_bins) mel filterbank
    ggml_backend_buffer_t mel_basis_buffer = nullptr;
    std::vector<uint8_t> mel_basis_ctx_buf;  // keeps tensor metadata alive after ggml_free
    int mel_n_frames = 0;  // number of frames for current buffer size

    // Pre-allocated buffers for mel spectrogram (avoid per-call heap alloc)
    std::vector<float> mel_samples_padded;
    std::vector<float> mel_frames_buf;
    std::vector<float> mel_raw_buf;
};

struct whisper_context {
    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    ggml_type wtype = ggml_type::GGML_TYPE_F16; // weight type (FP32 / FP16 / QX)
    ggml_type itype = ggml_type::GGML_TYPE_F16; // intermediate type (FP32 or FP16)

    whisper_context_params params;

    whisper_model model;
    whisper_vocab vocab;

    whisper_state * state = nullptr;

    std::string path_model; // populated by whisper_init_from_file_with_params()
};

struct whisper_global {
    // We save the log callback globally
    ggml_log_callback log_callback = whisper_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static whisper_global g_state;

template<typename T>
static void read_safe(whisper_model_loader * loader, T & dest) {
    loader->read(loader->context, &dest, sizeof(T));
    BYTESWAP_VALUE(dest);
}

static bool whisper_kv_cache_init(
             struct whisper_kv_cache & cache,
                      ggml_backend_t   backend,
                           ggml_type   wtype,
                             int64_t   n_text_state,
                             int64_t   n_text_layer,
                                 int   n_ctx) {
    const int64_t n_mem      = n_text_layer*n_ctx;
    const int64_t n_elements = n_text_state*n_mem;

    cache.ctx_buf.resize(2*ggml_tensor_overhead());

    struct ggml_init_params params = {
        /*.mem_size   =*/ cache.ctx_buf.size(),
        /*.mem_buffer =*/ cache.ctx_buf.data(),
        /*.no_alloc   =*/ true,
    };

    cache.head = 0;
    cache.size = n_ctx;

    cache.cells.clear();
    cache.cells.resize(n_ctx);

    struct ggml_context * ctx = ggml_init(params);

    if (!ctx) {
        WHISPER_LOG_ERROR("%s: failed to allocate memory for the kv cache context\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(ctx, wtype, n_elements);

    cache.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!cache.buffer) {
        WHISPER_LOG_ERROR("%s: failed to allocate memory for the kv cache\n", __func__);
        return false;
    }

    ggml_backend_buffer_clear(cache.buffer, 0);

    ggml_free(ctx);

    return true;
}

static void whisper_kv_cache_free(struct whisper_kv_cache & cache) {
    ggml_backend_buffer_free(cache.buffer);
}

static void aheads_masks_free(struct whisper_aheads_masks & aheads_masks) {
    ggml_free(aheads_masks.ctx);
    ggml_backend_buffer_free(aheads_masks.buffer);
    aheads_masks.ctx = nullptr;
}

static ggml_backend_t whisper_backend_init_gpu(const whisper_context_params & params) {
    ggml_log_set(g_state.log_callback, g_state.log_callback_user_data);

    if (params.use_gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                WHISPER_LOG_INFO("%s: using %s backend\n", __func__, ggml_backend_dev_name(dev));
                ggml_backend_t result = ggml_backend_dev_init(dev, nullptr);
                if (!result) {
                    WHISPER_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                }
                return result;
            }
        }
    }

    return nullptr;
}

static std::vector<ggml_backend_t> whisper_backend_init(const whisper_context_params & params) {
    std::vector<ggml_backend_t> result;

    ggml_backend_t backend_gpu = whisper_backend_init_gpu(params);

    if (backend_gpu) {
        result.push_back(backend_gpu);
    }

    // ACCEL backends
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            WHISPER_LOG_INFO("%s: using %s backend\n", __func__, ggml_backend_dev_name(dev));
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                WHISPER_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                continue;
            }
            result.push_back(backend);
        }
    }

    GGML_UNUSED(params);

    result.push_back(ggml_backend_cpu_init());

    return result;
}

static ggml_backend_buffer_type_t whisper_default_buffer_type(const whisper_context_params & params) {
    if (!params.use_gpu) {
        return ggml_backend_cpu_buffer_type();
    }

    // if we have a GPU device - use it
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            WHISPER_LOG_INFO("%s: using device %s (%s)\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
            return ggml_backend_dev_buffer_type(dev);
        }
    }

    return ggml_backend_cpu_buffer_type();
}

// whisper_lang_str needed by whisper_model_load for extra token names
const char * whisper_lang_str(int id) {
    for (const auto & kv : g_lang) {
        if (kv.second.first == id) {
            return kv.first.c_str();
        }
    }

    WHISPER_LOG_ERROR("%s: unknown language id %d\n", __func__, id);
    return nullptr;
}

// load the model from a ggml file
//
// file format:
//
//   - hparams
//   - pre-computed mel filters
//   - vocab
//   - weights
//
// see the convert-pt-to-ggml.py script for details
//
static bool whisper_model_load(struct whisper_model_loader * loader, whisper_context & wctx) {
    WHISPER_LOG_INFO("%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto & model = wctx.model;
    auto & vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            WHISPER_LOG_ERROR("%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }

    //load hparams
    {
        auto & hparams = model.hparams;

        read_safe(loader, hparams.n_vocab);
        read_safe(loader, hparams.n_audio_ctx);
        read_safe(loader, hparams.n_audio_state);
        read_safe(loader, hparams.n_audio_head);
        read_safe(loader, hparams.n_audio_layer);
        read_safe(loader, hparams.n_text_ctx);
        read_safe(loader, hparams.n_text_state);
        read_safe(loader, hparams.n_text_head);
        read_safe(loader, hparams.n_text_layer);
        read_safe(loader, hparams.n_mels);
        read_safe(loader, hparams.ftype);

        assert(hparams.n_text_state == hparams.n_audio_state);

        std::string mver = "";

        if (hparams.n_audio_layer == 4) {
            model.type = e_model::MODEL_TINY;
        }

        if (hparams.n_audio_layer == 6) {
            model.type = e_model::MODEL_BASE;
        }

        if (hparams.n_audio_layer == 12) {
            model.type = e_model::MODEL_SMALL;
        }

        if (hparams.n_audio_layer == 24) {
            model.type = e_model::MODEL_MEDIUM;
        }

        if (hparams.n_audio_layer == 32) {
            model.type = e_model::MODEL_LARGE;

            if (hparams.n_vocab == 51866) {
                mver = " v3";
            }
        }

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

        // for the big tensors, we have the option to store the data in 16-bit floats or quantized
        // in order to save memory and also to speed up the computation
        wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
        if (wctx.wtype == GGML_TYPE_COUNT) {
            WHISPER_LOG_ERROR("%s: invalid model (bad ftype value %d)\n", __func__, model.hparams.ftype);
            return false;
        }

        WHISPER_LOG_INFO("%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        WHISPER_LOG_INFO("%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        WHISPER_LOG_INFO("%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        WHISPER_LOG_INFO("%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        WHISPER_LOG_INFO("%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        WHISPER_LOG_INFO("%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        WHISPER_LOG_INFO("%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        WHISPER_LOG_INFO("%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        WHISPER_LOG_INFO("%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
        WHISPER_LOG_INFO("%s: n_mels        = %d\n", __func__, hparams.n_mels);
        WHISPER_LOG_INFO("%s: ftype         = %d\n", __func__, model.hparams.ftype);
        WHISPER_LOG_INFO("%s: qntvr         = %d\n", __func__, qntvr);
        WHISPER_LOG_INFO("%s: type          = %d (%s%s)\n", __func__, model.type, g_model_name.at(model.type).c_str(), mver.c_str());
    }

    // load mel filters
    {
        auto & filters = wctx.model.filters;

        read_safe(loader, filters.n_mel);
        read_safe(loader, filters.n_fft);

        filters.data.resize(filters.n_mel * filters.n_fft);
        loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
        BYTESWAP_FILTERS(filters);
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        read_safe(loader, n_vocab);

        //if (n_vocab != model.hparams.n_vocab) {
        //    WHISPER_LOG_ERROR("%s: invalid model file '%s' (bad vocab size %d != %d)\n",
        //            __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
        //    return false;
        //}

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            read_safe(loader, len);

            if (len > 0) {
                tmp.resize(len);
                loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
                word.assign(&tmp[0], tmp.size());
            } else {
                // seems like we have an empty-string token in multi-language models (i = 50256)
                //WHISPER_LOG_WARN("%s: warning: empty-string token in vocab, i = %d\n", __func__, i);
                word = "";
            }

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            //printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
        }

        vocab.n_vocab = model.hparams.n_vocab;
        if (vocab.is_multilingual()) {
            vocab.token_eot++;
            vocab.token_sot++;

            // account for variable number of language tokens
            const int dt = vocab.num_languages() - 98;

            vocab.token_translate  += dt;
            vocab.token_transcribe += dt;
            vocab.token_solm       += dt;
            vocab.token_prev       += dt;
            vocab.token_nosp       += dt;
            vocab.token_not        += dt;
            vocab.token_beg        += dt;
        }

        if (n_vocab < model.hparams.n_vocab) {
            WHISPER_LOG_INFO("%s: adding %d extra tokens\n", __func__, model.hparams.n_vocab - n_vocab);
            for (int i = n_vocab; i < model.hparams.n_vocab; i++) {
                if (i > vocab.token_beg) {
                    word = "[_TT_" + std::to_string(i - vocab.token_beg) + "]";
                } else if (i == vocab.token_eot) {
                    word = "[_EOT_]";
                } else if (i == vocab.token_sot) {
                    word = "[_SOT_]";
                } else if (i == vocab.token_translate) {
                    word = "[_TRANSLATE_]";
                } else if (i == vocab.token_transcribe) {
                    word = "[_TRANSCRIBE_]";
                } else if (i == vocab.token_solm) {
                    word = "[_SOLM_]";
                } else if (i == vocab.token_prev) {
                    word = "[_PREV_]";
                } else if (i == vocab.token_nosp) {
                    word = "[_NOSP_]";
                } else if (i == vocab.token_not) {
                    word = "[_NOT_]";
                } else if (i == vocab.token_beg) {
                    word = "[_BEG_]";
                } else if (i > vocab.token_sot && i <= vocab.token_sot + vocab.num_languages()) {
                    word = "[_LANG_" + std::string(whisper_lang_str(i - vocab.token_sot - 1)) + "]";
                } else {
                    word = "[_extra_token_" + std::to_string(i) + "]";
                }
                vocab.token_to_id[word] = i;
                vocab.id_to_token[i] = word;
            }
        }

        WHISPER_LOG_INFO("%s: n_langs       = %d\n", __func__, vocab.num_languages());
    }

    const ggml_type wtype = wctx.wtype;
    const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16; // conv type

    // create the ggml context
    {
        const auto & hparams = model.hparams;

        const int n_audio_layer = hparams.n_audio_layer;
        const int n_text_layer  = hparams.n_text_layer;

        const size_t n_tensors = 10 /* input */ + 15 + 15*n_audio_layer + 24*n_text_layer;

        struct ggml_init_params params = {
            /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            WHISPER_LOG_ERROR("%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare tensors for the weights
    {
        auto & ctx = model.ctx;

        const auto & hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;

        const int n_audio_ctx   = hparams.n_audio_ctx;
        const int n_audio_state = hparams.n_audio_state;
        const int n_audio_layer = hparams.n_audio_layer;

        const int n_text_ctx   = hparams.n_text_ctx;
        const int n_text_state = hparams.n_text_state;
        const int n_text_layer = hparams.n_text_layer;

        const int n_mels = hparams.n_mels;

        model.layers_encoder.resize(n_audio_layer);
        model.layers_decoder.resize(n_text_layer);

        // encoder
        {
            model.e_pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state, n_audio_ctx);

            model.e_conv_1_w     = ggml_new_tensor_3d(ctx, vtype,         3, n_mels,     n_audio_state);
            model.e_conv_1_b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,         1,     n_audio_state);

            model.e_conv_2_w     = ggml_new_tensor_3d(ctx, vtype,         3, n_audio_state, n_audio_state);
            model.e_conv_2_b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,                1, n_audio_state);

            model.e_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
            model.e_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);

            // map by name
            model.tensors["encoder.positional_embedding"] = model.e_pe;

            model.tensors["encoder.conv1.weight"]         = model.e_conv_1_w;
            model.tensors["encoder.conv1.bias"]           = model.e_conv_1_b;

            model.tensors["encoder.conv2.weight"]         = model.e_conv_2_w;
            model.tensors["encoder.conv2.bias"]           = model.e_conv_2_b;

            model.tensors["encoder.ln_post.weight"]       = model.e_ln_w;
            model.tensors["encoder.ln_post.bias"]         = model.e_ln_b;

            for (int i = 0; i < n_audio_layer; ++i) {
                auto & layer = model.layers_encoder[i];

                layer.mlp_ln_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);
                layer.mlp_ln_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.mlp_0_w     = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, 4*n_audio_state);
                layer.mlp_0_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_audio_state);

                layer.mlp_1_w     = ggml_new_tensor_2d(ctx, wtype,         4*n_audio_state, n_audio_state);
                layer.mlp_1_b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.attn_ln_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);
                layer.attn_ln_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.attn_q_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
                layer.attn_q_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.attn_k_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);

                layer.attn_v_w    = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
                layer.attn_v_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                layer.attn_ln_1_w = ggml_new_tensor_2d(ctx, wtype,           n_audio_state, n_audio_state);
                layer.attn_ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_audio_state);

                // map by name
                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp_ln.weight"]     = layer.mlp_ln_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp_ln.bias"]       = layer.mlp_ln_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.0.weight"]      = layer.mlp_0_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.0.bias"]        = layer.mlp_0_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.2.weight"]      = layer.mlp_1_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.2.bias"]        = layer.mlp_1_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn_ln.weight"]    = layer.attn_ln_0_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".attn_ln.bias"]      = layer.attn_ln_0_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.query.weight"] = layer.attn_q_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.query.bias"]   = layer.attn_q_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.key.weight"]   = layer.attn_k_w;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.value.weight"] = layer.attn_v_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.value.bias"]   = layer.attn_v_b;

                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.out.weight"]   = layer.attn_ln_1_w;
                model.tensors["encoder.blocks." + std::to_string(i) + ".attn.out.bias"]     = layer.attn_ln_1_b;
            }
        }

        // decoder
        {
            model.d_pe   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_text_state, n_text_ctx);

            model.d_te   = ggml_new_tensor_2d(ctx, wtype,         n_text_state, n_vocab);

            model.d_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
            model.d_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);

            // map by name
            model.tensors["decoder.positional_embedding"]   = model.d_pe;

            model.tensors["decoder.token_embedding.weight"] = model.d_te;

            model.tensors["decoder.ln.weight"]              = model.d_ln_w;
            model.tensors["decoder.ln.bias"]                = model.d_ln_b;

            for (int i = 0; i < n_text_layer; ++i) {
                auto & layer = model.layers_decoder[i];

                layer.mlp_ln_w          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);
                layer.mlp_ln_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.mlp_0_w           = ggml_new_tensor_2d(ctx, wtype,           n_text_state, 4*n_text_state);
                layer.mlp_0_b           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_text_state);

                layer.mlp_1_w           = ggml_new_tensor_2d(ctx, wtype,         4*n_text_state, n_text_state);
                layer.mlp_1_b           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.attn_ln_0_w       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);
                layer.attn_ln_0_b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.attn_q_w          = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.attn_q_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.attn_k_w          = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);

                layer.attn_v_w          = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.attn_v_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.attn_ln_1_w       = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.attn_ln_1_b       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.cross_attn_ln_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);
                layer.cross_attn_ln_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.cross_attn_q_w    = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.cross_attn_q_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.cross_attn_k_w    = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);

                layer.cross_attn_v_w    = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.cross_attn_v_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                layer.cross_attn_ln_1_w = ggml_new_tensor_2d(ctx, wtype,           n_text_state, n_text_state);
                layer.cross_attn_ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_text_state);

                // map by name
                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp_ln.weight"]           = layer.mlp_ln_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp_ln.bias"]             = layer.mlp_ln_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.0.weight"]            = layer.mlp_0_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.0.bias"]              = layer.mlp_0_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.2.weight"]            = layer.mlp_1_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.2.bias"]              = layer.mlp_1_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn_ln.weight"]          = layer.attn_ln_0_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".attn_ln.bias"]            = layer.attn_ln_0_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.query.weight"]       = layer.attn_q_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.query.bias"]         = layer.attn_q_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.key.weight"]         = layer.attn_k_w;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.value.weight"]       = layer.attn_v_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.value.bias"]         = layer.attn_v_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.out.weight"]         = layer.attn_ln_1_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".attn.out.bias"]           = layer.attn_ln_1_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn_ln.weight"]    = layer.cross_attn_ln_0_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn_ln.bias"]      = layer.cross_attn_ln_0_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.query.weight"] = layer.cross_attn_q_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.query.bias"]   = layer.cross_attn_q_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.key.weight"]   = layer.cross_attn_k_w;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.value.weight"] = layer.cross_attn_v_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.value.bias"]   = layer.cross_attn_v_b;

                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.out.weight"]   = layer.cross_attn_ln_1_w;
                model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.out.bias"]     = layer.cross_attn_ln_1_b;
            }
        }
    }

    // allocate tensors in the backend buffers
    model.buffer = ggml_backend_alloc_ctx_tensors_from_buft(model.ctx, whisper_default_buffer_type(wctx.params));
    if (!model.buffer) {
        WHISPER_LOG_ERROR("%s: failed to allocate memory for the model\n", __func__);
        return false;
    }

    size_t size_main = ggml_backend_buffer_get_size(model.buffer);
    WHISPER_LOG_INFO("%s: %8s total size = %8.2f MB\n", __func__, ggml_backend_buffer_name(model.buffer), size_main / 1e6);

    // load weights
    {
        size_t total_size = 0;

        model.n_loaded = 0;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            read_safe(loader, n_dims);
            read_safe(loader, length);
            read_safe(loader, ttype);

            if (loader->eof(loader->context)) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[4] = { 1, 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                read_safe(loader, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> tmp(length); // create a buffer
            loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
            name.assign(&tmp[0], tmp.size());

            if (model.tensors.find(name) == model.tensors.end()) {
                WHISPER_LOG_ERROR("%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];

            if (ggml_nelements(tensor) != nelements) {
                WHISPER_LOG_ERROR("%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                WHISPER_LOG_ERROR("%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n",
                        __func__, ne[0], ne[1], ne[2], (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2]);
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
                WHISPER_LOG_ERROR("%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d], expected [%d, %d, %d]\n",
                        __func__, name.data(), (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2], ne[0], ne[1], ne[2]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                WHISPER_LOG_ERROR("%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            //ggml_backend_t backend = wctx.backend;

            //printf("%s: [%5.5s] %s\n", __func__, ggml_backend_name(backend), name.c_str());

            if (ggml_backend_buffer_is_host(model.buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                loader->read(loader->context, tensor->data, ggml_nbytes(tensor));
                BYTESWAP_TENSOR(tensor);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));

                loader->read(loader->context, read_buf.data(), read_buf.size());

                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            //printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype), ggml_nbytes(tensor)/1e6);
            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        WHISPER_LOG_INFO("%s: model size    = %7.2f MB\n", __func__, total_size/1e6);

        if (model.n_loaded == 0) {
            WHISPER_LOG_WARN("%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        } else if (model.n_loaded != (int) model.tensors.size()) {
            WHISPER_LOG_ERROR("%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
            return false;
        }
    }

    ggml_backend_buffer_set_usage(model.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    wctx.t_load_us = ggml_time_us() - t_start_us;

    return true;
}

static bool whisper_encode_external(const whisper_state & wstate) {
    GGML_UNUSED(wstate);

#ifndef WHISPER_USE_COREML
    const bool use_coreml = false;
#else
    const bool use_coreml = wstate.ctx_coreml != nullptr;
#endif

#ifndef WHISPER_USE_OPENVINO
    const bool use_openvino = false;
#else
    const bool use_openvino = wstate.ctx_openvino != nullptr;
#endif

    return use_coreml || use_openvino;
}

// Build ggml graph for mel spectrogram: DFT via matmul + filterbank
//
// Input:  windowed_frames (frame_size, n_frames) — Hann-windowed audio frames
// Output: mel_out (n_mel, n_frames) — log mel spectrogram
//
// Graph:  re = dft_cos @ frames     (n_fft_bins, n_frames)
//         im = dft_sin @ frames     (n_fft_bins, n_frames)
//         power = re*re + im*im     (n_fft_bins, n_frames)
//         mel = filters @ power     (n_mel, n_frames)
//         mel = log(max(mel, 1e-10))
static struct ggml_cgraph * whisper_build_graph_mel(
        whisper_context & wctx,
          whisper_state & wstate) {
    const int n_fft_bins  = wctx.model.filters.n_fft;  // 201
    const int frame_size  = WHISPER_N_FFT;              // 400
    const int n_mel       = wctx.model.filters.n_mel;   // 80
    const int n_frames    = wstate.mel_n_frames;

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_mel.meta.size(),
        /*.mem_buffer =*/ wstate.sched_mel.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * frames = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, frame_size, n_frames);
    ggml_set_name(frames, "mel_frames");
    ggml_set_input(frames);

    struct ggml_tensor * re = ggml_mul_mat(ctx0, wstate.mel_dft_cos, frames);
    struct ggml_tensor * im = ggml_mul_mat(ctx0, wstate.mel_dft_sin, frames);

    struct ggml_tensor * power = ggml_add(ctx0, ggml_mul(ctx0, re, re), ggml_mul(ctx0, im, im));

    struct ggml_tensor * mel_raw = ggml_mul_mat(ctx0, wstate.mel_filters_tensor, power);

    // raw log10 mel; whisper-specific dynamic range compression applied post-graph
    struct ggml_tensor * mel_out = ggml_scale(ctx0, ggml_log(ctx0, ggml_clamp(ctx0, mel_raw, 1e-10f, FLT_MAX)), 1.0f / logf(10.0f));

    ggml_set_name(mel_out, "mel_out");
    ggml_set_output(mel_out);

    ggml_build_forward_expand(gf, mel_out);
    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * whisper_build_graph_conv(
        whisper_context & wctx,
          whisper_state & wstate) {
    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;

    const int n_ctx   = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state; GGML_UNUSED(n_state);

    const int n_mels = hparams.n_mels;

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_conv.meta.size(),
        /*.mem_buffer =*/ wstate.sched_conv.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2*n_ctx, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    struct ggml_tensor * cur = nullptr;

    if (!whisper_encode_external(wstate)) {
        // convolution + gelu
        {
            cur = ggml_conv_1d_ph(ctx0, model.e_conv_1_w, mel, 1, 1);
            cur = ggml_add(ctx0, cur, model.e_conv_1_b);

            cur = ggml_gelu(ctx0, cur);

            cur = ggml_conv_1d_ph(ctx0, model.e_conv_2_w, cur, 2, 1);
            cur = ggml_add(ctx0, cur, model.e_conv_2_b);

            cur = ggml_gelu(ctx0, cur);
        }

        ggml_set_name(cur, "embd_conv");
        wstate.embd_conv = cur;
    } else {
        ggml_build_forward_expand(gf, mel);

        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx);
        ggml_set_input(cur); // the external encoder will write into this tensor

        ggml_set_name(cur, "embd_enc");
        wstate.embd_enc = cur;
    }

    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static struct ggml_cgraph * whisper_build_graph_encoder(
        whisper_context & wctx,
          whisper_state & wstate) {
    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;

    const int n_ctx   = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    const int n_head  = hparams.n_audio_head;
    const int n_layer = hparams.n_audio_layer;

    const int n_state_head = n_state/n_head;

    auto & kv_pad = wstate.kv_pad;

    WHISPER_ASSERT(!!kv_pad.buffer);

    const int n_ctx_pad = GGML_PAD(n_ctx, 256);

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_encode.meta.size(),
        /*.mem_buffer =*/ wstate.sched_encode.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, WHISPER_MAX_NODES, false);

    struct ggml_tensor * cur = ggml_view_tensor(ctx0, wstate.embd_conv);

    const float KQscale = 1.0f/sqrtf(float(n_state_head));

    // ===================================================================
    // NOTE: experimenting with partial evaluation of the encoder (ignore)
    //static int iter = -1;
    //const int n_iter = 1500/n_ctx;

    //iter = (iter + 1) % n_iter;

    //if (iter == 0) {
    //    memset(model.memory_cross_k->data, 0, ggml_nbytes(model.memory_cross_k));
    //    memset(model.memory_cross_v->data, 0, ggml_nbytes(model.memory_cross_v));
    //}

    static int iter = 0;

    const size_t e_pe_stride = model.e_pe->ne[0]*ggml_element_size(model.e_pe);
    const size_t e_pe_offset = model.e_pe->ne[0]*ggml_element_size(model.e_pe)*n_ctx*iter;

    struct ggml_tensor * e_pe = ggml_view_2d(ctx0, model.e_pe, model.e_pe->ne[0], n_ctx, e_pe_stride, e_pe_offset);
    cur = ggml_add(ctx0, e_pe, ggml_cont(ctx0, ggml_transpose(ctx0, cur)));

    // ===================================================================

    // original:
    //cur = ggml_add(ctx0, model.e_pe, ggml_transpose(ctx0, cur));

    struct ggml_tensor * inpL = cur;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers_encoder[il];

        // norm
        {
            cur = ggml_norm(ctx0, inpL, hparams.eps);

            // cur = ln_0_w*cur + ln_0_b
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0, cur, layer.attn_ln_0_w),
                    layer.attn_ln_0_b);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0,
                    layer.attn_q_w,
                    cur);

            Qcur = ggml_add(ctx0, Qcur, layer.attn_q_b);

            //Qcur = ggml_scale(ctx0, Qcur, pow(float(n_state_head), -0.25));

            // note: no bias for Key
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0,
                    layer.attn_k_w,
                    cur);

            //Kcur = ggml_scale(ctx0, Kcur, pow(float(n_state_head), -0.25));

            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0,
                    layer.attn_v_w,
                    cur);

            Vcur = ggml_add(ctx0, Vcur, layer.attn_v_b);

            // ------

            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0, Qcur, n_state_head, n_head, n_ctx),
                        0, 2, 1, 3);

            if (wctx.params.flash_attn) {
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, ggml_view_1d(ctx0, kv_pad.k, n_ctx*n_state, 0)));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, ggml_view_1d(ctx0, kv_pad.v, n_ctx*n_state, 0)));

                struct ggml_tensor * K =
                    ggml_view_3d(ctx0, kv_pad.k,
                            n_state_head, n_ctx_pad, n_head,
                            ggml_element_size(kv_pad.k)*n_state,
                            ggml_element_size(kv_pad.k)*n_state_head,
                            0);

                struct ggml_tensor * V =
                    ggml_view_3d(ctx0, kv_pad.v,
                            n_state_head, n_ctx_pad, n_head,
                            ggml_element_size(kv_pad.v)*n_state,
                            ggml_element_size(kv_pad.v)*n_state_head,
                            0);

                cur = ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, KQscale, 0.0f, 0.0f);

                cur = ggml_reshape_2d(ctx0, cur, n_state, n_ctx);
            } else {
                struct ggml_tensor * K =
                    ggml_permute(ctx0,
                            ggml_cast(ctx0,
                                ggml_reshape_3d(ctx0, Kcur, n_state_head, n_head, n_ctx),
                                wctx.itype),
                            0, 2, 1, 3);

                // K * Q
                struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

                struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(ctx0, KQ, nullptr, KQscale, 0.0f);

                struct ggml_tensor * V =
                    ggml_cast(ctx0,
                            ggml_permute(ctx0,
                                ggml_reshape_3d(ctx0,
                                    Vcur,
                                    n_state_head, n_head, n_ctx),
                                1, 2, 0, 3),
                            wctx.itype);

                struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

                struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

                cur = ggml_cont_2d(ctx0, KQV_merged, n_state, n_ctx);
            }
        }

        // projection
        {
            cur = ggml_mul_mat(ctx0,
                    layer.attn_ln_1_w,
                    cur);

            cur = ggml_add(ctx0, cur, layer.attn_ln_1_b);
        }

        // add the input
        cur = ggml_add(ctx0, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF, hparams.eps);

                // cur = mlp_ln_w*cur + mlp_ln_b
                cur = ggml_add(ctx0,
                        ggml_mul(ctx0, cur, layer.mlp_ln_w),
                        layer.mlp_ln_b);
            }

            // fully connected
            cur = ggml_mul_mat(ctx0,
                    layer.mlp_0_w,
                    cur);

            cur = ggml_add(ctx0, cur, layer.mlp_0_b);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            cur = ggml_mul_mat(ctx0,
                    layer.mlp_1_w,
                    cur);

            cur = ggml_add(ctx0, cur, layer.mlp_1_b);
        }

        inpL = ggml_add(ctx0, cur, inpFF);
    }

    cur = inpL;

    // norm
    {
        cur = ggml_norm(ctx0, cur, hparams.eps);

        // cur = ln_f_g*cur + ln_f_b
        cur = ggml_add(ctx0,
                ggml_mul(ctx0, cur, model.e_ln_w),
                model.e_ln_b);
    }

    ggml_build_forward_expand(gf, cur);

    wstate.embd_enc = cur;

    //ggml_graph_print(gf);

    ////////////////////////////////////////////////////////////////////////////

    //printf("%s: used_mem = %f MB, %f MB, %f MB %f MB %f MB\n", __func__,
    //        ggml_used_mem(ctx0)/1e6,
    //        wstate.get_buf_max_mem(0)/1e6,
    //        wstate.get_buf_max_mem(1)/1e6,
    //        wstate.get_buf_max_mem(2)/1e6,
    //        wstate.get_buf_max_mem(3)/1e6);

    ggml_free(ctx0);

    return gf;
}

// pre-compute cross-attention memory
static struct ggml_cgraph * whisper_build_graph_cross(
        whisper_context & wctx,
          whisper_state & wstate) {
    const auto & model   = wctx.model;
    const auto & hparams = model.hparams;

    const int n_ctx   = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : hparams.n_audio_ctx;
    const int n_state = hparams.n_audio_state;
    const int n_head  = hparams.n_audio_head;

    const int n_state_head = n_state/n_head;

    const int n_ctx_pad = GGML_PAD(n_ctx, 256);

    struct ggml_init_params params = {
        /*.mem_size   =*/ wstate.sched_cross.meta.size(),
        /*.mem_buffer =*/ wstate.sched_cross.meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur = ggml_view_tensor(ctx0, wstate.embd_enc);

    const float  Kscale = pow(float(n_state_head), -0.25);

    for (int il = 0; il < model.hparams.n_text_layer; ++il) {
        auto & layer = model.layers_decoder[il];

        struct ggml_tensor * Kcross = ggml_mul_mat(ctx0,
                layer.cross_attn_k_w,
                cur);

        Kcross = ggml_scale(ctx0, Kcross, Kscale);

        struct ggml_tensor * Vcross = ggml_mul_mat(ctx0,
                layer.cross_attn_v_w,
                cur);

        Vcross = ggml_add(ctx0,
                    Vcross,
                    layer.cross_attn_v_b);

        struct ggml_tensor * k;
        struct ggml_tensor * v;

        if (wctx.params.flash_attn) {
            k = ggml_view_1d(ctx0, wstate.kv_cross.k, n_state*n_ctx,
                    (ggml_element_size(wstate.kv_cross.k)*n_state)*(il*n_ctx_pad));

            v = ggml_view_1d(ctx0, wstate.kv_cross.v, n_state*n_ctx,
                    (ggml_element_size(wstate.kv_cross.v)*n_state)*(il*n_ctx_pad));
        } else {
            Vcross = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcross, n_state, n_ctx));

            k = ggml_view_1d(ctx0, wstate.kv_cross.k, n_state*n_ctx,
                    (ggml_element_size(wstate.kv_cross.k)*n_state)*(il*n_ctx));

            v = ggml_view_2d(ctx0, wstate.kv_cross.v, n_ctx, n_state,
                    (   n_ctx)*ggml_element_size(wstate.kv_cross.v),
                    (il*n_ctx)*ggml_element_size(wstate.kv_cross.v)*n_state);
        }

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcross, k));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcross, v));
    }

    //ggml_graph_print(gf);

    ggml_free(ctx0);

    return gf;
}

// evaluate the encoder with the given state
//
// given audio recording (more specifically, its log mel spectrogram), runs forward pass of the encoder
// part of the transformer model and returns the encoded features
//
//   - wctx:      the model
//   - wstate:     the state of the encoder
//   - n_threads:  number of threads to use
//   - mel_offset: offset in the mel spectrogram (i.e. audio offset)
//
static bool whisper_encode_internal(
        whisper_context & wctx,
          whisper_state & wstate,
              const int   mel_offset,
              const int   n_threads,
    ggml_abort_callback   abort_callback,
                   void * abort_callback_data) {
    const int64_t t_start_us = ggml_time_us();

    // conv
    {
        auto & sched = wstate.sched_conv.sched;

        ggml_cgraph * gf = whisper_build_graph_conv(wctx, wstate);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        struct ggml_tensor * mel = ggml_graph_get_tensor(gf, "mel");

        // set the input
        {
            const auto & mel_inp = wstate.mel;
            const int n_ctx      = wstate.exp_n_audio_ctx > 0 ? wstate.exp_n_audio_ctx : wctx.model.hparams.n_audio_ctx;

            assert(mel->type == GGML_TYPE_F32);
            assert(mel_inp.n_mel == wctx.model.hparams.n_mels);

            wstate.inp_mel.resize(ggml_nelements(mel));

            float * dst = wstate.inp_mel.data();
            memset(dst, 0, ggml_nbytes(mel));

            const int i0 = std::min(mel_offset,           mel_inp.n_len);
            const int i1 = std::min(mel_offset + 2*n_ctx, mel_inp.n_len);

            for (int j = 0; j < mel_inp.n_mel; ++j) {
                for (int i = i0; i < i1; ++i) {
                    dst[j*2*n_ctx + (i - i0)] = mel_inp.data[j*mel_inp.n_len + i];
                }
            }

            ggml_backend_tensor_set(mel, wstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));
        }

        if (!whisper_encode_external(wstate)) {
            if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
                return false;
            }
        } else {
#if defined(WHISPER_USE_COREML)
            whisper_coreml_encode(wstate.ctx_coreml, mel->ne[0], mel->ne[1], (float *) mel->data, (float *) wstate.embd_enc->data);
#elif defined(WHISPER_USE_OPENVINO)
            whisper_openvino_encode(wstate.ctx_openvino, mel, wstate.embd_enc);
#endif
        }
    }
    // encoder
    if (!whisper_encode_external(wstate)) {
        auto & sched = wstate.sched_encode.sched;

        ggml_cgraph * gf = whisper_build_graph_encoder(wctx, wstate);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }
    // cross — only needed for decoding, skip if not requested
    if (!wstate.skip_cross) {
        auto & sched = wstate.sched_cross.sched;

        ggml_cgraph * gf = whisper_build_graph_cross(wctx, wstate);

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            // should never happen as we pre-allocate the memory
            return false;
        }

        if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
            return false;
        }
    }

    wstate.t_encode_us += ggml_time_us() - t_start_us;
    wstate.n_encode++;

    return !(abort_callback && abort_callback(abort_callback_data));
}

//
// Mel spectrogram
//

namespace {
struct whisper_global_cache {
    // Hann window (Use cosf to eliminate difference)
    // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
    // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
    float hann_window[WHISPER_N_FFT];

    whisper_global_cache() {
        for (int i = 0; i < WHISPER_N_FFT; i++) {
            hann_window[i] = 0.5f * (1.0f - cosf((2.0f * (float)M_PI * i) / WHISPER_N_FFT));
        }
    }
} global_cache;
}

// ggml-accelerated mel spectrogram: DFT via batched matmul + filterbank matmul.
// Replaces the per-frame scalar FFT with three SIMD-optimized matmuls.
static bool log_mel_spectrogram(
              whisper_context & wctx,
              whisper_state & wstate,
              const float * samples,
              const int   n_samples,
              const int   n_threads) {
    const int64_t t_start_us = ggml_time_us();

    const int frame_size = WHISPER_N_FFT;
    const int frame_step = WHISPER_HOP_LENGTH;
    const int n_mel      = wctx.model.filters.n_mel;
    const float * hann   = global_cache.hann_window;

    // Reflective padding (left edge only, matching original whisper.cpp)
    const int pad = frame_size / 2;
    const int n_padded = n_samples + 2 * pad;

    auto & padded = wstate.mel_samples_padded;
    padded.resize(n_padded);
    std::reverse_copy(samples + 1, samples + 1 + pad, padded.begin());
    std::copy(samples, samples + n_samples, padded.begin() + pad);
    // Only the right-pad region needs zeroing (left pad + samples already written above)
    std::fill(padded.begin() + pad + n_samples, padded.end(), 0.0f);

    const int n_frames = (n_padded - frame_size) / frame_step;

    wstate.mel.n_mel     = n_mel;
    wstate.mel.n_len     = n_frames;
    wstate.mel.n_len_org = n_frames;

    // Build windowed frames matrix: each column is one Hann-windowed frame
    auto & frames = wstate.mel_frames_buf;
    frames.resize(frame_size * n_frames);
    for (int i = 0; i < n_frames; i++) {
        const int offset = i * frame_step;
        float * col = frames.data() + i * frame_size;
        const int valid = std::min(frame_size, n_samples + pad - offset);
        for (int j = 0; j < valid; j++) {
            col[j] = hann[j] * padded[offset + j];
        }
        for (int j = valid; j < frame_size; j++) {
            col[j] = 0.0f;
        }
    }

    wstate.mel_n_frames = n_frames;

    auto & sched = wstate.sched_mel.sched;
    ggml_backend_sched_reset(sched);

    ggml_cgraph * gf = whisper_build_graph_mel(wctx, wstate);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        WHISPER_LOG_ERROR("%s: failed to alloc mel graph\n", __func__);
        return false;
    }

    struct ggml_tensor * frames_tensor = ggml_graph_get_tensor(gf, "mel_frames");
    ggml_backend_tensor_set(frames_tensor, frames.data(), 0, frame_size * n_frames * sizeof(float));

    if (!ggml_graph_compute_helper(sched, gf, n_threads)) {
        WHISPER_LOG_ERROR("%s: failed to compute mel graph\n", __func__);
        return false;
    }

    // Transpose: ggml output is column-major (n_mel per frame),
    // whisper expects row-major per mel bin: mel.data[bin * n_len + frame]
    struct ggml_tensor * mel_out = ggml_graph_get_tensor(gf, "mel_out");
    auto & raw = wstate.mel_raw_buf;
    raw.resize(n_mel * n_frames);
    ggml_backend_tensor_get(mel_out, raw.data(), 0, n_mel * n_frames * sizeof(float));

    auto & mel = wstate.mel;
    mel.data.resize(n_mel * n_frames);
    // j-outer for sequential writes into mel.data
    for (int j = 0; j < n_mel; j++) {
        for (int i = 0; i < n_frames; i++) {
            mel.data[j * n_frames + i] = raw[j + i * n_mel];
        }
    }

    // Dynamic range compression (whisper-specific): find max, clamp, normalize in one pass
    float mmax = -1e20f;
    for (int i = 0; i < n_mel * n_frames; i++) {
        if (mel.data[i] > mmax) mmax = mel.data[i];
    }
    mmax -= 8.0f;
    for (int i = 0; i < n_mel * n_frames; i++) {
        mel.data[i] = (std::max(mel.data[i], mmax) + 4.0f) / 4.0f;
    }

    wstate.t_mel_us += ggml_time_us() - t_start_us;
    return true;
}

//
// interface implementation
//

#ifdef WHISPER_USE_COREML
// replace .bin with -encoder.mlmodelc
static std::string whisper_get_coreml_path_encoder(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    // match "-qx_x"
    pos = path_bin.rfind('-');
    if (pos != std::string::npos) {
        auto sub = path_bin.substr(pos);
        if (sub.size() == 5 && sub[1] == 'q' && sub[3] == '_') {
            path_bin = path_bin.substr(0, pos);
        }
    }

    path_bin += "-encoder.mlmodelc";

    return path_bin;
}
#endif

#ifdef WHISPER_USE_OPENVINO
// replace .bin with-encoder-openvino.xml
static std::string whisper_openvino_get_path_encoder(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    path_bin += "-encoder-openvino.xml";

    return path_bin;
}

static std::string whisper_openvino_get_path_cache(std::string path_bin) {
    auto pos = path_bin.rfind('.');
    if (pos != std::string::npos) {
        path_bin = path_bin.substr(0, pos);
    }

    path_bin += "-encoder-openvino-cache";

    return path_bin;
}
#endif

struct whisper_state * whisper_init_state(whisper_context * ctx) {
    whisper_state * state = new whisper_state;

    state->backends = whisper_backend_init(ctx->params);
    if (state->backends.empty()) {
        WHISPER_LOG_ERROR("%s: whisper_backend_init() failed\n", __func__);
        whisper_free_state(state);
        return nullptr;
    }

    // cross-attention KV cache
    if (!whisper_kv_cache_init(state->kv_cross, state->backends[0], ctx->itype,
                ctx->model.hparams.n_text_state,
                ctx->model.hparams.n_text_layer,
                GGML_PAD(ctx->model.hparams.n_audio_ctx, 256))) {
        WHISPER_LOG_ERROR("%s: whisper_kv_cache_init() failed for cross-attention cache\n", __func__);
        whisper_free_state(state);
        return nullptr;
    }

    {
        const size_t memory_size = ggml_nbytes(state->kv_cross.k) + ggml_nbytes(state->kv_cross.v);
        WHISPER_LOG_INFO("%s: kv cross size = %7.2f MB\n", __func__, memory_size / 1e6);
    }

    // padded buffer for flash-attention
    if (!whisper_kv_cache_init(state->kv_pad, state->backends[0], ctx->itype,
                ctx->model.hparams.n_audio_state,
                1,
                GGML_PAD(ctx->model.hparams.n_audio_ctx, 256))) {
        WHISPER_LOG_ERROR("%s: whisper_kv_cache_init() failed for kv_pad cache\n", __func__);
        whisper_free_state(state);
        return nullptr;
    }

    {
        const size_t memory_size = ggml_nbytes(state->kv_pad.k) + ggml_nbytes(state->kv_pad.v);
        WHISPER_LOG_INFO("%s: kv pad  size  = %7.2f MB\n", __func__, memory_size / 1e6);
    }

#ifdef WHISPER_USE_COREML
    const auto path_coreml = whisper_get_coreml_path_encoder(ctx->path_model);

    WHISPER_LOG_INFO("%s: loading Core ML model from '%s'\n", __func__, path_coreml.c_str());
    WHISPER_LOG_INFO("%s: first run on a device may take a while ...\n", __func__);

    state->ctx_coreml = whisper_coreml_init(path_coreml.c_str());
    if (!state->ctx_coreml) {
        WHISPER_LOG_ERROR("%s: failed to load Core ML model from '%s'\n", __func__, path_coreml.c_str());
#ifndef WHISPER_COREML_ALLOW_FALLBACK
        whisper_free_state(state);
        return nullptr;
#endif
    } else {
        WHISPER_LOG_INFO("%s: Core ML model loaded\n", __func__);
    }
#endif

    state->batch = whisper_batch_init(ctx->model.hparams.n_text_ctx, WHISPER_MAX_DECODERS);

    // ggml mel spectrogram: precompute DFT basis matrices and mel filterbank as ggml tensors
    {
        const int frame_size = WHISPER_N_FFT;       // 400
        const int n_fft_bins = 1 + frame_size / 2;  // 201
        const int n_mel = ctx->model.filters.n_mel;  // 80

        // Determine n_frames for the expected buffer size.
        // For a 2s buffer at 16kHz: 32000 + 400 (pad) samples, hop=160 → 200 frames.
        // We use a generous upper bound; the graph handles any n_frames up to this.
        const int max_samples = WHISPER_SAMPLE_RATE * 30 + frame_size;  // 30s max (whisper limit)
        state->mel_n_frames = (max_samples - frame_size) / WHISPER_HOP_LENGTH;

        // Create ggml context for the persistent basis tensors (3 tensors)
        // The ctx_buf must outlive the ggml_context so tensor metadata persists
        state->mel_basis_ctx_buf.resize(3 * ggml_tensor_overhead());
        struct ggml_init_params params = {
            /*.mem_size   =*/ state->mel_basis_ctx_buf.size(),
            /*.mem_buffer =*/ state->mel_basis_ctx_buf.data(),
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx_basis = ggml_init(params);

        // ggml_mul_mat(A, B) computes A^T @ B. To get (n_fft_bins, n_frames) output
        // from input (frame_size, n_frames), A must be (frame_size, n_fft_bins).
        state->mel_dft_cos = ggml_new_tensor_2d(ctx_basis, GGML_TYPE_F32, frame_size, n_fft_bins);
        state->mel_dft_sin = ggml_new_tensor_2d(ctx_basis, GGML_TYPE_F32, frame_size, n_fft_bins);
        // For filterbank: ggml_mul_mat(A, B) = A^T @ B. To get (n_mel, n_frames) from (n_fft_bins, n_frames),
        // A must be (n_fft_bins, n_mel).
        state->mel_filters_tensor = ggml_new_tensor_2d(ctx_basis, GGML_TYPE_F32, n_fft_bins, n_mel);

        state->mel_basis_buffer = ggml_backend_alloc_ctx_tensors(ctx_basis, state->backends[0]);
        if (!state->mel_basis_buffer) {
            WHISPER_LOG_ERROR("%s: failed to allocate mel basis buffer\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }

        // Fill DFT cosine basis: cos[k][n] = cos(2π·k·n/N) for k=0..n_fft_bins-1, n=0..frame_size-1
        // Layout: (frame_size, n_fft_bins) row-major = for each k, frame_size values
        {
            std::vector<float> cos_data(n_fft_bins * frame_size);
            std::vector<float> sin_data(n_fft_bins * frame_size);
            for (int k = 0; k < n_fft_bins; k++) {
                for (int n = 0; n < frame_size; n++) {
                    double angle = 2.0 * M_PI * k * n / frame_size;
                    cos_data[k * frame_size + n] =  (float)cos(angle);
                    sin_data[k * frame_size + n] = -(float)sin(angle);  // negative for DFT convention
                }
            }
            ggml_backend_tensor_set(state->mel_dft_cos, cos_data.data(), 0, cos_data.size() * sizeof(float));
            ggml_backend_tensor_set(state->mel_dft_sin, sin_data.data(), 0, sin_data.size() * sizeof(float));
        }

        // Filterbank: same layout as model filters (n_mel, n_fft_bins) row-major
        ggml_backend_tensor_set(state->mel_filters_tensor,
            ctx->model.filters.data.data(), 0,
            n_mel * n_fft_bins * sizeof(float));

        ggml_free(ctx_basis);

        WHISPER_LOG_INFO("%s: mel basis (DFT+filters) = %7.2f MB\n", __func__,
                (ggml_nbytes(state->mel_dft_cos) + ggml_nbytes(state->mel_dft_sin) +
                 ggml_nbytes(state->mel_filters_tensor)) / 1e6);
    }

    // mel spectrogram allocator
    {
        bool ok = whisper_sched_graph_init(state->sched_mel, state->backends,
                [&]() {
                    return whisper_build_graph_mel(*ctx, *state);
                });

        if (!ok) {
            WHISPER_LOG_ERROR("%s: failed to init mel allocator\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }

        WHISPER_LOG_INFO("%s: compute buffer (mel)    = %7.2f MB\n", __func__, whisper_sched_size(state->sched_mel) / 1e6);
    }

    // conv allocator
    {
        bool ok = whisper_sched_graph_init(state->sched_conv, state->backends,
                [&]() {
                    return whisper_build_graph_conv(*ctx, *state);
                });

        if (!ok) {
            WHISPER_LOG_ERROR("%s: failed to init conv allocator\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }

        WHISPER_LOG_INFO("%s: compute buffer (conv)   = %7.2f MB\n", __func__, whisper_sched_size(state->sched_conv) / 1e6);
    }

    // encoder allocator
    if (!whisper_encode_external(*state)) {
        bool ok = whisper_sched_graph_init(state->sched_encode, state->backends,
                [&]() {
                    return whisper_build_graph_encoder(*ctx, *state);
                });

        if (!ok) {
            WHISPER_LOG_ERROR("%s: failed to init encoder allocator\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }

        WHISPER_LOG_INFO("%s: compute buffer (encode) = %7.2f MB\n", __func__, whisper_sched_size(state->sched_encode) / 1e6);
    }

    // cross allocator
    {
        bool ok = whisper_sched_graph_init(state->sched_cross, state->backends,
                [&]() {
                    return whisper_build_graph_cross(*ctx, *state);
                });

        if (!ok) {
            WHISPER_LOG_ERROR("%s: failed to init cross allocator\n", __func__);
            whisper_free_state(state);
            return nullptr;
        }

        WHISPER_LOG_INFO("%s: compute buffer (cross)  = %7.2f MB\n", __func__, whisper_sched_size(state->sched_cross) / 1e6);
    }

    return state;
}

int whisper_ctx_init_openvino_encoder_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
                    const char * model_path,
                    const char * device,
                    const char * cache_dir) {
#ifndef WHISPER_USE_OPENVINO
    (void)(ctx);
    (void)(state);
    (void)(model_path);
    (void)(device);
    (void)(cache_dir);

    return 1;
#else
    if (!model_path && ctx->path_model.empty()) {
        WHISPER_LOG_ERROR("%s: model_path is nullptr, and ctx has no model_path set.\n", __func__);
        return 1;
    }

    std::string path_encoder;
    if (!model_path) {
        //if model_path is not set, attempt to find it in the same directory as ggml-<model>.bin model
        path_encoder = whisper_openvino_get_path_encoder(ctx->path_model);
    } else {
        path_encoder = model_path;
    }

    std::string path_cache;
    if (!cache_dir) {
        //if cache_dir is not set, set it as a dir residing next to ggml-<model>.bin
        path_cache = whisper_openvino_get_path_cache(ctx->path_model);
    } else {
        path_cache = cache_dir;
    }

    WHISPER_LOG_INFO("%s: loading OpenVINO model from '%s'\n", __func__, path_encoder.c_str());
    WHISPER_LOG_INFO("%s: first run on a device may take a while ...\n", __func__);

    state->ctx_openvino = whisper_openvino_init(path_encoder.c_str(), device, path_cache.c_str());
    if (!state->ctx_openvino) {
        WHISPER_LOG_ERROR("%s: failed to init OpenVINO encoder from '%s'\n", __func__, path_encoder.c_str());
        return 1;
    } else {
        WHISPER_LOG_INFO("%s: OpenVINO model loaded\n", __func__);
    }

    return 0;
#endif
}

int whisper_ctx_init_openvino_encoder(
        struct whisper_context * ctx,
                    const char * model_path,
                    const char * device,
                    const char * cache_dir) {
    return whisper_ctx_init_openvino_encoder_with_state(ctx, ctx->state, model_path, device, cache_dir);
}

struct whisper_context_params whisper_context_default_params() {
    struct whisper_context_params result = {
        /*.use_gpu              =*/ true,
        /*.flash_attn           =*/ false,
        /*.gpu_device           =*/ 0,

        /*.dtw_token_timestamps =*/ false,
        /*.dtw_aheads_preset    =*/ WHISPER_AHEADS_NONE,
        /*.dtw_n_top            =*/ -1,
        /*.dtw_aheads           =*/ {
            /*.n_heads          =*/ 0,
            /*.heads            =*/ NULL,
        },
        /*.dtw_mem_size         =*/ 1024*1024*128,
    };
    return result;
}

struct whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model, struct whisper_context_params params) {
    WHISPER_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);
#ifdef _MSC_VER
    // Convert UTF-8 path to wide string (UTF-16) for Windows, resolving character encoding issues.
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring path_model_wide = converter.from_bytes(path_model);
    auto fin = std::ifstream(path_model_wide, std::ios::binary);
#else
    auto fin = std::ifstream(path_model, std::ios::binary);
#endif
    if (!fin) {
        WHISPER_LOG_ERROR("%s: failed to open '%s'\n", __func__, path_model);
        return nullptr;
    }

    whisper_model_loader loader = {};

    loader.context = &fin;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    loader.eof = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        return fin->eof();
    };

    loader.close = [](void * ctx) {
        std::ifstream * fin = (std::ifstream*)ctx;
        fin->close();
    };

    auto ctx = whisper_init_with_params_no_state(&loader, params);

    if (ctx) {
        ctx->path_model = path_model;
    }

    return ctx;
}

struct whisper_context * whisper_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size, struct whisper_context_params params) {
    struct buf_context {
        uint8_t* buffer;
        size_t size;
        size_t current_offset;
    };

    buf_context ctx = { reinterpret_cast<uint8_t*>(buffer), buffer_size, 0 };

    WHISPER_LOG_INFO("%s: loading model from buffer\n", __func__);

    whisper_model_loader loader = {};

    loader.context = &ctx;

    loader.read = [](void * ctx, void * output, size_t read_size) {
        buf_context * buf = reinterpret_cast<buf_context *>(ctx);

        size_t size_to_copy = buf->current_offset + read_size < buf->size ? read_size : buf->size - buf->current_offset;

        memcpy(output, buf->buffer + buf->current_offset, size_to_copy);
        buf->current_offset += size_to_copy;

        return size_to_copy;
    };

    loader.eof = [](void * ctx) {
        buf_context * buf = reinterpret_cast<buf_context *>(ctx);

        return buf->current_offset >= buf->size;
    };

    loader.close = [](void * /*ctx*/) { };

    return whisper_init_with_params_no_state(&loader, params);
}

struct whisper_context * whisper_init_with_params_no_state(struct whisper_model_loader * loader, struct whisper_context_params params) {
    ggml_time_init();

    if (params.flash_attn && params.dtw_token_timestamps) {
        WHISPER_LOG_WARN("%s: dtw_token_timestamps is not supported with flash_attn - disabling\n", __func__);
        params.dtw_token_timestamps = false;
    }

    WHISPER_LOG_INFO("%s: use gpu    = %d\n", __func__, params.use_gpu);
    WHISPER_LOG_INFO("%s: flash attn = %d\n", __func__, params.flash_attn);
    WHISPER_LOG_INFO("%s: gpu_device = %d\n", __func__, params.gpu_device);
    WHISPER_LOG_INFO("%s: dtw        = %d\n", __func__, params.dtw_token_timestamps);
    WHISPER_LOG_INFO("%s: devices    = %zu\n", __func__, ggml_backend_dev_count());
    WHISPER_LOG_INFO("%s: backends   = %zu\n", __func__, ggml_backend_reg_count());

    whisper_context * ctx = new whisper_context;
    ctx->params = params;

    if (!whisper_model_load(loader, *ctx)) {
        loader->close(loader->context);
        WHISPER_LOG_ERROR("%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }

    loader->close(loader->context);

    return ctx;
}

struct whisper_context * whisper_init_from_file_with_params(const char * path_model, struct whisper_context_params params) {
    whisper_context * ctx = whisper_init_from_file_with_params_no_state(path_model, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct whisper_context * whisper_init_from_buffer_with_params(void * buffer, size_t buffer_size, struct whisper_context_params params) {
    whisper_context * ctx = whisper_init_from_buffer_with_params_no_state(buffer, buffer_size, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct whisper_context * whisper_init_with_params(struct whisper_model_loader * loader, struct whisper_context_params params) {
    whisper_context * ctx = whisper_init_with_params_no_state(loader, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}

struct whisper_context * whisper_init_from_file(const char * path_model) {
    return whisper_init_from_file_with_params(path_model, whisper_context_default_params());
}

struct whisper_context * whisper_init_from_buffer(void * buffer, size_t buffer_size) {
    return whisper_init_from_buffer_with_params(buffer, buffer_size, whisper_context_default_params());
}

struct whisper_context * whisper_init(struct whisper_model_loader * loader) {
    return whisper_init_with_params(loader, whisper_context_default_params());
}

struct whisper_context * whisper_init_from_file_no_state(const char * path_model) {
    return whisper_init_from_file_with_params_no_state(path_model, whisper_context_default_params());
}

struct whisper_context * whisper_init_from_buffer_no_state(void * buffer, size_t buffer_size) {
    return whisper_init_from_buffer_with_params_no_state(buffer, buffer_size, whisper_context_default_params());
}

struct whisper_context * whisper_init_no_state(struct whisper_model_loader * loader) {
    return whisper_init_with_params_no_state(loader, whisper_context_default_params());
}

void whisper_free_state(struct whisper_state * state) {
    if (state) {
        whisper_kv_cache_free(state->kv_cross);
        whisper_kv_cache_free(state->kv_pad);

#ifdef WHISPER_USE_COREML
        if (state->ctx_coreml != nullptr) {
            whisper_coreml_free(state->ctx_coreml);
            state->ctx_coreml = nullptr;
        }
#endif

#ifdef WHISPER_USE_OPENVINO
        if (state->ctx_openvino != nullptr) {
            whisper_openvino_free(state->ctx_openvino);
            state->ctx_openvino = nullptr;
        }
#endif

        whisper_batch_free(state->batch);

        ggml_backend_sched_free(state->sched_mel.sched);
        ggml_backend_sched_free(state->sched_conv.sched);
        ggml_backend_sched_free(state->sched_encode.sched);
        ggml_backend_sched_free(state->sched_cross.sched);
        ggml_backend_buffer_free(state->mel_basis_buffer);

        for (auto & backend : state->backends) {
            ggml_backend_free(backend);
        }

        // [EXPERIMENTAL] Token-level timestamps with DTW
        aheads_masks_free(state->aheads_masks);

        delete state;
    }
}

void whisper_free(struct whisper_context * ctx) {
    if (ctx) {
        ggml_free(ctx->model.ctx);

        ggml_backend_buffer_free(ctx->model.buffer);

        whisper_free_state(ctx->state);

        delete ctx;
    }
}

int whisper_pcm_to_mel_with_state(struct whisper_context * ctx, struct whisper_state * state, const float * samples, int n_samples, int n_threads) {
    if (!log_mel_spectrogram(*ctx, *state, samples, n_samples, n_threads)) {
        WHISPER_LOG_ERROR("%s: failed to compute mel spectrogram\n", __func__);
        return -1;
    }

    return 0;
}

int whisper_pcm_to_mel(struct whisper_context * ctx, const float * samples, int n_samples, int n_threads) {
    return whisper_pcm_to_mel_with_state(ctx, ctx->state, samples, n_samples, n_threads);
}

int whisper_encode_with_state(struct whisper_context * ctx, struct whisper_state * state, int offset, int n_threads) {
    if (!whisper_encode_internal(*ctx, *state, offset, n_threads, nullptr, nullptr)) {
        WHISPER_LOG_ERROR("%s: failed to eval\n", __func__);
        return -1;
    }

    return 0;
}

int whisper_encode(struct whisper_context * ctx, int offset, int n_threads) {
    if (!whisper_encode_internal(*ctx, *ctx->state, offset, n_threads, nullptr, nullptr)) {
        WHISPER_LOG_ERROR("%s: failed to eval\n", __func__);
        return -1;
    }

    return 0;
}

int whisper_n_len_from_state(struct whisper_state * state) {
    return state->mel.n_len_org;
}

int whisper_n_len(struct whisper_context * ctx) {
    return ctx->state->mel.n_len_org;
}

////////////////////////////////////////////////////////////////////////////

void whisper_log_set(ggml_log_callback log_callback, void * user_data) {
    g_state.log_callback = log_callback ? log_callback : whisper_log_callback_default;
    g_state.log_callback_user_data = user_data;
    ggml_log_set(g_state.log_callback, g_state.log_callback_user_data);
}

GGML_ATTRIBUTE_FORMAT(2, 3)
static void whisper_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[1024];
    int len = vsnprintf(buffer, 1024, format, args);
    if (len < 1024) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args);
}

static void whisper_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
#ifndef WHISPER_DEBUG
    if (level == GGML_LOG_LEVEL_DEBUG) {
        return;
    }
#endif
    fputs(text, stderr);
    fflush(stderr);
}

// Expose encoder hidden states for downstream use (e.g. wake word detection).
int whisper_encoder_output(struct whisper_context * ctx, float * output, int max_floats) {
    if (!ctx || !ctx->state || !ctx->state->embd_enc) {
        return -1;
    }
    struct ggml_tensor * t = ctx->state->embd_enc;
    int n = (int)ggml_nelements(t);
    if (n > max_floats) {
        n = max_floats;
    }
    ggml_backend_tensor_get(t, output, 0, n * sizeof(float));
    return n;
}

// Set encoder context length for variable-length encoding.
void whisper_set_audio_ctx(struct whisper_context * ctx, int n_audio_ctx) {
    if (ctx && ctx->state) {
        ctx->state->exp_n_audio_ctx = n_audio_ctx;
    }
}

void whisper_set_encoder_only(struct whisper_context * ctx, bool encoder_only) {
    if (ctx && ctx->state) {
        ctx->state->skip_cross = encoder_only;
    }
}
