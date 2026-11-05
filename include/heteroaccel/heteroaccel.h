// include/heteroaccel/heteroaccel.h
// HeteroAccel Phase 24: Native Streaming and Intelligent Runtime Resource Management
//
// Standalone C API header providing full access to HeteroAccel runtime,
// model loading, hardware-aware backend orchestration, memory estimation,
// persistent performance profiling, native token streaming, and inference.

#ifndef HETEROACCEL_H
#define HETEROACCEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Windows DLL symbol export/import macros
#ifdef _WIN32
  #ifdef HETEROACCEL_BUILDING_DLL
    #define HA_API __declspec(dllexport)
  #elif defined(HETEROACCEL_STATIC)
    #define HA_API
  #else
    #define HA_API __declspec(dllimport)
  #endif
#else
  #define HA_API __attribute__((visibility("default")))
#endif

// Versioning
#define HETEROACCEL_VERSION_MAJOR 0
#define HETEROACCEL_VERSION_MINOR 25
#define HETEROACCEL_VERSION_PATCH 0

// Opaque handles
typedef struct HaRuntime_  *HaRuntime;
typedef struct HaModel_    *HaModel;
typedef struct HaSession_  *HaSession;

// Error codes
typedef enum {
    HA_OK              = 0,
    HA_ERR_INVALID_ARG = 1,
    HA_ERR_MODEL_LOAD  = 2,
    HA_ERR_NO_BACKEND  = 3,
    HA_ERR_INFERENCE   = 4,
    HA_ERR_CANCELLED   = 5,
    HA_ERR_OOM         = 6,
    HA_ERR_UNSUPPORTED = 7,
    HA_ERR_INTERNAL    = 8,
} HaError;

// Runtime lifecycle
HA_API HaRuntime   ha_runtime_create(void);
HA_API void        ha_runtime_destroy(HaRuntime rt);
HA_API const char* ha_version_string(void);

// Persistent performance profiling
HA_API HaError     ha_runtime_perf_reset(HaRuntime rt);
HA_API HaError     ha_runtime_perf_path(HaRuntime rt, char* buf, size_t buf_size);

// Model loading
HA_API HaModel     ha_model_load(HaRuntime rt, const char* path, HaError* err_out);
HA_API void        ha_model_unload(HaModel model);
HA_API const char* ha_model_arch(HaModel model);
HA_API uint32_t    ha_model_context_length(HaModel model);
HA_API uint32_t    ha_model_vocab_size(HaModel model);

// Context-aware memory estimation
typedef struct {
    size_t      weights_bytes;       // dequantized float32 model weights
    size_t      kv_cache_bytes;      // KV cache for requested max_seq_len
    size_t      activations_bytes;   // per-token scratch buffers
    size_t      overhead_bytes;      // safety overhead
    size_t      total_estimated;     // sum of above
    int         unified_memory;      // 1 if GPU and CPU share system RAM
    const char* notes;               // human-readable explanation
} HaMemoryEstimate;

HA_API HaError     ha_model_memory_estimate(HaModel model, uint32_t max_seq_len,
                                            HaMemoryEstimate* out);

// Inference session
HA_API HaSession   ha_session_create(HaModel model, HaError* err_out);
HA_API void        ha_session_destroy(HaSession session);

// Inference request params
typedef struct {
    const char* prompt;          // text prompt (may be NULL if token_ids provided)
    const int*  token_ids;       // pre-tokenized ids (may be NULL)
    int         n_token_ids;     // count of token_ids
    int         max_new_tokens;  // maximum number of generated tokens
    float       temperature;     // sampling temperature (0.0 = greedy)
    float       top_p;           // nucleus sampling threshold
} HaInferParams;

// Inference result
typedef struct {
    char*       output_text;       // caller must NOT free (valid until next call or session destroy)
    const int*  output_token_ids;  // valid until next call or session destroy
    int         n_output_tokens;
    int         n_prompt_tokens;
    float       tokens_per_second;
    int         gpu_used;          // 1 if GPU ran, 0 if CPU
    const char* backend_used;
    HaError     error;
    const char* error_message;
} HaResult;

// Synchronous inference
HA_API HaResult    ha_infer(HaSession session, const HaInferParams* params);

// Streaming token callback (return 0 to continue, non-zero to cancel)
typedef int (*HaTokenCallback)(const char* token_text, int token_id, void* user_data);

// Streaming inference: native incremental token streaming (or post-hoc fallback)
HA_API HaError     ha_infer_stream(HaSession session, const HaInferParams* params,
                                   HaTokenCallback cb, void* user_data);

// Cancellation
HA_API HaError     ha_session_cancel(HaSession session);
HA_API HaError     ha_infer_cancel(HaSession session);

// Backend introspection
typedef struct {
    const char* backend_id;
    int         available;
    int         requires_vulkan;
    int         requires_cuda;
} HaBackendInfo;

HA_API int         ha_backend_count(HaRuntime rt);
HA_API HaError     ha_backend_get(HaRuntime rt, int index, HaBackendInfo* info_out);
HA_API const char* ha_error_string(HaError err);

#ifdef __cplusplus
}
#endif

#endif // HETEROACCEL_H
