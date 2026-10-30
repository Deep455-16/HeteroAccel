// examples/c_example.c
// Phase 23: Minimal C SDK Consumer Example

#define _CRT_SECURE_NO_WARNINGS
#include <heteroaccel/heteroaccel.h>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== HeteroAccel C SDK Example (v%s) ===\n", ha_version_string());

    // 1. Initialize Runtime
    HaRuntime rt = ha_runtime_create();
    if (!rt) {
        fprintf(stderr, "Failed to create HeteroAccel runtime.\n");
        return 1;
    }

    // 2. Query available backends
    int count = ha_backend_count(rt);
    printf("Registered backends: %d\n", count);
    for (int i = 0; i < count; ++i) {
        HaBackendInfo info;
        if (ha_backend_get(rt, i, &info) == HA_OK) {
            printf("  [%d] %-18s (Available: %s, Vulkan: %s, CUDA: %s)\n",
                   i, info.backend_id,
                   info.available ? "YES" : "NO",
                   info.requires_vulkan ? "YES" : "NO",
                   info.requires_cuda ? "YES" : "NO");
        }
    }

    // 3. Load model if path is provided via env
    const char* model_path = getenv("HETEROACCEL_MODEL_PATH");
    if (!model_path) {
        printf("HETEROACCEL_MODEL_PATH not set. Skipping model inference.\n");
        ha_runtime_destroy(rt);
        return 0;
    }

    printf("Loading model: %s\n", model_path);
    HaError err = HA_OK;
    HaModel model = ha_model_load(rt, model_path, &err);
    if (!model || err != HA_OK) {
        fprintf(stderr, "Failed to load model: %s\n", ha_error_string(err));
        ha_runtime_destroy(rt);
        return 1;
    }

    printf("Model loaded:\n");
    printf("  Architecture:   %s\n", ha_model_arch(model));
    printf("  Context Length: %u\n", ha_model_context_length(model));
    printf("  Vocab Size:     %u\n", ha_model_vocab_size(model));

    // 4. Create inference session
    HaSession session = ha_session_create(model, &err);
    if (!session || err != HA_OK) {
        fprintf(stderr, "Failed to create session: %s\n", ha_error_string(err));
        ha_model_unload(model);
        ha_runtime_destroy(rt);
        return 1;
    }

    // 5. Run inference
    HaInferParams params;
    params.prompt = "The capital of France is";
    params.token_ids = NULL;
    params.n_token_ids = 0;
    params.max_new_tokens = 8;
    params.temperature = 0.0f;
    params.top_p = 0.95f;

    printf("Running inference...\n");
    HaResult res = ha_infer(session, &params);
    if (res.error != HA_OK) {
        fprintf(stderr, "Inference error: %s (%s)\n",
                ha_error_string(res.error),
                res.error_message ? res.error_message : "");
    } else {
        printf("Output: '%s'\n", res.output_text);
        printf("Tokens generated: %d (%d prompt tokens)\n",
               res.n_output_tokens, res.n_prompt_tokens);
        printf("Throughput:       %.2f tok/s\n", res.tokens_per_second);
        printf("Backend:          %s (GPU: %s)\n",
               res.backend_used, res.gpu_used ? "YES" : "NO");
    }

    // 6. Cleanup
    ha_session_destroy(session);
    ha_model_unload(model);
    ha_runtime_destroy(rt);
    printf("Session, model, and runtime cleaned up cleanly.\n");

    return 0;
}
