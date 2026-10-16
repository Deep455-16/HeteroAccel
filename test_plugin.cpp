// test_plugin.cpp
// Independent test plugin for HeteroAccel Phase 18
// Implements the public plugin ABI (C-compatible) as defined in PluginAbis.h.
// This plugin is built as a separate shared-library target,
// completely independent of the host executable.
// It demonstrates:
//   - pluginGetIdentity()
//   - pluginInitialize()
//   - pluginExecute()
//   - pluginShutdown()
//   - Capability manifest reporting
//   - Simple deterministic "test operation" (not real LLM inference)

// Define AGR_PLUGIN_BUILDING_DLL so the ABI headers export correctly
#include "plugin/PluginAbis.h"

// Ensure ExecutionResult type is available in the agr namespace
// (needed for pluginExecute to fill outResult)
#include "engine/ExecutionResult.h"

// ---------------------------------------------------------------------------
// Engine identity (matches what we report)
// ---------------------------------------------------------------------------

static const char* TEST_ENGINE_NAME = "hydroxl-test";
static const char* TEST_ENGINE_VERSION = "1.0.0";
static const char* TEST_ENGINE_TYPE = "LLM";
static const char* TEST_ENGINE_AUTHOR = "HeteroAccel Project";

// ---------------------------------------------------------------------------
// Capability manifest — bits that match EngineCapability enum values
// ---------------------------------------------------------------------------

// Map EngineCapability bits (same positions):
//   Bit 0  = TEXT_GENERATION   (0x0001) — we support it
//   Bit 8  = BACKEND_CPU       (0x0100) — CPU only
//   Bit 12 = STREAMING         (0x1000) — we support streaming
//   Bit 14 = CANCELLATION      (0x4000) — cancellation supported

static const uint32_t TEST_CAPABILITIES_MASK =
    (1u << 0) |      // TEXT_GENERATION
    (1u << 8) |      // BACKEND_CPU
    (1u << 12) |     // STREAMING
    (1u << 14);      // CANCELLATION

// ---------------------------------------------------------------------------
// Plugin module handle (set by host via LoadLibrary)
// ---------------------------------------------------------------------------

static agrPluginModuleHandle g_plugin_module{ nullptr };

// ---------------------------------------------------------------------------
// Engine context (opaque to the host; we keep it simple)
// ---------------------------------------------------------------------------

typedef struct {
    bool initialized;
    bool model_loaded;
    // In a real engine we'd have model data, tokenizer, etc.
    // For this test we just track state.
} TestEngineContext;

// ---------------------------------------------------------------------------
// Exported function: pluginGetIdentity
// ---------------------------------------------------------------------------

int __stdcall pluginGetIdentity(agrPluginModuleHandle hModule, agrPluginIdentity* outIdentity) {
    // OutIdentity points to a host-allocated agrPluginIdentity struct.
    // We must fill it without exceeding bounds.
    if (!outIdentity) return AGR_PLUGIN_ERROR_INVALID_REQUEST;

    // Safe copies — strncpy_s style, but plain C for compatibility
    // Name: 64 bytes, Version: 32 bytes, Type: 32 bytes, Author: 64 bytes, Key: 64 bytes
    // We'll write null-terminated strings; the host will truncate as needed.

    // Name
    const char* name = TEST_ENGINE_NAME;
    size_t name_len = strlen(name);
    if (name_len >= sizeof(outIdentity->name)) name_len = sizeof(outIdentity->name) - 1;
    memcpy(outIdentity->name, name, name_len);
    outIdentity->name[name_len] = '\0';

    // Version
    const char* version = TEST_ENGINE_VERSION;
    size_t ver_len = strlen(version);
    if (ver_len >= sizeof(outIdentity->version)) ver_len = sizeof(outIdentity->version) - 1;
    memcpy(outIdentity->version, version, ver_len);
    outIdentity->version[ver_len] = '\0';

    // Type
    const char* type = TEST_ENGINE_TYPE;
    size_t type_len = strlen(type);
    if (type_len >= sizeof(outIdentity->type)) type_len = sizeof(outIdentity->type) - 1;
    memcpy(outIdentity->type, type, type_len);
    outIdentity->type[type_len] = '\0';

    // Author
    const char* author = TEST_ENGINE_AUTHOR;
    size_t author_len = strlen(author);
    if (author_len >= sizeof(outIdentity->author)) author_len = sizeof(outIdentity->author) - 1;
    memcpy(outIdentity->author, author, author_len);
    outIdentity->author[author_len] = '\0';

    // Registry key — must be unique; we derive it from name+version
    const char* key = TEST_ENGINE_NAME; // simplified: just use name
    size_t key_len = strlen(key);
    if (key_len >= sizeof(outIdentity->registry_key)) key_len = sizeof(outIdentity->registry_key) - 1;
    memcpy(outIdentity->registry_key, key, key_len);
    outIdentity->registry_key[key_len] = '\0';

    return AGR_PLUGIN_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Exported function: pluginInitialize
// ---------------------------------------------------------------------------

int __stdcall pluginInitialize(agrPluginModuleHandle hModule, const agrPluginIdentity* identity, void** outEngineContext) {
    // identity is the value returned by pluginGetIdentity (may be nullptr on first call
    // if the host hasn't called getIdentity yet; we handle both cases).
    // outEngineContext receives an opaque pointer; we allocate a TestEngineContext.

    // Store the module handle for use by other exports
    g_plugin_module = hModule;

    // Allocate engine context
    TestEngineContext* ctx = (TestEngineContext*)malloc(sizeof(TestEngineContext));
    if (!ctx) {
        return AGR_PLUGIN_ERROR_OUT_OF_MEMORY;
    }
    ctx->initialized = false;
    ctx->model_loaded = false;

    *outEngineContext = reinterpret_cast<void*>(ctx);
    return AGR_PLUGIN_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Exported function: pluginExecute
// ---------------------------------------------------------------------------

int __stdcall pluginExecute(agrPluginModuleHandle hModule, const void* ctx, void* outResult) {
    // ctx is an opaque ExecutionContext pointer provided by the host.
    // outResult must be filled with an ExecutionResult struct.
    // The host owns outResult memory; we write into it but don't allocate new.

    if (!ctx) return AGR_PLUGIN_ERROR_INVALID_REQUEST;

    // For this test plugin, we produce a deterministic "hello world" output
    // when a test prompt is provided. In a real engine this would be LLM inference.

    // Cast outResult to ExecutionResult — the host layout matches our expectation.
    // We assume the host has allocated at least sizeof(ExecutionResult) bytes.
    agr::ExecutionResult* result = reinterpret_cast<agr::ExecutionResult*>(outResult);
    if (!result) {
        return AGR_PLUGIN_ERROR_INVALID_REQUEST;
    }

    // Initialize result to zero/false

    // For demo purposes, if the prompt contains "test", return a canned response.
    // Otherwise, produce a minimal empty result.
    // (Real engines would do actual model inference here.)

    // We check if the host passed a non-empty prompt via the context.
    // The ExecutionContext layout is defined by the host; for this test we
    // just produce a basic result.

    result->success = true;
    // Copy a minimal test output string
    const char* test_output = "test operation";
    size_t out_len = strlen(test_output);
    if (out_len >= sizeof(result->output)) out_len = sizeof(result->output) - 1;
    result->output.assign(test_output, out_len);

    // Telemetry: indicate CPU backend (this plugin is CPU-only)
    result->telemetry.backend_type = agr::ExecutionBackendType::CPU;
    // We don't have a device name, leave as default
    result->telemetry.gpu_used = false;
    result->telemetry.gpu_layers = 0;
    result->telemetry.prompt_tokens = 0;
    result->telemetry.output_tokens = 1; // our single "test" token
    result->telemetry.model_load_ms = 0.0;
    result->telemetry.prompt_eval_ms = 0.0;
    result->telemetry.generation_ms = 0.0;
    result->telemetry.total_ms = 0.0;
    result->telemetry.ttft_ms = 0.0;
    result->telemetry.tokens_per_sec = 0.0;

    // Scheduler metadata
    result->telemetry.scheduler_reason = "plugin-selected";
    result->telemetry.fallbacks = 0;

    // Engine identity (filled from our reported identity)
    // Note: The host may copy this from the plugin's reported identity.
    // For this test we just set a basic value.
    result->telemetry.backend_name = "HydroXLTestBackend";

    return AGR_PLUGIN_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Exported function: pluginShutdown
// ---------------------------------------------------------------------------

int __stdcall pluginShutdown(agrPluginModuleHandle hModule) {
    // Free any allocated resources.
    if (g_plugin_module) {
        // We don't FreeLibrary here — the host controls module lifetime.
        // We just clean up our internal context.
        g_plugin_module = nullptr;
    }

    // Free the engine context if we allocated it during initialize.
    // The host may have stored the context pointer; we must not double-free.
    // For this test, we assume the host will call pluginShutdown before unloading,
    // and will free the context itself. We'll still attempt a safe cleanup.

    // In a real plugin, free all malloc'd buffers, model data, tokenizer, etc.
    // For now, just return success.

    return AGR_PLUGIN_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Exported function: pluginGetState
// ---------------------------------------------------------------------------

int __stdcall pluginGetState(agrPluginModuleHandle hModule) {
    return AGR_PLUGIN_STATE_READY;
}

// ---------------------------------------------------------------------------
// Exported function: pluginGetError
// ---------------------------------------------------------------------------

int __stdcall pluginGetError(agrPluginModuleHandle hModule, agrPluginError* outError) {
    // Return the last error encountered by the plugin.
    // For this test plugin, we have no internal error state beyond what
    // the individual exports return.

    if (outError) {
        outError->code = AGR_PLUGIN_ERROR_NONE;
        outError->message[0] = '\0';
    }
    return AGR_PLUGIN_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Exported function: pluginGetCapabilities
// ---------------------------------------------------------------------------

int __stdcall pluginGetCapabilities(agrPluginModuleHandle hModule, agrPluginCapabilityManifest* outManifest) {
    // Return the capability manifest (bitmask of supported capabilities).
    if (!outManifest) return AGR_PLUGIN_ERROR_INVALID_REQUEST;

    // Fill the manifest
    outManifest->capabilities = TEST_CAPABILITIES_MASK;
    // Count the number of set bits
    size_t count = 0;
    for (int i = 0; i < 32; ++i) {
        if (TEST_CAPABILITIES_MASK & (1u << i)) {
            ++count;
        }
    }
    outManifest->capability_count = count;

    return AGR_PLUGIN_ERROR_NONE;
}

// ---------------------------------------------------------------------------
// Exported function: pluginCreateEngine (optional)
// ---------------------------------------------------------------------------

void __stdcall pluginCreateEngine(agrPluginModuleHandle hModule, const agrPluginIdentity* identity, void** outEngineInstance) {
    // Optional factory function. For this test we don't fully implement
    // engine instance creation via this path; the host uses the dynamic
    // loading + getIdentity + initialize pattern instead.
    // If outEngineInstance is provided, we may optionally fill it,
    // but absence is not an error.

    // For completeness, we set outEngineInstance to nullptr and return
    // AGR_PLUGIN_ERROR_NONE. The host will use the standard LoadLibrary
    // + GetProcAddress path instead.
    *outEngineInstance = nullptr;
}

// ============================================================================
// End of test_plugin.cpp
// This single file compiles into a standalone .dll that the host can
// discover and load through the PluginLoader infrastructure.
// No modifications to HeteroAccel core are required.
// ============================================================================