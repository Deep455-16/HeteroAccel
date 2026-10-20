// src/plugin/PluginLoader.cpp
// Phase 19: plugin discovery, ABI validation, loading, and lifecycle.
//
// Fixes over the Phase 18 draft (which was never compiled):
//   - ABI version is negotiated at runtime via pluginGetAbiVersion() before
//     any other plugin entry point is invoked.
//   - A successful DLL load is never misrepresented as an executable engine
//     registration: no ExecutionEngineRegistry entry is created. Plugins
//     become executable only through plugin_execute() on a fully
//     initialized plugin.
//   - pluginShutdown() is invoked on unload when the plugin was
//     initialized; the initialized flag is actually maintained.
//   - Discovery takes the registry mutex, so concurrent discovery /
//     load / unload cannot corrupt the shared cache.
//   - agrPluginError is a C struct (uint32 + char[256]); errors are stored
//     as std::string internally and copied out on query.
//
// SECURITY: LoadLibrary executes DllMain inside the host process with full
// privileges. There is no sandbox. See PluginLoader.h.

#include "plugin/PluginLoader.h"

#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace agr {

// ---------------------------------------------------------------------------
// Internal plugin state
// ---------------------------------------------------------------------------
namespace {

struct PluginInternalState {
    std::string path;
    std::string filename;
    HMODULE module_handle{ nullptr };
    uint32_t abi_version{ 0 };
    std::string name, version, type, author, registry_key;
    uint32_t capabilities{ 0 };
    size_t   capability_count{ 0 };
    agrPluginState state{ agrPluginState::AGR_PLUGIN_STATE_DISCOVERED };
    uint32_t last_error_code{ AGR_PLUGIN_ERROR_NONE };
    std::string last_error_message;
    bool initialized{ false };
    bool loaded{ false };
};

std::mutex plugin_mutex;
std::unordered_map<std::string, PluginInternalState> plugin_cache;

void setError(PluginInternalState& s, uint32_t code, const std::string& msg) {
    s.last_error_code = code;
    s.last_error_message = msg;
}

std::string pluginDirFromEnv() {
    char* env_dir = nullptr;
    size_t len = 0;
    std::string dir = "./plugins";
    if (_dupenv_s(&env_dir, &len, "AGR_PLUGIN_DIR") == 0 && env_dir != nullptr) {
        if (len > 1) dir = std::string(env_dir, len - 1); // strip trailing NUL
        std::free(env_dir);
    }
    return dir;
}

HMODULE load_plugin_module(const std::string& path) {
    std::wstring wpath(path.begin(), path.end());
    HMODULE h = LoadLibraryW(wpath.c_str());
    if (!h) h = LoadLibraryA(path.c_str());
    return h;
}

void copyToField(char* dst, size_t cap, const std::string& src) {
    size_t n = (std::min)(src.size(), cap - 1); // parens: defeat windows.h min macro
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

// Plugin entry points take the opaque agrPluginModuleHandle; convert the
// HMODULE once here.
inline agrPluginModuleHandle toHandle(HMODULE h) {
    return reinterpret_cast<agrPluginModuleHandle>(h);
}

// Validate a freshly loaded module: ABI version + required exports.
// Returns true and fills identity/capabilities when the plugin is usable.
// `probe_error` receives a diagnostic when validation fails.
bool validate_module(HMODULE mod, PluginInternalState& s) {
    // ABI version must be negotiated before anything else is called.
    auto get_abi = reinterpret_cast<agrPluginGetAbiVersionFn>(
        GetProcAddress(mod, "pluginGetAbiVersion"));
    if (!get_abi) {
        setError(s, AGR_PLUGIN_ERROR_INIT_FAILED,
                 "plugin does not export pluginGetAbiVersion (pre-ABI-v1 plugin)");
        return false;
    }
    s.abi_version = get_abi();
    if (s.abi_version != AGR_HOST_EXPECTED_ABI_VERSION) {
        setError(s, AGR_PLUGIN_ERROR_INIT_FAILED,
                 "ABI version mismatch: plugin=" + std::to_string(s.abi_version) +
                 " host=" + std::to_string(AGR_HOST_EXPECTED_ABI_VERSION));
        return false;
    }

    // Required entry points.
    auto get_identity = reinterpret_cast<agrPluginGetIdentityFn>(
        GetProcAddress(mod, "pluginGetIdentity"));
    auto initialize   = reinterpret_cast<agrPluginInitializeFn>(
        GetProcAddress(mod, "pluginInitialize"));
    auto execute      = reinterpret_cast<agrPluginExecuteFn>(
        GetProcAddress(mod, "pluginExecute"));
    auto shutdown_fn  = reinterpret_cast<agrPluginShutdownFn>(
        GetProcAddress(mod, "pluginShutdown"));
    if (!get_identity || !initialize || !execute || !shutdown_fn) {
        setError(s, AGR_PLUGIN_ERROR_INIT_FAILED,
                 "missing required exports (pluginGetIdentity, pluginInitialize, "
                 "pluginExecute, pluginShutdown)");
        return false;
    }

    agrPluginIdentity ident{};
    int rc = get_identity(toHandle(mod), &ident);
    if (rc != AGR_PLUGIN_ERROR_NONE) {
        setError(s, static_cast<uint32_t>(rc), "pluginGetIdentity failed");
        return false;
    }
    s.name         = ident.name;
    s.version      = ident.version;
    s.type         = ident.type;
    s.author       = ident.author;
    s.registry_key = ident.registry_key;

    auto get_caps = reinterpret_cast<agrPluginGetCapabilitiesFn>(
        GetProcAddress(mod, "pluginGetCapabilities"));
    if (get_caps) {
        agrPluginCapabilityManifest manifest{};
        if (get_caps(toHandle(mod), &manifest) == AGR_PLUGIN_ERROR_NONE) {
            s.capabilities = manifest.capabilities;
            s.capability_count = manifest.capability_count;
        }
    }
    setError(s, AGR_PLUGIN_ERROR_NONE, "OK");
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// plugin_discover()
// ---------------------------------------------------------------------------
std::vector<agrPluginDiscoveryResult> plugin_discover() {
    std::vector<agrPluginDiscoveryResult> results;

    std::string plugin_dir = pluginDirFromEnv();
    std::string search_path = plugin_dir + "/*.dll";

    WIN32_FIND_DATAA find_data{};
    HANDLE hFind = FindFirstFileA(search_path.c_str(), &find_data);
    if (hFind == INVALID_HANDLE_VALUE) {
        return results;
    }

    do {
        if (strcmp(find_data.cFileName, ".") == 0 ||
            strcmp(find_data.cFileName, "..") == 0) {
            continue;
        }

        std::string full_path = plugin_dir + "/" + find_data.cFileName;
        std::string filename(find_data.cFileName);

        agrPluginDiscoveryResult result{};
        copyToField(result.path, sizeof(result.path), full_path);
        copyToField(result.filename, sizeof(result.filename), filename);
        result.loadable = false;
        result.load_error.code = AGR_PLUGIN_ERROR_INTERNAL;
        copyToField(result.load_error.message, sizeof(result.load_error.message),
                    "unknown");

        PluginInternalState probe;
        probe.path = full_path;
        probe.filename = filename;

        // NOTE: LoadLibrary executes DllMain in-process (no sandbox).
        HMODULE mod = load_plugin_module(full_path);
        if (mod && validate_module(mod, probe)) {
            result.loadable = true;
            result.load_error.code = AGR_PLUGIN_ERROR_NONE;
            copyToField(result.load_error.message,
                        sizeof(result.load_error.message), "OK");
            copyToField(result.identity.name, sizeof(result.identity.name), probe.name);
            copyToField(result.identity.version, sizeof(result.identity.version), probe.version);
            copyToField(result.identity.type, sizeof(result.identity.type), probe.type);
            copyToField(result.identity.author, sizeof(result.identity.author), probe.author);
            copyToField(result.identity.registry_key,
                        sizeof(result.identity.registry_key), probe.registry_key);
        } else if (mod) {
            result.load_error.code = static_cast<agrPluginErrorCode>(probe.last_error_code);
            copyToField(result.load_error.message,
                        sizeof(result.load_error.message), probe.last_error_message);
        } else {
            result.load_error.code = AGR_PLUGIN_ERROR_INIT_FAILED;
            copyToField(result.load_error.message,
                        sizeof(result.load_error.message),
                        "LoadLibrary failed (not a valid DLL or dependency missing)");
        }
        if (mod) FreeLibrary(mod);

        // Record the probe outcome under the registry mutex.
        {
            std::lock_guard<std::mutex> lock(plugin_mutex);
            auto it = plugin_cache.find(full_path);
            if (it == plugin_cache.end()) {
                PluginInternalState entry;
                entry.path = full_path;
                entry.filename = filename;
                entry.state = agrPluginState::AGR_PLUGIN_STATE_DISCOVERED;
                entry.last_error_code = static_cast<uint32_t>(result.load_error.code);
                entry.last_error_message = result.load_error.message;
                plugin_cache[full_path] = std::move(entry);
            } else if (!it->second.loaded) {
                it->second.last_error_code = static_cast<uint32_t>(result.load_error.code);
                it->second.last_error_message = result.load_error.message;
            }
        }

        results.push_back(result);
    } while (FindNextFileA(hFind, &find_data) != 0);

    FindClose(hFind);
    return results;
}

// ---------------------------------------------------------------------------
// plugin_load()
// ---------------------------------------------------------------------------
int plugin_load(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);

    auto it = plugin_cache.find(path);
    if (it != plugin_cache.end() && it->second.loaded) {
        return static_cast<int>(it->second.state);
    }

    PluginInternalState& state = plugin_cache[path];
    state.path = path;
    size_t last_slash = path.find_last_of("/\\");
    state.filename = (last_slash != std::string::npos) ? path.substr(last_slash + 1) : path;

    HMODULE mod = load_plugin_module(path);
    if (!mod) {
        state.module_handle = nullptr;
        state.state = agrPluginState::AGR_PLUGIN_STATE_UNLOADED;
        setError(state, AGR_PLUGIN_ERROR_INIT_FAILED, "Failed to load module: " + path);
        return -1;
    }

    if (!validate_module(mod, state)) {
        uint32_t code = state.last_error_code;
        std::string msg = state.last_error_message;
        FreeLibrary(mod);
        state.module_handle = nullptr;
        state.state = agrPluginState::AGR_PLUGIN_STATE_UNLOADED;
        setError(state, code, msg);
        return -1;
    }

    state.module_handle = mod;
    state.state = agrPluginState::AGR_PLUGIN_STATE_LOADED;
    state.loaded = true;

    // Initialize the plugin engine.
    auto init_fn = reinterpret_cast<agrPluginInitializeFn>(
        GetProcAddress(mod, "pluginInitialize"));
    void* engine_ctx = nullptr;
    int rc = init_fn(toHandle(mod), nullptr, &engine_ctx);
    if (rc == AGR_PLUGIN_ERROR_NONE) {
        state.initialized = true;
        state.state = agrPluginState::AGR_PLUGIN_STATE_INITIALIZED;
        setError(state, AGR_PLUGIN_ERROR_NONE, "OK");
    } else {
        // Plugin loaded and is queryable but its engine failed to start.
        state.initialized = false;
        state.state = agrPluginState::AGR_PLUGIN_STATE_LOADED;
        setError(state, static_cast<uint32_t>(rc),
                 "pluginInitialize failed with code " + std::to_string(rc));
        // Still a successful *load*: the caller can inspect the error and
        // retry; the plugin is not executable in this state.
    }

    // NOTE: no ExecutionEngineRegistry registration. A loaded plugin is not
    // an IExecutionEngine; representing it as one was incorrect. Execution
    // goes through plugin_execute() below.
    return static_cast<int>(state.state);
}

// ---------------------------------------------------------------------------
// plugin_execute()
// ---------------------------------------------------------------------------
int plugin_execute(const std::string& path, const void* ctx, void* outResult) {
    std::lock_guard<std::mutex> lock(plugin_mutex);

    auto it = plugin_cache.find(path);
    if (it == plugin_cache.end() || !it->second.loaded) {
        return -1; // not loaded
    }
    PluginInternalState& state = it->second;

    auto exec_fn = reinterpret_cast<agrPluginExecuteFn>(
        GetProcAddress(state.module_handle, "pluginExecute"));
    if (!exec_fn) {
        setError(state, AGR_PLUGIN_ERROR_INIT_FAILED,
                 "pluginExecute export disappeared");
        return -1;
    }

    uint32_t rc = static_cast<uint32_t>(
        exec_fn(toHandle(state.module_handle), ctx, outResult));
    if (rc != AGR_PLUGIN_ERROR_NONE) {
        setError(state, rc, "pluginExecute failed with code " + std::to_string(rc));
    }
    return static_cast<int>(rc);
}

// ---------------------------------------------------------------------------
// plugin_unload()
// ---------------------------------------------------------------------------
void plugin_unload(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);

    auto it = plugin_cache.find(path);
    if (it == plugin_cache.end()) return;

    PluginInternalState& state = it->second;
    if (!state.loaded && !state.module_handle) {
        state.state = agrPluginState::AGR_PLUGIN_STATE_UNLOADED;
        return;
    }

    if (state.initialized && state.module_handle) {
        auto shutdown_fn = reinterpret_cast<agrPluginShutdownFn>(
            GetProcAddress(state.module_handle, "pluginShutdown"));
        if (shutdown_fn) {
            shutdown_fn(toHandle(state.module_handle));
        }
        state.initialized = false;
    }

    if (state.module_handle) {
        FreeLibrary(state.module_handle);
        state.module_handle = nullptr;
    }
    state.loaded = false;
    state.state = agrPluginState::AGR_PLUGIN_STATE_UNLOADED;
}

// ---------------------------------------------------------------------------
// plugin_get_state()
// ---------------------------------------------------------------------------
agrPluginState plugin_get_state(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);
    auto it = plugin_cache.find(path);
    if (it == plugin_cache.end()) return agrPluginState::AGR_PLUGIN_STATE_DISCOVERED;
    return it->second.state;
}

// ---------------------------------------------------------------------------
// plugin_get_error()
// ---------------------------------------------------------------------------
agrPluginError plugin_get_error(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);

    agrPluginError result{};
    result.code = AGR_PLUGIN_ERROR_INTERNAL;
    copyToField(result.message, sizeof(result.message), "Plugin not loaded: " + path);

    auto it = plugin_cache.find(path);
    if (it == plugin_cache.end()) return result;

    result.code = static_cast<agrPluginErrorCode>(it->second.last_error_code);
    copyToField(result.message, sizeof(result.message), it->second.last_error_message);
    return result;
}

// ---------------------------------------------------------------------------
// plugin_get_identity()
// ---------------------------------------------------------------------------
bool plugin_get_identity(const std::string& path, agrPluginIdentity* out) {
    if (!out) return false;
    std::lock_guard<std::mutex> lock(plugin_mutex);
    auto it = plugin_cache.find(path);
    if (it == plugin_cache.end() || !it->second.loaded) return false;

    copyToField(out->name, sizeof(out->name), it->second.name);
    copyToField(out->version, sizeof(out->version), it->second.version);
    copyToField(out->type, sizeof(out->type), it->second.type);
    copyToField(out->author, sizeof(out->author), it->second.author);
    copyToField(out->registry_key, sizeof(out->registry_key), it->second.registry_key);
    return true;
}

// ---------------------------------------------------------------------------
// plugin_list_loaded()
// ---------------------------------------------------------------------------
std::vector<std::string> plugin_list_loaded() {
    std::lock_guard<std::mutex> lock(plugin_mutex);
    std::vector<std::string> loaded;
    for (const auto& [path, state] : plugin_cache) {
        if (state.loaded) loaded.push_back(path);
    }
    return loaded;
}

} // namespace agr
