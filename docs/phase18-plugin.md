# HeteroAccel — Phase 18: Dynamic Plugin System

## Overview

Phase 18 adds a **dynamic engine plugin system** that allows third-party inference engines to be discovered, loaded, and invoked at runtime without recompiling the host application.

## Plugin ABI

Plugins are native Windows DLLs that export a fixed set of C-linkage, `__stdcall` entry points defined in `src/plugin/PluginAbis.h`.

### Required exports

| Export | Signature | Purpose |
|--------|-----------|---------|
| `pluginGetAbiVersion` | `uint32_t()` | First call; must equal `AGR_PLUGIN_ABI_VERSION` |
| `pluginGetIdentity` | `int(handle, agrPluginIdentity*)` | Fills name, version, type, author, registry_key |
| `pluginInitialize` | `int(handle, identity*, void**)` | Allocates engine context |
| `pluginExecute` | `int(handle, ctx*, result*)` | Runs one inference call |
| `pluginShutdown` | `int(handle)` | Frees engine context |

### ABI version

`AGR_PLUGIN_ABI_VERSION = 1`. A plugin reporting a different value is **rejected at load time** before any other entry point is called. This prevents silent misbehavior from mismatched plugin versions.

## Security Model

Plugins are loaded with `LoadLibrary` into the host process with **full host-process privileges**. There is no sandbox. Only load plugins from trusted locations.

## Plugin Lifecycle

```
plugin_discover()  →  plugin_load()  →  plugin_execute()  →  plugin_unload()
```

1. **Discovery** (`plugin_discover()`): Scans the directory set by `AGR_PLUGIN_DIR` (env var, default `./plugins`). Each `.dll` is probed: loaded temporarily, ABI checked, identity fetched. Returns one `agrPluginDiscoveryResult` per file.

2. **Load** (`plugin_load(path)`): Calls `pluginGetAbiVersion`, `pluginGetIdentity`, `pluginInitialize`. Returns the plugin's `agrPluginState` on success or `-1` on failure.

3. **Execute** (`plugin_execute(path, ctx, result)`): Calls `pluginExecute` on a fully initialized plugin.

4. **Unload** (`plugin_unload(path)`): Calls `pluginShutdown`, then `FreeLibrary`. Idempotent.

## Thread Safety

All plugin registry operations are protected by a `std::mutex`. Discovery, load, unload, and state queries are safe to call concurrently.

## Limitations

- Plugin factories (`pluginCreateEngine`) are defined in the ABI but not yet integrated into `ExecutionEngineRegistry`. Plugins are invoked directly via `plugin_execute()`.
- Plugin-based LLM inference is a demonstration of the lifecycle mechanism. Actual inference in a plugin would require the plugin to embed its own model loading and tokenization.

## Test Plugin

`test_plugin.cpp` builds as `hydroxl_test_plugin.dll`. It implements the full plugin ABI and returns `"test operation"` from `pluginExecute`, demonstrating a correct lifecycle.

`tests/test_plugin_badabi.cpp` builds as `test_plugin_badabi.dll`. It reports `AGR_PLUGIN_ABI_VERSION + 99` and is rejected by the host.

## Build

The plugin system is built as `agr_plugin` (static library, `src/plugin/PluginLoader.cpp`). Link against it with `target_link_libraries(... PRIVATE agr_plugin)`.
