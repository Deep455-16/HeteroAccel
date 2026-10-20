// tests/test_p18_plugin.cpp
// Phase 19: plugin discovery, ABI negotiation, lifecycle and execution tests.
//
// Covers:
//   - discovery finds the well-formed test plugin and reports its identity
//   - a plugin with a mismatched ABI version is rejected with a clear
//     diagnostic, before any of its other entry points run
//   - load -> initialize -> execute -> shutdown -> unload lifecycle
//   - plugin code is genuinely INVOKED (pluginExecute runs and fills the
//     result) — not merely loaded
//   - repeated load/unload cycles
//   - concurrent discovery/load/state queries do not crash or corrupt state
//   - a non-plugin DLL file is reported as not loadable
//
// SECURITY NOTE (verified by design, not sandboxing): all DLLs here run
// in-process with host privileges.

#include "mini_test.h"

#include "plugin/PluginLoader.h"
#include "plugin/PluginAbis.h"
#include "engine/ExecutionResult.h"
#include "engine/ExecutionContext.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace agr;

#ifndef AGR_TEST_PLUGIN_DIR
#define AGR_TEST_PLUGIN_DIR "."
#endif

namespace {
std::string dirOf(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return (s == std::string::npos) ? "." : p.substr(0, s);
}
bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}
} // namespace

int main() {
    const std::string good_dir = AGR_TEST_PLUGIN_DIR;
    const std::string good_dll = good_dir + "/hydroxl_test_plugin.dll";
    const std::string bad_dll  = good_dir + "/test_plugin_badabi.dll";

    AGR_CHECK(std::filesystem::exists(good_dll)); // plugin DLLs were built
    AGR_CHECK(std::filesystem::exists(bad_dll));

    // Directory with only junk .dll files (not valid PE images)
    std::string junk_dir = "p18_junk_plugins";
    std::filesystem::create_directories(junk_dir);
    {
        std::ofstream j(junk_dir + "/junk.dll", std::ios::binary);
        j << "this is not a PE image";
    }

    // Old state from a previous run must not leak into this one
    plugin_unload(good_dll);

    // ==================================================================
    // 1. Discovery (directory containing both plugins)
    // ==================================================================
    {
        _putenv_s("AGR_PLUGIN_DIR", good_dir.c_str());
        auto results = plugin_discover();

        bool found_good = false, found_bad = false;
        for (const auto& r : results) {
            std::string fn(r.filename);
            if (fn == "hydroxl_test_plugin.dll") {
                found_good = true;
                AGR_CHECK(r.loadable);
                AGR_CHECK(r.load_error.code == AGR_PLUGIN_ERROR_NONE);
                AGR_CHECK(std::string(r.identity.name) == "hydroxl-test");
                AGR_CHECK(std::string(r.identity.type) == "LLM");
            }
            if (fn == "test_plugin_badabi.dll") {
                found_bad = true;
                AGR_CHECK(!r.loadable);
                AGR_CHECK(contains(r.load_error.message, "ABI version mismatch"));
            }
        }
        AGR_CHECK(found_good);
        AGR_CHECK(found_bad);
    }
    {   // junk directory: invalid PE reported not loadable
        _putenv_s("AGR_PLUGIN_DIR", junk_dir.c_str());
        auto results = plugin_discover();
        AGR_CHECK(results.size() == 1);
        if (!results.empty()) {
            AGR_CHECK(!results[0].loadable);
            AGR_CHECK(contains(results[0].load_error.message, "LoadLibrary failed"));
        }
    }

    // ==================================================================
    // 2. Load / initialize / identity / error
    // ==================================================================
    {
        int st = plugin_load(good_dll);
        AGR_CHECK(st == static_cast<int>(agrPluginState::AGR_PLUGIN_STATE_INITIALIZED));
        AGR_CHECK(plugin_get_state(good_dll) ==
                  agrPluginState::AGR_PLUGIN_STATE_INITIALIZED);

        agrPluginIdentity ident{};
        AGR_CHECK(plugin_get_identity(good_dll, &ident));
        AGR_CHECK(std::string(ident.name) == "hydroxl-test");
        AGR_CHECK(std::string(ident.version) == "1.0.0");

        agrPluginError err = plugin_get_error(good_dll);
        AGR_CHECK(err.code == AGR_PLUGIN_ERROR_NONE);

        // plugin_list_loaded contains it
        bool listed = false;
        for (const auto& p : plugin_list_loaded()) {
            if (p == good_dll) listed = true;
        }
        AGR_CHECK(listed);
    }

    // ==================================================================
    // 3. Execution: plugin code actually runs in-process
    // ==================================================================
    {
        ExecutionContext ctx{}; // the test plugin only requires non-null
        ExecutionResult result{};
        int rc = plugin_execute(good_dll, &ctx, &result);
        AGR_CHECK(rc == AGR_PLUGIN_ERROR_NONE);
        AGR_CHECK(result.success);
        AGR_CHECK(result.output == "test operation");
        AGR_CHECK(result.telemetry.backend_type == ExecutionBackendType::CPU);
        AGR_CHECK(result.telemetry.output_tokens == 1);
    }

    // ==================================================================
    // 4. Bad-ABI plugin is refused at load
    // ==================================================================
    {
        int st = plugin_load(bad_dll);
        AGR_CHECK(st == -1);
        agrPluginError err = plugin_get_error(bad_dll);
        AGR_CHECK(contains(err.message, "ABI version mismatch"));
        AGR_CHECK(plugin_get_state(bad_dll) == agrPluginState::AGR_PLUGIN_STATE_UNLOADED);
        // Executing an unloaded plugin fails
        ExecutionContext ctx{};
        ExecutionResult r{};
        AGR_CHECK(plugin_execute(bad_dll, &ctx, &r) == -1);
    }

    // ==================================================================
    // 5. Unload / repeated lifecycle
    // ==================================================================
    {
        plugin_unload(good_dll);
        AGR_CHECK(plugin_get_state(good_dll) ==
                  agrPluginState::AGR_PLUGIN_STATE_UNLOADED);
        AGR_CHECK(plugin_get_identity(good_dll, nullptr) == false);

        // Reload and execute again
        int st = plugin_load(good_dll);
        AGR_CHECK(st == static_cast<int>(agrPluginState::AGR_PLUGIN_STATE_INITIALIZED));
        ExecutionContext ctx{};
        ExecutionResult r2{};
        AGR_CHECK(plugin_execute(good_dll, &ctx, &r2) == AGR_PLUGIN_ERROR_NONE);
        AGR_CHECK(r2.output == "test operation");

        plugin_unload(good_dll);
        plugin_unload(good_dll); // idempotent
        AGR_CHECK(plugin_get_state(good_dll) ==
                  agrPluginState::AGR_PLUGIN_STATE_UNLOADED);
    }

    // ==================================================================
    // 6. Concurrency: parallel load / unload / state / discover
    // ==================================================================
    {
        std::atomic<bool> stop{false};
        std::atomic<int> failures{0};
        std::vector<std::thread> threads;

        for (int t = 0; t < 3; ++t) {
            threads.emplace_back([&] {
                while (!stop.load()) {
                    int st = plugin_load(good_dll);
                    if (st != -1 &&
                        st != static_cast<int>(agrPluginState::AGR_PLUGIN_STATE_INITIALIZED) &&
                        st != static_cast<int>(agrPluginState::AGR_PLUGIN_STATE_UNLOADED)) {
                        ++failures;
                    }
                    plugin_get_state(good_dll);
                }
            });
        }
        threads.emplace_back([&] {
            for (int i = 0; i < 50 && !stop.load(); ++i) {
                plugin_unload(good_dll);
            }
            stop = true;
        });
        threads.emplace_back([&] {
            _putenv_s("AGR_PLUGIN_DIR", good_dir.c_str());
            for (int i = 0; i < 5 && !stop.load(); ++i) {
                plugin_discover();
            }
        });

        for (auto& th : threads) th.join();
        AGR_CHECK(failures.load() == 0);
        plugin_unload(good_dll); // leave clean state
    }

    // Cleanup junk dir
    std::filesystem::remove_all(junk_dir);

    AGR_TEST_MAIN_END()
}
