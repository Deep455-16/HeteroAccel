// tests/test_p31_e2e_unified_cli_sdk.cpp
// Unified Collaborative Runtime — End-to-End Public C API, C++ SDK, and
// HeteroRuntime Verification across HydroXL, llama.cpp, and Colibri.
//
// Validates (Sections 3, 13 & 15.5):
//   1. Real GGUF v3 execution across ALL THREE engines (HydroXL + llama.cpp +
//      Colibri) through the unified C API (`ha_collaborate`, `ha_collaborate_stream`).
//   2. Real GGUF v3 execution through the C++ RAII SDK (`ha::Session::collaborate`,
//      `ha::Session::collaborate_stream`).
//   3. Real GGUF v3 execution through `agr::HeteroRuntime::generateCollaborative`
//      and `generateCollaborativeStreaming`.
//   4. Verification that every engine participates (`hydroxl_participated == 1`,
//      `llamacpp_participated == 1`, `colibri_participated == 1`), intermediate
//      results are exchanged, findings are deduplicated, and a unified coherent
//      response + telemetry report is returned.

#include "mini_test.h"
#include "gguf_fixture.h"

#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>
#include "inference/HeteroRuntime.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

int main() {
    std::cout << "=== test_p31_e2e_unified_cli_sdk ===\n";

    const char* kModelPath = "test_p31_e2e_tiny_llama.gguf";
    fixture::TinyLlamaSpec spec;
    spec.n_embd = 64;
    spec.n_head = 4;
    spec.n_kv_head = 2;
    spec.n_layer = 2;
    spec.n_ff = 128;
    spec.n_ctx = 512;
    fixture::writeTinyLlama(kModelPath, spec);

    // -----------------------------------------------------------------------
    // 1. Public C API: ha_collaborate & ha_collaborate_stream
    // -----------------------------------------------------------------------
    {
        HaRuntime rt = ha_runtime_create();
        AGR_CHECK(rt != nullptr);

        HaError err = HA_OK;
        HaModel model = ha_model_load(rt, kModelPath, &err);
        AGR_CHECK(model != nullptr);
        AGR_CHECK(err == HA_OK);

        HaSession session = ha_session_create(model, &err);
        AGR_CHECK(session != nullptr);
        AGR_CHECK(err == HA_OK);

        HaCollaborateParams cparams = {};
        cparams.prompt = "Analyze this C++ loop and optimize memory access:\n"
                         "for (int i = 0; i < N; ++i) { for (int j = 0; j < N; ++j) { sum += A[j * N + i]; } }";
        cparams.max_new_tokens = 6;
        cparams.temperature = 0.0f;
        cparams.top_p = 0.95f;
        cparams.max_ram_budget_bytes = 0;
        cparams.allow_parallel_waves = 1;
        cparams.preferred_gpu_backend = "cpu";

        HaCollaborateResult cres = ha_collaborate(session, &cparams);
        AGR_CHECK(cres.error == HA_OK);
        AGR_CHECK(cres.n_tasks_total == 3);
        AGR_CHECK(cres.n_tasks_completed == 3);
        AGR_CHECK(cres.n_waves == 2);
        AGR_CHECK(cres.hydroxl_participated == 1);
        AGR_CHECK(cres.llamacpp_participated == 1);
        AGR_CHECK(cres.colibri_participated == 1);
        AGR_CHECK(cres.unified_response != nullptr && std::strlen(cres.unified_response) > 0);
        AGR_CHECK(cres.telemetry_report != nullptr && std::strlen(cres.telemetry_report) > 0);
        AGR_CHECK(cres.overall_confidence > 0.5f);

        // Verify streaming C API
        std::string streamed_text;
        int chunk_count = 0;
        auto c_stream_cb = [](const char* text, int, void* udata) -> int {
            auto* pair_ptr = static_cast<std::pair<std::string*, int*>*>(udata);
            if (text) *(pair_ptr->first) += text;
            *(pair_ptr->second) += 1;
            return 0;
        };
        std::pair<std::string*, int*> cb_state{&streamed_text, &chunk_count};
        HaCollaborateResult stream_res = {};
        HaError stream_err = ha_collaborate_stream(session, &cparams, c_stream_cb, &cb_state, &stream_res);
        AGR_CHECK(stream_err == HA_OK);
        AGR_CHECK(chunk_count > 0);
        AGR_CHECK(!streamed_text.empty());
        AGR_CHECK(streamed_text == std::string(stream_res.unified_response));

        ha_session_destroy(session);
        ha_model_unload(model);
        ha_runtime_destroy(rt);
        std::cout << "  [OK] C API ha_collaborate & ha_collaborate_stream verified across all 3 engines\n";
    }

    // -----------------------------------------------------------------------
    // 2. Public C++ RAII SDK: ha::Session::collaborate & collaborate_stream
    // -----------------------------------------------------------------------
    {
        ha::Runtime rt;
        ha::Model model = rt.load_model(kModelPath);
        ha::Session session = model.create_session();

        ha::CollaborateParams params;
        params.prompt = "Explain grouped-query attention and KV cache memory scaling.";
        params.max_new_tokens = 6;
        params.preferred_gpu_backend = "cpu";

        ha::CollaborateResult res = session.collaborate(params);
        AGR_CHECK(res.ok());
        AGR_CHECK(res.hydroxl_participated);
        AGR_CHECK(res.llamacpp_participated);
        AGR_CHECK(res.colibri_participated);
        AGR_CHECK(res.n_tasks_completed == 3);
        AGR_CHECK(res.unified_response.find("HeteroAccel Unified Collaborative Response") != std::string::npos);
        AGR_CHECK(res.telemetry_report.find("Engine Participation Summary") != std::string::npos);

        std::string cpp_streamed;
        ha::CollaborateResult sres = session.collaborate_stream(
            params, [&](const std::string& chunk, int) -> bool {
                cpp_streamed += chunk;
                return true;
            });
        AGR_CHECK(sres.ok());
        AGR_CHECK(cpp_streamed == sres.unified_response);
        std::cout << "  [OK] C++ SDK ha::Session::collaborate & collaborate_stream verified\n";
    }

    // -----------------------------------------------------------------------
    // 3. HeteroRuntime::generateCollaborative & generateCollaborativeStreaming
    // -----------------------------------------------------------------------
    {
        agr::HeteroRuntime runtime;
        AGR_CHECK(runtime.initialize());

        agr::collaboration::CollaborativeRequest req;
        req.model_path = kModelPath;
        req.prompt = "Verify thread safety and complexity of a mutex-guarded std::map cache.";
        req.max_new_tokens_per_task = 6;
        req.preferred_gpu_backend = "cpu";
        req.conversation_history = {
            {"User", "We are designing a shared tensor residency cache."},
            {"Assistant", "We should consider lookup complexity and lock granularity."}
        };

        std::string rt_streamed;
        auto cres = runtime.generateCollaborativeStreaming(
            req, [&](const std::string& chunk, int) -> bool {
                rt_streamed += chunk;
                return true;
            });

        AGR_CHECK(cres.success);
        AGR_CHECK(cres.engineParticipated("hydroxl"));
        AGR_CHECK(cres.engineParticipated("llama.cpp"));
        AGR_CHECK(cres.engineParticipated("colibri"));
        AGR_CHECK(rt_streamed == cres.unified_response);
        AGR_CHECK(cres.fusion.validation_passed);
        AGR_CHECK(cres.fusion.deduplicated_findings_count > 0);

        std::cout << "\n" << cres.formatTelemetryReport() << "\n";
        runtime.shutdown();
        std::cout << "  [OK] HeteroRuntime collaborative multi-turn synthesis verified\n";
    }

    std::remove(kModelPath);
    AGR_TEST_MAIN_END();
}
