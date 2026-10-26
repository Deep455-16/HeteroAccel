// tests/test_p20_benchmark.cpp
// Phase 20: Performance and memory benchmark harness for HydroXL.
// Measures cold load time, warm inference throughput (prompt & generation),
// TTFT, and peak RSS memory across multiple runs.
// Exits 77 (SKIP) if model file is absent.

#include "mini_test.h"
#include "hydroxl/HydroXLEngine.h"
#include "engine/ExecutionContext.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

static void getMemoryStats(double& current_rss_mb, double& peak_rss_mb) {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                            sizeof(pmc))) {
        current_rss_mb = static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
        peak_rss_mb    = static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    } else {
        current_rss_mb = 0.0;
        peak_rss_mb    = 0.0;
    }
}
#else
static void getMemoryStats(double& current_rss_mb, double& peak_rss_mb) {
    current_rss_mb = 0.0;
    peak_rss_mb    = 0.0;
}
#endif

using namespace agr;

static const char* DEFAULT_MODEL =
    "C:\\HeteroAccel\\Models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";

static const char* modelPath() {
    const char* e = std::getenv("HETEROACCEL_MODEL_PATH");
    return (e && *e) ? e : DEFAULT_MODEL;
}

static bool modelAvailable() {
    return std::filesystem::exists(modelPath());
}

struct MetricStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
};

static MetricStats calcStats(const std::vector<double>& vals) {
    MetricStats ms;
    if (vals.empty()) return ms;
    ms.min_val = vals[0];
    ms.max_val = vals[0];
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
        if (v < ms.min_val) ms.min_val = v;
        if (v > ms.max_val) ms.max_val = v;
    }
    ms.mean = sum / vals.size();
    if (vals.size() > 1) {
        double sq_sum = 0.0;
        for (double v : vals) {
            double diff = v - ms.mean;
            sq_sum += diff * diff;
        }
        ms.stddev = std::sqrt(sq_sum / (vals.size() - 1));
    }
    return ms;
}

int main() {
    if (!modelAvailable()) {
        std::cout << "[SKIP] Model not found: " << modelPath() << "\n";
        std::cout << "       Set HETEROACCEL_MODEL_PATH env var to override.\n";
        return 77;
    }

    std::cout << "=== HeteroAccel / HydroXL Phase 20 Benchmark ===\n";
    std::cout << "Model path: " << modelPath() << "\n";

    // -----------------------------------------------------------------------
    // 1. Cold Load Time Measurement
    // -----------------------------------------------------------------------
    HydroXLEngine eng;
    AGR_CHECK(eng.initialize());

    ModelDescriptor desc;
    desc.model_id     = "qwen2.5-0.5b";
    desc.source_path  = modelPath();
    desc.format       = ModelFormat::GGUF;
    desc.architecture = "qwen2";

    double cold_load_ms = 0.0;
    {
        auto t0 = std::chrono::steady_clock::now();
        bool ok = eng.loadModel(desc);
        auto t1 = std::chrono::steady_clock::now();
        cold_load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!ok) {
            std::cerr << "[FAIL] Failed to load model: " << eng.lastError() << "\n";
            return 1;
        }
        AGR_CHECK(ok);
        AGR_CHECK(eng.isModelLoaded());
    }
    std::cout << "Cold load time: " << cold_load_ms << " ms\n";

    // -----------------------------------------------------------------------
    // 2. Warm Inference Benchmark (5 Runs)
    // -----------------------------------------------------------------------
    const int NUM_RUNS = 5;
    const int GEN_TOKENS = 8;
    const std::string PROMPT = "The capital of France is";

    std::vector<double> prompt_eval_ms_list;
    std::vector<double> prompt_eval_tok_s_list;
    std::vector<double> gen_ms_list;
    std::vector<double> gen_tok_s_list;
    std::vector<double> ttft_ms_list;
    std::vector<double> total_ms_list;

    int prompt_tokens_count = 0;
    int gen_tokens_count = 0;

    for (int r = 0; r < NUM_RUNS; ++r) {
        ExecutionContext ctx;
        ctx.prompt      = PROMPT;
        ctx.max_tokens  = GEN_TOKENS;
        ctx.temperature = 0.0f; // greedy for consistent timing
        ctx.seed        = 42;

        auto t0 = std::chrono::steady_clock::now();
        auto result = eng.execute(ctx);
        auto t1 = std::chrono::steady_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        AGR_CHECK(result.success);
        AGR_CHECK(result.telemetry.output_tokens > 0);

        prompt_tokens_count = result.telemetry.prompt_tokens;
        gen_tokens_count    = result.telemetry.output_tokens;

        double p_ms = result.telemetry.prompt_eval_ms;
        double p_tok_s = (p_ms > 0.0) ? (result.telemetry.prompt_tokens / (p_ms / 1000.0)) : 0.0;

        double g_ms = result.telemetry.generation_ms;
        double g_tok_s = result.telemetry.tokens_per_sec;

        double ttft = result.telemetry.ttft_ms;

        prompt_eval_ms_list.push_back(p_ms);
        prompt_eval_tok_s_list.push_back(p_tok_s);
        gen_ms_list.push_back(g_ms);
        gen_tok_s_list.push_back(g_tok_s);
        ttft_ms_list.push_back(ttft);
        total_ms_list.push_back(wall_ms);

        std::cout << "  Run " << (r + 1) << "/" << NUM_RUNS << ":"
                  << " prompt=" << p_ms << "ms (" << p_tok_s << " tok/s)"
                  << ", gen=" << g_ms << "ms (" << g_tok_s << " tok/s)"
                  << ", ttft=" << ttft << "ms"
                  << ", wall=" << wall_ms << "ms\n";
    }

    // -----------------------------------------------------------------------
    // 3. Memory Statistics
    // -----------------------------------------------------------------------
    double cur_rss_mb = 0.0;
    double peak_rss_mb = 0.0;
    getMemoryStats(cur_rss_mb, peak_rss_mb);

    // -----------------------------------------------------------------------
    // 4. Calculate Aggregate Statistics
    // -----------------------------------------------------------------------
    MetricStats p_ms_stats    = calcStats(prompt_eval_ms_list);
    MetricStats p_tok_stats   = calcStats(prompt_eval_tok_s_list);
    MetricStats g_ms_stats    = calcStats(gen_ms_list);
    MetricStats g_tok_stats   = calcStats(gen_tok_s_list);
    MetricStats ttft_stats    = calcStats(ttft_ms_list);
    MetricStats total_stats   = calcStats(total_ms_list);

    // -----------------------------------------------------------------------
    // 5. Output Key=Value and JSON Report
    // -----------------------------------------------------------------------
    std::cout << "\n=== Key=Value Results ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "model_path=" << modelPath() << "\n";
    std::cout << "cold_load_ms=" << cold_load_ms << "\n";
    std::cout << "runs=" << NUM_RUNS << "\n";
    std::cout << "prompt_tokens=" << prompt_tokens_count << "\n";
    std::cout << "gen_tokens=" << gen_tokens_count << "\n";
    std::cout << "prompt_eval_tok_s_mean=" << p_tok_stats.mean << "\n";
    std::cout << "prompt_eval_tok_s_stddev=" << p_tok_stats.stddev << "\n";
    std::cout << "prompt_eval_ms_mean=" << p_ms_stats.mean << "\n";
    std::cout << "prompt_eval_ms_stddev=" << p_ms_stats.stddev << "\n";
    std::cout << "gen_tok_s_mean=" << g_tok_stats.mean << "\n";
    std::cout << "gen_tok_s_stddev=" << g_tok_stats.stddev << "\n";
    std::cout << "gen_ms_mean=" << g_ms_stats.mean << "\n";
    std::cout << "gen_ms_stddev=" << g_ms_stats.stddev << "\n";
    std::cout << "ttft_ms_mean=" << ttft_stats.mean << "\n";
    std::cout << "ttft_ms_stddev=" << ttft_stats.stddev << "\n";
    std::cout << "total_ms_mean=" << total_stats.mean << "\n";
    std::cout << "total_ms_stddev=" << total_stats.stddev << "\n";
    std::cout << "current_rss_mb=" << cur_rss_mb << "\n";
    std::cout << "peak_rss_mb=" << peak_rss_mb << "\n";

    std::cout << "\n=== JSON Output ===\n";
    std::cout << "{\n"
              << "  \"benchmark\": \"HydroXL_Phase20\",\n"
              << "  \"model\": \"" << desc.model_id << "\",\n"
              << "  \"cold_load_ms\": " << cold_load_ms << ",\n"
              << "  \"runs\": " << NUM_RUNS << ",\n"
              << "  \"prompt_tokens\": " << prompt_tokens_count << ",\n"
              << "  \"gen_tokens\": " << gen_tokens_count << ",\n"
              << "  \"prompt_eval\": {\n"
              << "    \"mean_tok_s\": " << p_tok_stats.mean << ",\n"
              << "    \"stddev_tok_s\": " << p_tok_stats.stddev << ",\n"
              << "    \"mean_ms\": " << p_ms_stats.mean << ",\n"
              << "    \"stddev_ms\": " << p_ms_stats.stddev << "\n"
              << "  },\n"
              << "  \"generation\": {\n"
              << "    \"mean_tok_s\": " << g_tok_stats.mean << ",\n"
              << "    \"stddev_tok_s\": " << g_tok_stats.stddev << ",\n"
              << "    \"mean_ms\": " << g_ms_stats.mean << ",\n"
              << "    \"stddev_ms\": " << g_ms_stats.stddev << "\n"
              << "  },\n"
              << "  \"ttft_ms\": {\n"
              << "    \"mean\": " << ttft_stats.mean << ",\n"
              << "    \"stddev\": " << ttft_stats.stddev << "\n"
              << "  },\n"
              << "  \"memory_mb\": {\n"
              << "    \"current_rss\": " << cur_rss_mb << ",\n"
              << "    \"peak_rss\": " << peak_rss_mb << "\n"
              << "  }\n"
              << "}\n";

    AGR_CHECK(cold_load_ms > 0.0);
    AGR_CHECK(p_tok_stats.mean > 0.0);
    AGR_CHECK(g_tok_stats.mean > 0.0);
    AGR_CHECK(peak_rss_mb > 0.0);

    AGR_TEST_MAIN_END()
}
