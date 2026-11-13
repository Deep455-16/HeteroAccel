// tests/test_p29_perf_regression.cpp
// Phase 29: benchmark harness correctness + configurable performance regression gate.
//
// Env:
//   HETEROACCEL_MODEL_PATH       GGUF used for the integration part (skipped if unset)
//   HETEROACCEL_PERF_BASELINE    report JSON from heteroaccel-bench to gate against (optional)
//   HETEROACCEL_PERF_TOLERANCE   allowed fractional slowdown, default 0.30
#include "mini_test.h"
#include "benchmark/EndToEndBench.h"

#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace agr::bench;

static int countChar(const std::string& s, char c) {
    int n = 0; for (char x : s) n += (x == c); return n;
}

int main() {
    std::cout << "== test_p29_perf_regression ==\n";

    // ---- prompt determinism ---------------------------------------------------
    AGR_CHECK(makePromptIds(32) == makePromptIds(32));
    AGR_CHECK(makePromptIds(8).size() == 8);
    for (int id : makePromptIds(256)) AGR_CHECK(id >= 1000 && id < 5000);

    // ---- regression comparator (pure logic, synthetic numbers) -------------------
    {
        std::map<std::string, double> base = {{"a.decode_tps", 100}, {"a.ttft_ms", 50}, {"a.load_ms", 1000},
                                              {"a.peak_rss_mb", 500}, {"b.decode_tps", 10}};
        // Within tolerance (-10% tps, +10% ttft) -> no regression.
        std::map<std::string, double> ok = {{"a.decode_tps", 90}, {"a.ttft_ms", 55}, {"a.load_ms", 1100}, {"b.decode_tps", 12}};
        for (auto& f : compareMetrics(base, ok, 0.25)) AGR_CHECK(!f.regressed);
        // Throughput -40% and latency +60% -> regress at 25% tolerance, not at 70%.
        std::map<std::string, double> bad = {{"a.decode_tps", 60}, {"a.ttft_ms", 80}};
        int regressed = 0;
        for (auto& f : compareMetrics(base, bad, 0.25)) regressed += f.regressed;
        AGR_CHECK(regressed == 2);
        for (auto& f : compareMetrics(base, bad, 0.70)) AGR_CHECK(!f.regressed);
        // Improvements never regress; non-gated metrics (rss) and missing keys are ignored.
        std::map<std::string, double> better = {{"a.decode_tps", 200}, {"a.ttft_ms", 10}, {"a.peak_rss_mb", 99999}};
        auto fs = compareMetrics(base, better, 0.0);
        AGR_CHECK(fs.size() == 2);
        for (auto& f : fs) AGR_CHECK(!f.regressed);
        // Exactly at the boundary is allowed.
        std::map<std::string, double> edge = {{"a.decode_tps", 75}};
        for (auto& f : compareMetrics(base, edge, 0.25)) AGR_CHECK(!f.regressed);
    }

    // ---- serialisation round trips ---------------------------------------------
    {
        BackendBench b;
        b.id = "hydroxl_cpu"; b.status = "ok"; b.device = "host CPU";
        b.load_ms = 1234.5; b.ttft_ms = 321.25; b.decode_tps = 7.5; b.e2e_tps = 6.5;
        b.peak_rss_mb = 2048; b.device_mem_mb = -1; b.cpu_utilization_pct = 55.5;
        b.ctx_scaling = {{16, 100, 8}, {64, 300, 7}};
        b.output_tokens = {1, 2, 3, 40000};
        BackendBench r;
        AGR_CHECK(backendFromText(backendToText(b), r));
        AGR_CHECK(r.id == b.id && r.status == "ok");
        AGR_CHECK(r.decode_tps == b.decode_tps && r.ttft_ms == b.ttft_ms);
        AGR_CHECK(r.ctx_scaling.size() == 2 && r.ctx_scaling[1].prompt_tokens == 64);
        AGR_CHECK(r.output_tokens == b.output_tokens);

        BenchSettings s; s.model_path = "nonexistent.gguf";
        std::vector<BackendBench> all = {b};
        BackendBench skipped; skipped.id = "cuda"; skipped.status = "pending_hardware"; skipped.reason = "no \"GPU\"";
        all.push_back(skipped);
        std::string rep = buildReport(s, all);
        AGR_CHECK(countChar(rep, '{') == countChar(rep, '}'));
        AGR_CHECK(countChar(rep, '[') == countChar(rep, ']'));
        AGR_CHECK(rep.find("\"schema\": \"heteroaccel-benchmark/1\"") != std::string::npos);
        AGR_CHECK(rep.find("\"software_emulated\"") != std::string::npos || rep.find("\"vulkan\"") != std::string::npos);
        AGR_CHECK(rep.find("pending_hardware") != std::string::npos);
        AGR_CHECK(rep.find("\\\"GPU\\\"") != std::string::npos);   // escaped
        auto sum = parseSummary(rep);
        AGR_CHECK(sum.count("hydroxl_cpu.decode_tps") == 1);
        AGR_CHECK(sum["hydroxl_cpu.decode_tps"] == 7.5);
        AGR_CHECK(sum.count("cuda.decode_tps") == 0);   // non-ok backends contribute nothing
    }

    // ---- host measurements are real, positive numbers ---------------------------------
    AGR_CHECK(processPeakRssMb() > 1.0);
    AGR_CHECK(processCurrentRssMb() > 1.0);
    AGR_CHECK(processCpuSeconds() >= 0.0);
    double bw = measureHostMemoryBandwidthGBs(64, 3);
    std::cout << "  host triad bandwidth: " << bw << " GB/s\n";
    AGR_CHECK(bw > 0.1 && bw < 5000);

    // ---- integration: real model, real engines ------------------------------------------
    const char* mp = std::getenv("HETEROACCEL_MODEL_PATH");
    std::ifstream probe(mp ? mp : "", std::ios::binary);
    if (!mp || !probe.good()) {
        std::cout << "[SKIP] integration: HETEROACCEL_MODEL_PATH not set/readable\n";
        AGR_TEST_MAIN_END();
    }
    probe.close();

    BenchSettings s;
    s.model_path = mp; s.prompt_tokens = 8; s.gen_tokens = 8; s.warmup_runs = 0; s.runs = 2;
    s.ctx_lengths = {16}; s.concurrency = 2;

    std::vector<BackendBench> all;
    for (const char* id : {"hydroxl_cpu", "llama_cpu"}) {
        BackendBench b = benchBackend(id, s);
        std::cout << "  " << id << ": " << b.status << " load=" << b.load_ms << "ms ttft=" << b.ttft_ms
                  << "ms decode=" << b.decode_tps << " tok/s rss=" << b.peak_rss_mb << "MB\n";
        AGR_CHECK(b.status == "ok");
        AGR_CHECK(b.load_ms > 0 && b.ttft_ms > 0 && b.decode_tps > 0 && b.e2e_tps > 0);
        AGR_CHECK(b.peak_rss_mb > 0);
        AGR_CHECK(b.rss_after_load_mb > 0);
        AGR_CHECK(!b.output_tokens.empty());
        AGR_CHECK(b.concurrent_aggregate_tps > 0);
        AGR_CHECK(!b.ctx_scaling.empty());
        all.push_back(b);
    }
    // Synthetic random-token prompts have near-tied logits, so their argmax is not a
    // meaningful reference check; it is reported for information only.
    if (all.size() == 2 && !all[0].output_tokens.empty() && !all[1].output_tokens.empty())
        std::cout << "  synthetic-prompt first token: hydroxl=" << all[0].output_tokens[0]
                  << " llama=" << all[1].output_tokens[0] << " (informational)\n";
    // Natural-language prompt: identical greedy settings must give the same leading tokens
    // (later divergence is possible: Q4 weights x int8 activations in llama.cpp vs f32 here).
    if (all.size() == 2) {
        size_t n = std::min(all[0].natural_tokens.size(), all[1].natural_tokens.size()), same = 0;
        while (same < n && all[0].natural_tokens[same] == all[1].natural_tokens[same]) ++same;
        std::cout << "  natural-prompt greedy agreement: " << same << "/" << n << " tokens\n";
        AGR_CHECK(same >= 4);
    }

    // Software/absent GPUs must be reported honestly, never benchmarked as GPUs.
    {
        BackendBench c = benchBackend("cuda", s);
        AGR_CHECK(c.status == "ok" || c.status == "pending_hardware");
        BackendBench bad = benchBackend("no_such_backend", s);
        AGR_CHECK(bad.status == "failed");
    }

    // ---- optional gate against a stored baseline --------------------------------------------
    if (const char* bp = std::getenv("HETEROACCEL_PERF_BASELINE")) {
        std::ifstream f(bp, std::ios::binary);
        std::stringstream ss; ss << f.rdbuf();
        double tol = 0.30;
        if (const char* t = std::getenv("HETEROACCEL_PERF_TOLERANCE")) tol = std::atof(t);
        auto findings = compareMetrics(parseSummary(ss.str()), summaryMetrics(all), tol);
        AGR_CHECK(!findings.empty());
        for (auto& fd : findings) {
            std::cout << "  [gate] " << fd.metric << " " << fd.baseline << " -> " << fd.current
                      << " (" << fd.change_pct << "%)" << (fd.regressed ? "  REGRESSED" : "") << "\n";
            AGR_CHECK(!fd.regressed);
        }
    } else {
        std::cout << "  (no HETEROACCEL_PERF_BASELINE set; regression gate not applied)\n";
    }

    AGR_TEST_MAIN_END();
}
