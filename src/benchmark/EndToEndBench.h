// src/benchmark/EndToEndBench.h
// Phase 29: end-to-end inference benchmarking (model load, TTFT, decode tok/s,
// context scaling, concurrency, memory, utilisation) over the real
// IInferenceEngine implementations, plus machine-readable reporting and
// regression comparison. Nothing here is synthetic: every number comes from
// timing a real inference run on the host. Metrics that cannot be measured
// portably are reported as null with an explanatory note, never invented.
#pragma once

#include "orchestration/IInferenceEngine.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace agr {
namespace bench {

struct BenchSettings {
    std::string model_path;
    int prompt_tokens = 32;
    int gen_tokens    = 32;
    int warmup_runs   = 1;
    int runs          = 3;
    std::vector<int> ctx_lengths = {16, 64, 256};
    int concurrency   = 4;   // threads sharing one engine instance
};

struct CtxPoint {
    int prompt_tokens = 0;
    double ttft_ms = 0;
    double decode_tps = 0;
};

struct BackendBench {
    std::string id;
    std::string status = "not_run";   // ok | skipped | failed | pending_hardware
    std::string reason;
    std::string device;
    double load_ms = 0;
    double ttft_ms = 0;               // median, prompt_tokens prompt + 1 generated token
    double decode_tps = 0;            // median, (n-1) tokens / (total - ttft)
    double e2e_tps = 0;               // median, n tokens / total
    double decode_tps_min = 0, decode_tps_max = 0;
    std::vector<CtxPoint> ctx_scaling;
    int    concurrent_threads = 0;
    double concurrent_aggregate_tps = 0;
    double concurrent_latency_p50_ms = 0, concurrent_latency_max_ms = 0;
    double peak_rss_mb = 0;           // process peak working set (isolate per backend!)
    double rss_after_load_mb = 0;
    double device_mem_mb = -1;        // -1 = not measurable for this backend
    double cpu_utilization_pct = -1;  // process CPU time / wall / logical cores during decode
    double effective_weight_bw_gbs = 0; // model_file_bytes * decode_tps (lower bound, labelled estimate)
    std::vector<int> output_tokens;   // greedy output for the synthetic prompt
    std::vector<int> natural_tokens;  // greedy output for the default natural-language prompt
                                      // ("The capital of France is"), the meaningful agreement check
};

/// Deterministic synthetic prompt ids (valid in every supported vocabulary).
std::vector<int> makePromptIds(int n);

/// Known ids: hydroxl_cpu, hydroxl_vulkan, llama_cpu, llama_vulkan, cuda.
/// Returns nullptr and sets status/reason when the backend cannot exist on this host.
std::unique_ptr<orchestration::IInferenceEngine> makeEngine(const std::string& id,
                                                            const BenchSettings& s,
                                                            std::string& skip_status,
                                                            std::string& skip_reason);

BackendBench benchBackend(const std::string& id, const BenchSettings& s);

/// Line-based key=value serialisation used to pass results from isolated
/// per-backend child processes back to the parent report builder.
std::string backendToText(const BackendBench& b);
bool backendFromText(const std::string& text, BackendBench& out);

// ---- reporting -------------------------------------------------------------
std::string jsonEscape(const std::string& s);
std::string backendToJson(const BackendBench& b, int indent = 4);
/// Host + software metadata block (JSON object body, without surrounding braces).
std::string metadataJson(const BenchSettings& s);
/// Flat metric map: "<backend>.decode_tps" etc. Used by regression checks.
std::map<std::string, double> summaryMetrics(const std::vector<BackendBench>& all);
std::string buildReport(const BenchSettings& s, const std::vector<BackendBench>& all);

// ---- regression ------------------------------------------------------------
struct RegressionFinding {
    std::string metric;
    double baseline = 0, current = 0, change_pct = 0;
    bool regressed = false;
};
/// higher-is-better: *.decode_tps, *.e2e_tps.  lower-is-better: *.ttft_ms, *.load_ms.
/// A metric regresses when it is worse than baseline by more than `tolerance`
/// (fraction, e.g. 0.25). Metrics missing on either side are ignored.
std::vector<RegressionFinding> compareMetrics(const std::map<std::string, double>& baseline,
                                              const std::map<std::string, double>& current,
                                              double tolerance);
/// Parses the flat "summary" object of a report written by buildReport().
std::map<std::string, double> parseSummary(const std::string& report_json);

// ---- host measurements -----------------------------------------------------
double processPeakRssMb();
double processCurrentRssMb();
double processCpuSeconds();      // user+kernel CPU time of this process
/// Triad-style host memory bandwidth (GB/s) over `mb` MiB working sets; real measurement.
double measureHostMemoryBandwidthGBs(int mb = 256, int repeats = 5);

} // namespace bench
} // namespace agr
