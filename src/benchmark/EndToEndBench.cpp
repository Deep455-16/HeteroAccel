// src/benchmark/EndToEndBench.cpp
#include "benchmark/EndToEndBench.h"

#include "core/HardwareInfo.h"
#include "hardware/CPUDetector.h"
#include "hardware/MemoryDetector.h"
#include "hardware/VulkanDetector.h"
#include "hardware/CUDADetector.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/HydroXLEngineAdapter.h"
#include "orchestration/LlamaCppEngineAdapter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>

#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
  #include <psapi.h>
#else
  #include <sys/resource.h>
  #include <sys/time.h>
  #include <unistd.h>
  #if defined(__APPLE__)
    #include <mach/mach.h>
  #endif
#endif

#ifndef AGR_BENCH_VERSION
#define AGR_BENCH_VERSION "unknown"
#endif
#ifndef AGR_BENCH_GIT_COMMIT
#define AGR_BENCH_GIT_COMMIT "unknown"
#endif
#ifndef AGR_BENCH_BUILD_TYPE
#define AGR_BENCH_BUILD_TYPE "unknown"
#endif

namespace agr {
namespace bench {

using Clock = std::chrono::steady_clock;
using orchestration::InferenceRequest;
using orchestration::InferenceResult;
using orchestration::IInferenceEngine;

static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---- host measurements -----------------------------------------------------
double processPeakRssMb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    return -1;
#elif defined(__APPLE__)
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
    return -1;
#else
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) return static_cast<double>(ru.ru_maxrss) / 1024.0;
    return -1;
#endif
}

double processCurrentRssMb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    return -1;
#elif defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &cnt) == KERN_SUCCESS)
        return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
    return -1;
#else
    std::ifstream f("/proc/self/statm");
    long pages_total = 0, pages_res = 0;
    if (f >> pages_total >> pages_res)
        return static_cast<double>(pages_res) * static_cast<double>(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
    return -1;
#endif
}

double processCpuSeconds() {
#if defined(_WIN32)
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return -1;
    auto to_s = [](const FILETIME& ft) {
        ULARGE_INTEGER x; x.LowPart = ft.dwLowDateTime; x.HighPart = ft.dwHighDateTime;
        return static_cast<double>(x.QuadPart) * 1e-7;
    };
    return to_s(k) + to_s(u);
#else
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
#endif
}

double measureHostMemoryBandwidthGBs(int mb, int repeats) {
    const size_t n = static_cast<size_t>(mb) * 1024 * 1024 / sizeof(double) / 3;
    std::vector<double> a(n, 1.0), b(n, 2.0), c(n, 0.5);
    double best = 0;
    volatile double sink = 0;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = Clock::now();
        for (size_t i = 0; i < n; ++i) a[i] = b[i] + 3.0 * c[i];   // STREAM triad
        double sec = msSince(t0) / 1000.0;
        sink = sink + a[n / 2];
        double bytes = 3.0 * static_cast<double>(n) * sizeof(double);
        if (sec > 0) best = std::max(best, bytes / sec / 1e9);
    }
    (void)sink;
    return best;
}

// ---- engines ---------------------------------------------------------------
std::vector<int> makePromptIds(int n) {
    std::vector<int> ids;
    ids.reserve(static_cast<size_t>(n));
    uint32_t x = 12345u;
    for (int i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        ids.push_back(1000 + static_cast<int>((x >> 8) % 4000u));   // < 5000: valid in all vocabs
    }
    return ids;
}

std::unique_ptr<IInferenceEngine> makeEngine(const std::string& id, const BenchSettings& s,
                                             std::string& st, std::string& why) {
    (void)s;
    auto& hw = orchestration::HardwareDetector::instance();
    hw.detect();
    if (id == "hydroxl_cpu") return std::make_unique<orchestration::HydroXLCpuEngineAdapter>();
    if (id == "llama_cpu")
        return std::make_unique<orchestration::LlamaCppEngineAdapter>("llama_cpu", "cpu");
    if (id == "hydroxl_vulkan" || id == "llama_vulkan") {
        if (!hw.has_physical_vulkan_gpu()) {
            st = "skipped";
            why = "no physical Vulkan GPU (software rasterizers such as llvmpipe are excluded)";
            return nullptr;
        }
        if (id == "hydroxl_vulkan") return std::make_unique<orchestration::HydroXLVulkanEngineAdapter>();
        return std::make_unique<orchestration::LlamaCppEngineAdapter>("llama_vulkan", "vulkan");
    }
    if (id == "cuda") {
        if (!orchestration::HardwareDetector::cuda_inference_compiled()) {
            st = "pending_hardware";
            why = "binary built without HETEROACCEL_ENABLE_CUDA";
            return nullptr;
        }
        if (!hw.has_cuda()) {
            st = "pending_hardware";
            why = "no usable NVIDIA CUDA device: " + hw.cuda_unavailable_reason();
            return nullptr;
        }
        int dev = 0;
        if (const auto* g = hw.best_cuda_gpu()) if (g->cuda_device_index >= 0) dev = g->cuda_device_index;
        return std::make_unique<orchestration::LlamaCppEngineAdapter>("cuda", "cuda", dev);
    }
    st = "failed";
    why = "unknown backend id '" + id + "'";
    return nullptr;
}

static InferenceRequest makeReq(int prompt_n, int gen_n) {
    InferenceRequest r;
    r.prompt_token_ids = makePromptIds(prompt_n);
    r.max_new_tokens = gen_n;
    r.temperature = 0.0f;
    return r;
}

struct Timed { double total_ms = 0; int out_tokens = 0; bool ok = false; std::string err; std::vector<int> ids; size_t peak_mem = 0; bool gpu = false; };

static Timed timedRun(IInferenceEngine& e, int prompt_n, int gen_n) {
    Timed t;
    auto req = makeReq(prompt_n, gen_n);
    auto t0 = Clock::now();
    InferenceResult r = e.run(req);
    t.total_ms = msSince(t0);
    t.ok = r.success;
    t.err = r.error;
    t.out_tokens = r.output_tokens;
    t.ids = r.output_token_ids;
    t.peak_mem = r.peak_memory_bytes;
    t.gpu = r.gpu_used;
    return t;
}

BackendBench benchBackend(const std::string& id, const BenchSettings& s) {
    BackendBench b;
    b.id = id;
    std::string st = "skipped", why;
    auto eng = makeEngine(id, s, st, why);
    if (!eng) { b.status = st; b.reason = why; return b; }

    // ---- model load latency -------------------------------------------------
    std::string err;
    auto t0 = Clock::now();
    if (!eng->initialize(s.model_path, err)) {
        b.status = "failed"; b.reason = err; return b;
    }
    b.load_ms = msSince(t0);
    b.rss_after_load_mb = processCurrentRssMb();

    for (int i = 0; i < s.warmup_runs; ++i) timedRun(*eng, s.prompt_tokens, 2);

    // ---- TTFT / decode / e2e ------------------------------------------------
    std::vector<double> ttfts, decs, e2es;
    double cpu0 = processCpuSeconds();
    auto wall0 = Clock::now();
    for (int i = 0; i < s.runs; ++i) {
        Timed first = timedRun(*eng, s.prompt_tokens, 1);
        Timed full  = timedRun(*eng, s.prompt_tokens, s.gen_tokens);
        if (!first.ok || !full.ok) { b.status = "failed"; b.reason = first.ok ? full.err : first.err; return b; }
        ttfts.push_back(first.total_ms);
        double decode_ms = full.total_ms - first.total_ms;
        if (full.out_tokens > 1 && decode_ms > 0)
            decs.push_back((full.out_tokens - 1) / (decode_ms / 1000.0));
        if (full.total_ms > 0) e2es.push_back(full.out_tokens / (full.total_ms / 1000.0));
        if (i == 0) {
            b.output_tokens = full.ids;
            if (full.peak_mem > 0) b.device_mem_mb = static_cast<double>(full.peak_mem) / (1024.0 * 1024.0);
        }
    }
    double wall_s = msSince(wall0) / 1000.0, cpu_s = processCpuSeconds() - cpu0;
    unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    if (cpu0 >= 0 && wall_s > 0) b.cpu_utilization_pct = 100.0 * cpu_s / wall_s / cores;
    b.ttft_ms = median(ttfts);
    b.decode_tps = median(decs);
    b.e2e_tps = median(e2es);
    if (!decs.empty()) {
        b.decode_tps_min = *std::min_element(decs.begin(), decs.end());
        b.decode_tps_max = *std::max_element(decs.begin(), decs.end());
    }
    {
        std::ifstream f(s.model_path, std::ios::binary | std::ios::ate);
        if (f) b.effective_weight_bw_gbs = static_cast<double>(f.tellg()) * b.decode_tps / 1e9;
    }

    // ---- natural-language reference prompt (empty ids => engine default prompt) ----
    {
        Timed nat = timedRun(*eng, 0, 12);
        if (nat.ok) b.natural_tokens = nat.ids;
    }

    // ---- context-length scaling --------------------------------------------
    for (int n : s.ctx_lengths) {
        Timed first = timedRun(*eng, n, 1);
        Timed full  = timedRun(*eng, n, 16);
        if (!first.ok || !full.ok) continue;   // e.g. beyond this backend's context window
        CtxPoint p;
        p.prompt_tokens = n;
        p.ttft_ms = first.total_ms;
        double dm = full.total_ms - first.total_ms;
        if (full.out_tokens > 1 && dm > 0) p.decode_tps = (full.out_tokens - 1) / (dm / 1000.0);
        b.ctx_scaling.push_back(p);
    }

    // ---- concurrent workload (one shared engine, N client threads) ---------
    if (s.concurrency > 1) {
        std::vector<double> lat(static_cast<size_t>(s.concurrency), 0.0);
        std::vector<int> toks(static_cast<size_t>(s.concurrency), 0);
        std::vector<std::thread> th;
        auto cw = Clock::now();
        for (int i = 0; i < s.concurrency; ++i) {
            th.emplace_back([&, i] {
                auto ts = Clock::now();
                Timed t = timedRun(*eng, s.prompt_tokens, 16);
                lat[static_cast<size_t>(i)] = msSince(ts);
                toks[static_cast<size_t>(i)] = t.ok ? t.out_tokens : 0;
            });
        }
        for (auto& t : th) t.join();
        double wall_ms = msSince(cw);
        int total = 0;
        for (int t : toks) total += t;
        b.concurrent_threads = s.concurrency;
        if (wall_ms > 0) b.concurrent_aggregate_tps = total / (wall_ms / 1000.0);
        b.concurrent_latency_p50_ms = median(lat);
        b.concurrent_latency_max_ms = *std::max_element(lat.begin(), lat.end());
    }

    b.peak_rss_mb = processPeakRssMb();
    b.device = id;
    if (auto* la = dynamic_cast<orchestration::LlamaCppEngineAdapter*>(eng.get())) b.device = la->device_name();
    else if (id == "hydroxl_cpu") b.device = "host CPU";
    b.status = "ok";
    eng->shutdown();
    return b;
}

// ---- JSON ------------------------------------------------------------------
std::string jsonEscape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); o += buf; }
                else o += static_cast<char>(c);
        }
    }
    return o;
}

static std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream o; o << std::setprecision(6) << v; return o.str();
}

std::string backendToJson(const BackendBench& b, int ind) {
    std::string p(static_cast<size_t>(ind), ' ');
    std::ostringstream o;
    o << p << "{\n";
    o << p << "  \"id\": \"" << jsonEscape(b.id) << "\",\n";
    o << p << "  \"status\": \"" << b.status << "\",\n";
    o << p << "  \"reason\": \"" << jsonEscape(b.reason) << "\",\n";
    if (b.status == "ok") {
        o << p << "  \"device\": \"" << jsonEscape(b.device) << "\",\n";
        o << p << "  \"load_ms\": " << num(b.load_ms) << ",\n";
        o << p << "  \"ttft_ms\": " << num(b.ttft_ms) << ",\n";
        o << p << "  \"decode_tps\": " << num(b.decode_tps) << ",\n";
        o << p << "  \"decode_tps_min\": " << num(b.decode_tps_min) << ",\n";
        o << p << "  \"decode_tps_max\": " << num(b.decode_tps_max) << ",\n";
        o << p << "  \"e2e_tps\": " << num(b.e2e_tps) << ",\n";
        o << p << "  \"peak_rss_mb\": " << num(b.peak_rss_mb) << ",\n";
        o << p << "  \"rss_after_load_mb\": " << num(b.rss_after_load_mb) << ",\n";
        o << p << "  \"peak_vram_mb\": " << (b.device_mem_mb >= 0 ? num(b.device_mem_mb) : "null") << ",\n";
        o << p << "  \"cpu_utilization_pct\": " << (b.cpu_utilization_pct >= 0 ? num(b.cpu_utilization_pct) : "null") << ",\n";
        o << p << "  \"gpu_utilization_pct\": null,\n";
        o << p << "  \"effective_weight_bandwidth_gbs_lower_bound\": " << num(b.effective_weight_bw_gbs) << ",\n";
        o << p << "  \"concurrent\": {\"threads\": " << b.concurrent_threads
          << ", \"aggregate_tps\": " << num(b.concurrent_aggregate_tps)
          << ", \"latency_p50_ms\": " << num(b.concurrent_latency_p50_ms)
          << ", \"latency_max_ms\": " << num(b.concurrent_latency_max_ms) << "},\n";
        o << p << "  \"context_scaling\": [";
        for (size_t i = 0; i < b.ctx_scaling.size(); ++i) {
            const auto& c = b.ctx_scaling[i];
            o << (i ? ", " : "") << "{\"prompt_tokens\": " << c.prompt_tokens
              << ", \"ttft_ms\": " << num(c.ttft_ms) << ", \"decode_tps\": "
              << (c.decode_tps > 0 ? num(c.decode_tps) : std::string("null")) << "}";
        }
        o << "],\n";
        o << p << "  \"natural_prompt_tokens\": [";
        for (size_t i = 0; i < b.natural_tokens.size(); ++i) o << (i ? "," : "") << b.natural_tokens[i];
        o << "],\n";
        o << p << "  \"output_tokens\": [";
        for (size_t i = 0; i < b.output_tokens.size(); ++i) o << (i ? "," : "") << b.output_tokens[i];
        o << "]\n";
    } else {
        o << p << "  \"device\": null\n";
    }
    o << p << "}";
    return o.str();
}

static std::string utcNow() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string metadataJson(const BenchSettings& s) {
    auto& hw = orchestration::HardwareDetector::instance();
    hw.detect();
    CPUInfo cpu = CPUDetector::detect();
    MemoryInfo mem = MemoryDetector::detect();
    VulkanInfo vk = VulkanDetector::detect();
    CUDAInfo cu = CUDADetector::detect(true);

    const char* os =
#if defined(_WIN32)
        "windows";
#elif defined(__APPLE__)
        "macos";
#else
        "linux";
#endif
    std::string compiler =
#if defined(_MSC_VER)
        "msvc " + std::to_string(_MSC_VER);
#elif defined(__clang__)
        std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
        std::string("gcc ") + __VERSION__;
#else
        "unknown";
#endif
    uint64_t model_bytes = 0;
    { std::ifstream f(s.model_path, std::ios::binary | std::ios::ate); if (f) model_bytes = static_cast<uint64_t>(f.tellg()); }
    std::string base = s.model_path;
    auto pos = base.find_last_of("/\\");
    if (pos != std::string::npos) base = base.substr(pos + 1);

    std::ostringstream o;
    o << "  \"metadata\": {\n";
    o << "    \"timestamp_utc\": \"" << utcNow() << "\",\n";
    o << "    \"heteroaccel_version\": \"" AGR_BENCH_VERSION "\",\n";
    o << "    \"git_commit\": \"" AGR_BENCH_GIT_COMMIT "\",\n";
    o << "    \"build_type\": \"" AGR_BENCH_BUILD_TYPE "\",\n";
    o << "    \"compiler\": \"" << jsonEscape(compiler) << "\",\n";
    o << "    \"os\": \"" << os << "\",\n";
    o << "    \"cuda_inference_compiled\": " << (orchestration::HardwareDetector::cuda_inference_compiled() ? "true" : "false") << ",\n";
    o << "    \"cpu\": {\"model\": \"" << jsonEscape(cpu.model_name) << "\", \"vendor\": \"" << jsonEscape(cpu.vendor)
      << "\", \"arch\": \"" << cpu.architecture << "\", \"physical_cores\": " << cpu.physical_cores
      << ", \"logical_processors\": " << cpu.logical_processors << "},\n";
    o << "    \"memory\": {\"total_mb\": " << mem.total_physical_mb << ", \"available_mb\": " << mem.available_physical_mb
      << ", \"host_triad_bandwidth_gbs\": " << num(measureHostMemoryBandwidthGBs()) << "},\n";
    o << "    \"vulkan\": {\"physical_gpu_available\": " << (vk.available ? "true" : "false") << ", \"devices\": [";
    for (size_t i = 0; i < vk.devices.size(); ++i) {
        const auto& d = vk.devices[i];
        o << (i ? ", " : "") << "{\"name\": \"" << jsonEscape(d.name) << "\", \"type\": \"" << d.device_type
          << "\", \"software_emulated\": " << (d.is_software_emulated ? "true" : "false")
          << ", \"device_local_mb\": " << d.device_local_memory_mb << "}";
    }
    o << "]},\n";
    o << "    \"cuda\": {\"available\": " << (cu.available ? "true" : "false") << ", \"driver_version\": " << cu.driver_version
      << ", \"reason\": \"" << jsonEscape(cu.unavailable_reason) << "\", \"devices\": [";
    for (size_t i = 0; i < cu.devices.size(); ++i) {
        const auto& d = cu.devices[i];
        o << (i ? ", " : "") << "{\"name\": \"" << jsonEscape(d.name) << "\", \"compute\": \"" << d.compute_major << "."
          << d.compute_minor << "\", \"total_mb\": " << d.total_memory_bytes / (1024 * 1024) << "}";
    }
    o << "]},\n";
    o << "    \"model\": {\"file\": \"" << jsonEscape(base) << "\", \"bytes\": " << model_bytes << "},\n";
    o << "    \"settings\": {\"prompt_tokens\": " << s.prompt_tokens << ", \"gen_tokens\": " << s.gen_tokens
      << ", \"warmup_runs\": " << s.warmup_runs << ", \"runs\": " << s.runs << ", \"temperature\": 0, \"concurrency\": "
      << s.concurrency << "},\n";
    o << "    \"notes\": [\"ttft_ms = wall time of a 1-token generation (prefill + first sample)\", "
         "\"decode_tps = (n-1)/(t_n - t_1) using greedy generation\", "
         "\"peak_rss_mb is the process peak working set; each backend is benchmarked in its own process\", "
         "\"peak_vram_mb is reported only where the engine exposes device memory accounting; null otherwise\", "
         "\"gpu_utilization_pct is not measured (no portable vendor-neutral API); null\", "
         "\"effective_weight_bandwidth is model_file_bytes*decode_tps, a lower bound on memory traffic, not a counter reading\", "
         "\"software Vulkan (llvmpipe/lavapipe) is never benchmarked as a GPU\"]\n";
    o << "  }";
    return o.str();
}

std::map<std::string, double> summaryMetrics(const std::vector<BackendBench>& all) {
    std::map<std::string, double> m;
    for (const auto& b : all) {
        if (b.status != "ok") continue;
        m[b.id + ".decode_tps"] = b.decode_tps;
        m[b.id + ".e2e_tps"]    = b.e2e_tps;
        m[b.id + ".ttft_ms"]    = b.ttft_ms;
        m[b.id + ".load_ms"]    = b.load_ms;
        m[b.id + ".peak_rss_mb"] = b.peak_rss_mb;
    }
    auto ratio = [&](const char* hx, const char* ll, const char* name) {
        auto a = m.find(std::string(hx) + ".decode_tps"), c = m.find(std::string(ll) + ".decode_tps");
        if (a != m.end() && c != m.end() && c->second > 0) m[name] = a->second / c->second;
    };
    ratio("hydroxl_cpu", "llama_cpu", "ratio.hydroxl_cpu_vs_llama_cpu.decode_tps");
    ratio("hydroxl_vulkan", "llama_vulkan", "ratio.hydroxl_vulkan_vs_llama_vulkan.decode_tps");
    return m;
}

std::string buildReport(const BenchSettings& s, const std::vector<BackendBench>& all) {
    std::ostringstream o;
    o << "{\n  \"schema\": \"heteroaccel-benchmark/1\",\n";
    o << metadataJson(s) << ",\n";
    o << "  \"backends\": [\n";
    for (size_t i = 0; i < all.size(); ++i) o << backendToJson(all[i]) << (i + 1 < all.size() ? ",\n" : "\n");
    o << "  ],\n";
    // Reference-correctness: identical greedy tokens between HydroXL and llama.cpp.
    auto find = [&](const std::string& id) -> const BackendBench* {
        for (const auto& b : all) if (b.id == id && b.status == "ok") return &b;
        return nullptr;
    };
    o << "  \"token_agreement\": {";
    bool first = true;
    for (auto pr : {std::make_pair("hydroxl_cpu", "llama_cpu"), std::make_pair("hydroxl_vulkan", "llama_vulkan")}) {
        const auto *a = find(pr.first), *c = find(pr.second);
        if (!a || !c) continue;
        size_t n = std::min(a->output_tokens.size(), c->output_tokens.size()), same = 0;
        while (same < n && a->output_tokens[same] == c->output_tokens[same]) ++same;
        size_t nn = std::min(a->natural_tokens.size(), c->natural_tokens.size()), nsame = 0;
        while (nsame < nn && a->natural_tokens[nsame] == c->natural_tokens[nsame]) ++nsame;
        o << (first ? "" : ", ") << "\"" << pr.first << "_vs_" << pr.second << "\": {\"natural_prompt_matching_prefix\": "
          << nsame << ", \"natural_prompt_compared\": " << nn << ", \"synthetic_prompt_matching_prefix\": "
          << same << ", \"synthetic_prompt_compared\": " << n << "}";
        first = false;
    }
    o << "},\n";
    o << "  \"summary\": {";
    auto m = summaryMetrics(all);
    bool f2 = true;
    for (const auto& kv : m) { o << (f2 ? "\n    " : ",\n    ") << "\"" << kv.first << "\": " << num(kv.second); f2 = false; }
    o << "\n  }\n}\n";
    return o.str();
}

// ---- child/parent transport -------------------------------------------------
std::string backendToText(const BackendBench& b) {
    std::ostringstream o;
    o << std::setprecision(10);
    auto esc = [](std::string s) { for (auto& c : s) if (c == '\n' || c == '\r') c = ' '; return s; };
    o << "id=" << b.id << "\nstatus=" << b.status << "\nreason=" << esc(b.reason) << "\ndevice=" << esc(b.device) << "\n";
    o << "load_ms=" << b.load_ms << "\nttft_ms=" << b.ttft_ms << "\ndecode_tps=" << b.decode_tps
      << "\ne2e_tps=" << b.e2e_tps << "\ndecode_tps_min=" << b.decode_tps_min << "\ndecode_tps_max=" << b.decode_tps_max
      << "\nconc_threads=" << b.concurrent_threads << "\nconc_tps=" << b.concurrent_aggregate_tps
      << "\nconc_p50=" << b.concurrent_latency_p50_ms << "\nconc_max=" << b.concurrent_latency_max_ms
      << "\npeak_rss_mb=" << b.peak_rss_mb << "\nrss_after_load_mb=" << b.rss_after_load_mb
      << "\ndevice_mem_mb=" << b.device_mem_mb << "\ncpu_util=" << b.cpu_utilization_pct
      << "\nweight_bw=" << b.effective_weight_bw_gbs << "\n";
    for (const auto& c : b.ctx_scaling) o << "ctx=" << c.prompt_tokens << "," << c.ttft_ms << "," << c.decode_tps << "\n";
    o << "tokens=";
    for (size_t i = 0; i < b.output_tokens.size(); ++i) o << (i ? "," : "") << b.output_tokens[i];
    o << "\nnatural=";
    for (size_t i = 0; i < b.natural_tokens.size(); ++i) o << (i ? "," : "") << b.natural_tokens[i];
    o << "\n";
    return o.str();
}

bool backendFromText(const std::string& text, BackendBench& b) {
    std::istringstream in(text);
    std::string line;
    bool got_id = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        auto d = [&] { return std::atof(v.c_str()); };
        if (k == "id") { b.id = v; got_id = true; }
        else if (k == "status") b.status = v;
        else if (k == "reason") b.reason = v;
        else if (k == "device") b.device = v;
        else if (k == "load_ms") b.load_ms = d();
        else if (k == "ttft_ms") b.ttft_ms = d();
        else if (k == "decode_tps") b.decode_tps = d();
        else if (k == "e2e_tps") b.e2e_tps = d();
        else if (k == "decode_tps_min") b.decode_tps_min = d();
        else if (k == "decode_tps_max") b.decode_tps_max = d();
        else if (k == "conc_threads") b.concurrent_threads = std::atoi(v.c_str());
        else if (k == "conc_tps") b.concurrent_aggregate_tps = d();
        else if (k == "conc_p50") b.concurrent_latency_p50_ms = d();
        else if (k == "conc_max") b.concurrent_latency_max_ms = d();
        else if (k == "peak_rss_mb") b.peak_rss_mb = d();
        else if (k == "rss_after_load_mb") b.rss_after_load_mb = d();
        else if (k == "device_mem_mb") b.device_mem_mb = d();
        else if (k == "cpu_util") b.cpu_utilization_pct = d();
        else if (k == "weight_bw") b.effective_weight_bw_gbs = d();
        else if (k == "ctx") {
            CtxPoint c; double a = 0;
            if (std::sscanf(v.c_str(), "%lf,%lf,%lf", &a, &c.ttft_ms, &c.decode_tps) == 3) {
                c.prompt_tokens = static_cast<int>(a); b.ctx_scaling.push_back(c);
            }
        } else if (k == "natural") {
            std::stringstream ss(v); std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) b.natural_tokens.push_back(std::atoi(t.c_str()));
        } else if (k == "tokens") {
            std::stringstream ss(v); std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) b.output_tokens.push_back(std::atoi(t.c_str()));
        }
    }
    return got_id;
}

// ---- regression ------------------------------------------------------------
static bool endsWith(const std::string& s, const char* suf) {
    size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

std::vector<RegressionFinding> compareMetrics(const std::map<std::string, double>& base,
                                              const std::map<std::string, double>& cur, double tol) {
    std::vector<RegressionFinding> out;
    for (const auto& kv : base) {
        auto it = cur.find(kv.first);
        if (it == cur.end() || kv.second <= 0) continue;
        bool higher_better = endsWith(kv.first, ".decode_tps") || endsWith(kv.first, ".e2e_tps");
        bool lower_better  = endsWith(kv.first, ".ttft_ms") || endsWith(kv.first, ".load_ms");
        if (!higher_better && !lower_better) continue;
        RegressionFinding f;
        f.metric = kv.first; f.baseline = kv.second; f.current = it->second;
        f.change_pct = 100.0 * (f.current - f.baseline) / f.baseline;
        f.regressed = higher_better ? (f.current < f.baseline * (1.0 - tol))
                                    : (f.current > f.baseline * (1.0 + tol));
        out.push_back(f);
    }
    return out;
}

std::map<std::string, double> parseSummary(const std::string& js) {
    std::map<std::string, double> m;
    auto pos = js.find("\"summary\"");
    if (pos == std::string::npos) return m;
    pos = js.find('{', pos);
    auto end = js.find('}', pos);
    if (pos == std::string::npos || end == std::string::npos) return m;
    std::string body = js.substr(pos + 1, end - pos - 1);
    size_t i = 0;
    while ((i = body.find('"', i)) != std::string::npos) {
        size_t j = body.find('"', i + 1);
        if (j == std::string::npos) break;
        std::string key = body.substr(i + 1, j - i - 1);
        size_t c = body.find(':', j);
        if (c == std::string::npos) break;
        char* endp = nullptr;
        double v = std::strtod(body.c_str() + c + 1, &endp);
        if (endp != body.c_str() + c + 1) m[key] = v;
        i = static_cast<size_t>(endp - body.c_str());
    }
    return m;
}

} // namespace bench
} // namespace agr
