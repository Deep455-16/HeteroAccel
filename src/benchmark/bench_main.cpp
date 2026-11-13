// src/benchmark/bench_main.cpp
// heteroaccel-bench: reproducible end-to-end benchmark driver (Phase 29).
//
//   heteroaccel-bench --model M.gguf [--backends hydroxl_cpu,llama_cpu,...]
//        [--prompt 32] [--gen 32] [--runs 3] [--warmup 1] [--ctx 16,64,256]
//        [--concurrency 4] [--out report.json]
//        [--baseline old.json --tolerance 0.25]     (exit 3 on regression)
//
// Each backend runs in its own child process so peak-RSS figures are not
// contaminated by earlier backends. Use --in-process to disable isolation.
#include "benchmark/EndToEndBench.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace agr::bench;

static std::vector<std::string> splitCsv(const std::string& s) {
    std::vector<std::string> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(t);
    return v;
}
static std::string readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main(int argc, char** argv) {
    BenchSettings s;
    std::vector<std::string> backends = {"hydroxl_cpu", "llama_cpu", "hydroxl_vulkan", "llama_vulkan", "cuda"};
    std::string out_path, baseline_path, child_id, child_out;
    double tol = 0.25;
    bool in_process = false;

    if (const char* e = std::getenv("HETEROACCEL_MODEL_PATH")) s.model_path = e;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--model") s.model_path = next();
        else if (a == "--backends") backends = splitCsv(next());
        else if (a == "--prompt") s.prompt_tokens = std::atoi(next().c_str());
        else if (a == "--gen") s.gen_tokens = std::atoi(next().c_str());
        else if (a == "--runs") s.runs = std::atoi(next().c_str());
        else if (a == "--warmup") s.warmup_runs = std::atoi(next().c_str());
        else if (a == "--concurrency") s.concurrency = std::atoi(next().c_str());
        else if (a == "--ctx") { s.ctx_lengths.clear(); for (auto& t : splitCsv(next())) s.ctx_lengths.push_back(std::atoi(t.c_str())); }
        else if (a == "--out") out_path = next();
        else if (a == "--baseline") baseline_path = next();
        else if (a == "--tolerance") tol = std::atof(next().c_str());
        else if (a == "--in-process") in_process = true;
        else if (a == "--child") child_id = next();
        else if (a == "--child-out") child_out = next();
        else { std::cerr << "unknown argument: " << a << "\n"; return 2; }
    }
    if (s.model_path.empty()) { std::cerr << "error: --model (or HETEROACCEL_MODEL_PATH) required\n"; return 2; }

    // Child mode: benchmark exactly one backend and hand the result back via a file.
    if (!child_id.empty()) {
        BackendBench b = benchBackend(child_id, s);
        std::ofstream(child_out, std::ios::binary) << backendToText(b);
        return 0;
    }

    std::vector<BackendBench> results;
    for (const auto& id : backends) {
        std::cerr << "[bench] " << id << " ...\n";
        if (in_process) { results.push_back(benchBackend(id, s)); continue; }
        std::string tmp = (out_path.empty() ? std::string("heteroaccel_bench") : out_path) + "." + id + ".tmp";
        std::string ctx;
        for (size_t k = 0; k < s.ctx_lengths.size(); ++k) ctx += (k ? "," : "") + std::to_string(s.ctx_lengths[k]);
        if (ctx.empty()) ctx = "0";
        std::ostringstream cmd;
        cmd << "\"" << argv[0] << "\" --child " << id << " --child-out \"" << tmp << "\" --model \"" << s.model_path
            << "\" --prompt " << s.prompt_tokens << " --gen " << s.gen_tokens << " --runs " << s.runs
            << " --warmup " << s.warmup_runs << " --concurrency " << s.concurrency << " --ctx " << ctx;
        std::string full = cmd.str();
#if defined(_WIN32)
        full = "\"" + full + "\"";   // cmd.exe quoting for std::system
#endif
        int rc = std::system(full.c_str());
        BackendBench b;
        std::string txt = readAll(tmp);
        if (rc != 0 || !backendFromText(txt, b)) {
            b = BackendBench{}; b.id = id; b.status = "failed";
            b.reason = "benchmark child process failed (exit code " + std::to_string(rc) + ")";
        }
        std::remove(tmp.c_str());
        results.push_back(b);
    }

    std::string report = buildReport(s, results);
    if (out_path.empty()) std::cout << report;
    else { std::ofstream(out_path, std::ios::binary) << report; std::cerr << "[bench] wrote " << out_path << "\n"; }

    for (const auto& b : results)
        std::cerr << "[bench] " << b.id << ": " << b.status << (b.reason.empty() ? "" : " (" + b.reason + ")") << "\n";

    if (!baseline_path.empty()) {
        auto base = parseSummary(readAll(baseline_path));
        auto findings = compareMetrics(base, parseSummary(report), tol);
        bool bad = false;
        for (const auto& f : findings) {
            std::cerr << "[regress] " << (f.regressed ? "FAIL " : "ok   ") << f.metric << " baseline=" << f.baseline
                      << " current=" << f.current << " (" << f.change_pct << "%)\n";
            bad = bad || f.regressed;
        }
        if (findings.empty()) std::cerr << "[regress] no comparable metrics in baseline\n";
        return bad ? 3 : 0;
    }
    return 0;
}
