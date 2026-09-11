#include "benchmark/VectorAddBenchmark.h"
#include "gpu/CpuReference.h"

#include <chrono>
#include <cstdio>
#include <sstream>
#include <vector>

namespace agr {

namespace {

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::string humanCount(size_t n) {
    if (n >= 1'000'000) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1fM", n / 1'000'000.0);
        return buf;
    }
    if (n >= 1'000) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1fK", n / 1'000.0);
        return buf;
    }
    return std::to_string(n);
}

} // namespace

BenchmarkReport VectorAddBenchmark::run(GPUBackend& backend, const std::vector<size_t>& sizes) {
    BenchmarkReport report;
    if (backend.isAvailable()) {
        report.device_name = backend.deviceName();
    }

    for (size_t n : sizes) {
        BenchmarkSizeResult r;
        r.element_count = n;

        std::vector<float> a(n), b(n), cCpu(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.25f;
        }

        double cpuStart = nowMs();
        CpuReference::vectorAdd(a.data(), b.data(), cCpu.data(), n);
        r.cpu_time_ms = nowMs() - cpuStart;

        if (!backend.isAvailable()) {
            r.gpu_ran = false;
            r.gpu_skip_reason = backend.lastError().empty()
                ? "Vulkan backend not available"
                : backend.lastError();
            report.results.push_back(r);
            continue;
        }

        Buffer bufA = backend.createBuffer(n * sizeof(float));
        Buffer bufB = backend.createBuffer(n * sizeof(float));
        Buffer bufC = backend.createBuffer(n * sizeof(float));

        if (bufA.id == 0 || bufB.id == 0 || bufC.id == 0) {
            r.gpu_ran = false;
            r.gpu_skip_reason = "Buffer allocation failed: " + backend.lastError();
            report.results.push_back(r);
            if (bufA.id) backend.destroyBuffer(bufA);
            if (bufB.id) backend.destroyBuffer(bufB);
            if (bufC.id) backend.destroyBuffer(bufC);
            continue;
        }

        bool ok = true;
        ok = ok && backend.upload(bufA, a.data(), n, &r.gpu_upload_ms);
        double uploadB = 0.0;
        ok = ok && backend.upload(bufB, b.data(), n, &uploadB);
        r.gpu_upload_ms += uploadB; // total host->GPU transfer time for this run

        if (ok) {
            ok = backend.executeVectorAdd(bufA, bufB, bufC, static_cast<uint32_t>(n), &r.gpu_execute_ms);
        }

        std::vector<float> cGpu(n, 0.0f);
        if (ok) {
            ok = backend.download(bufC, cGpu.data(), n, &r.gpu_download_ms);
        }

        backend.destroyBuffer(bufA);
        backend.destroyBuffer(bufB);
        backend.destroyBuffer(bufC);

        if (!ok) {
            r.gpu_ran = false;
            r.gpu_skip_reason = "GPU execution failed: " + backend.lastError();
            report.results.push_back(r);
            continue;
        }

        r.gpu_ran = true;
        r.gpu_total_ms = r.gpu_upload_ms + r.gpu_execute_ms + r.gpu_download_ms;
        r.correctness_checked = true;
        r.correctness_passed = CpuReference::nearlyEqual(cCpu.data(), cGpu.data(), n);
        r.speedup = (r.gpu_total_ms > 0.0) ? (r.cpu_time_ms / r.gpu_total_ms) : 0.0;

        report.results.push_back(r);
    }

    return report;
}

std::string VectorAddBenchmark::formatText(const BenchmarkReport& report) {
    std::ostringstream out;
    out << "Vector Add Benchmark (CPU vs Vulkan)\n";
    out << "GPU device: " << (report.device_name.empty() ? "N/A (Vulkan unavailable)" : report.device_name) << "\n\n";

    char line[512];
    std::snprintf(line, sizeof(line), "%-10s %10s %10s %10s %10s %10s %8s %10s\n",
                  "Size", "CPU(ms)", "Upload", "Execute", "Download", "GPUTotal", "Speedup", "Valid");
    out << line;

    for (const auto& r : report.results) {
        if (r.gpu_ran) {
            std::snprintf(line, sizeof(line), "%-10s %10.3f %10.3f %10.3f %10.3f %10.3f %7.2fx %10s\n",
                          humanCount(r.element_count).c_str(),
                          r.cpu_time_ms, r.gpu_upload_ms, r.gpu_execute_ms, r.gpu_download_ms,
                          r.gpu_total_ms, r.speedup,
                          r.correctness_passed ? "PASS" : "FAIL");
        } else {
            std::snprintf(line, sizeof(line), "%-10s %10.3f %10s %10s %10s %10s %8s %10s\n",
                          humanCount(r.element_count).c_str(), r.cpu_time_ms,
                          "N/A", "N/A", "N/A", "N/A", "N/A", "N/A");
        }
        out << line;
    }

    for (const auto& r : report.results) {
        if (!r.gpu_ran) {
            out << "  [" << humanCount(r.element_count) << "] GPU skipped: " << r.gpu_skip_reason << "\n";
        }
    }

    return out.str();
}

} // namespace agr
