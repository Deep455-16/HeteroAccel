#include "mini_test.h"
#include "benchmark/VectorAddBenchmark.h"
#include "gpu/GPUBackend.h"

namespace {

// A backend double that always reports "unavailable" -- lets us test the
// N/A formatting path deterministically, without depending on whether
// this machine happens to have a GPU.
class UnavailableBackend : public agr::GPUBackend {
public:
    bool initialize() override { return false; }
    bool isAvailable() const override { return false; }
    std::string deviceName() const override { return ""; }
    std::string lastError() const override { return "test double: intentionally unavailable"; }
    agr::Buffer createBuffer(size_t) override { return agr::Buffer{}; }
    void destroyBuffer(agr::Buffer&) override {}
    bool upload(const agr::Buffer&, const float*, size_t, double*) override { return false; }
    bool download(const agr::Buffer&, float*, size_t, double*) override { return false; }
    bool executeVectorAdd(const agr::Buffer&, const agr::Buffer&, agr::Buffer&, uint32_t, double*) override { return false; }
    void shutdown() override {}
};

} // namespace

int main() {
    std::cout << "== test_benchmark_formatting ==\n";

    UnavailableBackend backend;
    agr::BenchmarkReport report = agr::VectorAddBenchmark::run(backend, {1024, 4096});

    AGR_CHECK(report.results.size() == 2);
    for (const auto& r : report.results) {
        // CPU side must always have run for real.
        AGR_CHECK(r.cpu_time_ms >= 0.0);
        // GPU side must be honestly marked unavailable, never a fabricated number.
        AGR_CHECK(!r.gpu_ran);
        AGR_CHECK(!r.gpu_skip_reason.empty());
        AGR_CHECK(r.gpu_total_ms == 0.0);
        AGR_CHECK(r.speedup == 0.0);
    }

    std::string text = agr::VectorAddBenchmark::formatText(report);
    std::cout << text;

    AGR_CHECK(text.find("N/A") != std::string::npos);
    AGR_CHECK(text.find("intentionally unavailable") != std::string::npos);

    AGR_TEST_MAIN_END();
}
