// tests/test_p16_backend_executor.cpp
// Phase 16: IBackendExecutor / CpuBackendExecutor capability and availability tests.

#include "primitives/cpu/CpuBackendExecutor.h"
#include "mini_test.h"
#include <iostream>
#include <cmath>

using namespace agr::primitives;

static TensorShape F32(std::initializer_list<size_t> dims) {
    return TensorShape(std::vector<size_t>(dims), DataType::FLOAT32);
}

int main() {
    CpuBackendExecutor exec;

    // Availability
    AGR_CHECK(exec.isAvailable());
    AGR_CHECK(exec.name() == std::string("CpuBackendExecutor"));

    // Capabilities
    auto caps = exec.capabilities();
    AGR_CHECK(caps.available);
    AGR_CHECK(caps.copy);
    AGR_CHECK(caps.fill);
    AGR_CHECK(caps.add);
    AGR_CHECK(caps.multiply);
    AGR_CHECK(caps.scale);
    AGR_CHECK(caps.sum);
    AGR_CHECK(caps.maxVal);
    // CPU executor does NOT claim device transfer ops
    AGR_CHECK(!caps.deviceBufferAllocation);
    AGR_CHECK(!caps.hostToDevice);
    AGR_CHECK(!caps.deviceToHost);

    // fill via executor
    {
        Tensor t = Tensor::zeros(F32({4}));
        auto r = exec.fill(t, 2.0f);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 4; ++i)
            AGR_CHECK(std::fabs(*t.at<float>(i) - 2.0f) < 1e-5f);
    }

    // add via executor
    {
        Tensor a = Tensor::zeros(F32({3}));
        Tensor b = Tensor::zeros(F32({3}));
        Tensor c = Tensor::zeros(F32({3}));
        exec.fill(a, 10.0f);
        exec.fill(b, 20.0f);
        auto r = exec.add(c, a, b);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 3; ++i)
            AGR_CHECK(std::fabs(*c.at<float>(i) - 30.0f) < 1e-5f);
    }

    // multiply via executor
    {
        Tensor a = Tensor::zeros(F32({4}));
        Tensor b = Tensor::zeros(F32({4}));
        Tensor c = Tensor::zeros(F32({4}));
        exec.fill(a, 3.0f);
        exec.fill(b, 4.0f);
        auto r = exec.multiply(c, a, b);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 4; ++i)
            AGR_CHECK(std::fabs(*c.at<float>(i) - 12.0f) < 1e-5f);
    }

    // scale via executor
    {
        Tensor src = Tensor::zeros(F32({4}));
        Tensor dst = Tensor::zeros(F32({4}));
        exec.fill(src, 5.0f);
        auto r = exec.scale(dst, src, 3.0f);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 4; ++i)
            AGR_CHECK(std::fabs(*dst.at<float>(i) - 15.0f) < 1e-5f);
    }

    // sum via executor
    {
        Tensor t = Tensor::zeros(F32({4}));
        exec.fill(t, 2.5f);
        double s = 0.0;
        auto r = exec.sum(t, &s);
        AGR_CHECK(r.success);
        AGR_CHECK(std::fabs(s - 10.0) < 1e-4);
    }

    // maxVal via executor
    {
        Tensor t = Tensor::zeros(F32({5}));
        float* p = t.dataAs<float>();
        p[0]=1.0f; p[1]=9.0f; p[2]=3.0f; p[3]=7.0f; p[4]=5.0f;
        float m = 0.0f;
        auto r = exec.maxVal(t, &m);
        AGR_CHECK(r.success);
        AGR_CHECK(std::fabs(m - 9.0f) < 1e-5f);
    }

    // Error propagation: invalid tensor
    {
        Tensor bad;
        auto r = exec.fill(bad, 1.0f);
        AGR_CHECK(!r.success);
        AGR_CHECK(!r.error.empty());
    }

    std::cout << "test_p16_backend_executor PASSED\n";
    AGR_TEST_MAIN_END();
}
