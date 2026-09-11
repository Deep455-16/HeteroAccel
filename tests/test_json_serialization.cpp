#include "mini_test.h"
#include "json/HardwareJson.h"

namespace {

// Very small brace/bracket balance checker -- enough to catch structural
// bugs in the hand-rolled serializer without pulling in a JSON parser
// dependency just for tests.
bool bracesBalanced(const std::string& s) {
    int curly = 0, square = 0;
    bool inString = false;
    bool escaped = false;
    for (char c : s) {
        if (inString) {
            if (escaped) { escaped = false; continue; }
            if (c == '\\') { escaped = true; continue; }
            if (c == '"') { inString = false; }
            continue;
        }
        if (c == '"') { inString = true; continue; }
        if (c == '{') curly++;
        if (c == '}') curly--;
        if (c == '[') square++;
        if (c == ']') square--;
        if (curly < 0 || square < 0) return false;
    }
    return curly == 0 && square == 0 && !inString;
}

} // namespace

int main() {
    std::cout << "== test_json_serialization ==\n";

    agr::HardwareInfo info;
    info.cpu.vendor = "GenuineIntel";
    info.cpu.model_name = "Intel(R) Core(TM) i5-1235U";
    info.cpu.logical_processors = 12;
    info.cpu.physical_cores = 10;
    info.cpu.architecture = "x86_64";

    info.memory.total_physical_mb = 16384;
    info.memory.available_physical_mb = 9000;

    agr::GPUInfo gpu;
    gpu.name = "Intel(R) Iris(R) Xe Graphics";
    gpu.vendor = agr::GPUVendor::Intel;
    gpu.kind = agr::GPUKind::Integrated;
    gpu.dedicated_vram_mb = 128;
    gpu.shared_system_memory_mb = 8192;
    gpu.total_accessible_memory_mb = 8320;
    info.gpus.push_back(gpu);

    info.vulkan.available = true;
    info.vulkan.instance_api_version_major = 1;
    info.vulkan.instance_api_version_minor = 3;
    agr::VulkanDeviceInfo vdev;
    vdev.name = "Intel(R) Iris(R) Xe Graphics";
    vdev.device_type = "Integrated GPU";
    vdev.has_compute_capable_queue = true;
    info.vulkan.devices.push_back(vdev);

    info.cuda.available = false;
    info.cuda.unavailable_reason = "No NVIDIA driver present";

    std::string json = toJson(info);

    std::cout << json << "\n";

    AGR_CHECK(!json.empty());
    AGR_CHECK(bracesBalanced(json));
    AGR_CHECK(json.find("\"cpu\"") != std::string::npos);
    AGR_CHECK(json.find("Iris(R) Xe") != std::string::npos);
    AGR_CHECK(json.find("\"vulkan\"") != std::string::npos);
    AGR_CHECK(json.find("\"cuda\"") != std::string::npos);
    AGR_CHECK(json.find("No NVIDIA driver present") != std::string::npos);

    // Edge case: a string containing quotes/backslashes must escape cleanly.
    agr::HardwareInfo edge;
    edge.cpu.model_name = "Weird \"CPU\" \\Name\\";
    std::string edgeJson = toJson(edge);
    AGR_CHECK(bracesBalanced(edgeJson));

    AGR_TEST_MAIN_END();
}
