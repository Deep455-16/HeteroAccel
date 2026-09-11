#include "json/HardwareJson.h"
#include <sstream>

namespace agr {

namespace {

std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

struct Writer {
    std::ostringstream out;
    int indentLevel = 0;
    bool pretty = true;

    void nl() { if (pretty) out << "\n" << std::string(indentLevel * 2, ' '); }
    void str(const std::string& key, const std::string& value, bool comma) {
        nl();
        out << "\"" << key << "\": \"" << escape(value) << "\"" << (comma ? "," : "");
    }
    void num(const std::string& key, unsigned long long value, bool comma) {
        nl();
        out << "\"" << key << "\": " << value << (comma ? "," : "");
    }
    void bnum(const std::string& key, bool value, bool comma) {
        nl();
        out << "\"" << key << "\": " << (value ? "true" : "false") << (comma ? "," : "");
    }
};

} // namespace

std::string toJson(const HardwareInfo& info, bool pretty) {
    Writer w;
    w.pretty = pretty;
    std::ostringstream& out = w.out;

    out << "{";
    w.indentLevel++;

    // ---- cpu ----
    w.nl();
    out << "\"cpu\": {";
    w.indentLevel++;
    w.str("vendor", info.cpu.vendor, true);
    w.str("model_name", info.cpu.model_name, true);
    w.num("logical_processors", info.cpu.logical_processors, true);
    w.num("physical_cores", info.cpu.physical_cores, true);
    w.str("architecture", info.cpu.architecture, false);
    w.indentLevel--;
    w.nl();
    out << "},";

    // ---- memory ----
    w.nl();
    out << "\"memory\": {";
    w.indentLevel++;
    w.num("total_mb", info.memory.total_physical_mb, true);
    w.num("available_mb", info.memory.available_physical_mb, false);
    w.indentLevel--;
    w.nl();
    out << "},";

    // ---- gpus ----
    w.nl();
    out << "\"gpus\": [";
    w.indentLevel++;
    for (size_t i = 0; i < info.gpus.size(); ++i) {
        const auto& g = info.gpus[i];
        w.nl();
        out << "{";
        w.indentLevel++;
        w.str("name", g.name, true);
        w.str("vendor", toString(g.vendor), true);
        w.str("kind", toString(g.kind), true);
        w.num("dedicated_vram_mb", g.dedicated_vram_mb, true);
        w.num("shared_system_memory_mb", g.shared_system_memory_mb, true);
        w.num("total_accessible_memory_mb", g.total_accessible_memory_mb, false);
        w.indentLevel--;
        w.nl();
        out << "}" << (i + 1 < info.gpus.size() ? "," : "");
    }
    w.indentLevel--;
    w.nl();
    out << "],";

    // ---- vulkan ----
    w.nl();
    out << "\"vulkan\": {";
    w.indentLevel++;
    w.bnum("available", info.vulkan.available, true);
    if (!info.vulkan.available) {
        w.str("unavailable_reason", info.vulkan.unavailable_reason, true);
    }
    {
        std::ostringstream ver;
        ver << info.vulkan.instance_api_version_major << "."
            << info.vulkan.instance_api_version_minor << "."
            << info.vulkan.instance_api_version_patch;
        w.str("instance_api_version", ver.str(), true);
    }
    w.nl();
    out << "\"devices\": [";
    w.indentLevel++;
    for (size_t i = 0; i < info.vulkan.devices.size(); ++i) {
        const auto& d = info.vulkan.devices[i];
        w.nl();
        out << "{";
        w.indentLevel++;
        w.str("name", d.name, true);
        w.num("vendor_id", d.vendor_id, true);
        w.num("device_id", d.device_id, true);
        w.str("device_type", d.device_type, true);
        {
            std::ostringstream ver;
            ver << d.api_version_major << "." << d.api_version_minor << "." << d.api_version_patch;
            w.str("api_version", ver.str(), true);
        }
        w.num("device_local_memory_mb", d.device_local_memory_mb, true);
        w.num("host_visible_memory_mb", d.host_visible_memory_mb, true);
        w.bnum("has_compute_capable_queue", d.has_compute_capable_queue, true);
        w.nl();
        out << "\"queue_families\": [";
        w.indentLevel++;
        for (size_t q = 0; q < d.queue_families.size(); ++q) {
            const auto& qf = d.queue_families[q];
            w.nl();
            out << "{";
            w.indentLevel++;
            w.num("index", qf.index, true);
            w.num("queue_count", qf.queue_count, true);
            w.bnum("graphics", qf.supports_graphics, true);
            w.bnum("compute", qf.supports_compute, true);
            w.bnum("transfer", qf.supports_transfer, false);
            w.indentLevel--;
            w.nl();
            out << "}" << (q + 1 < d.queue_families.size() ? "," : "");
        }
        w.indentLevel--;
        w.nl();
        out << "]";
        w.indentLevel--;
        w.nl();
        out << "}" << (i + 1 < info.vulkan.devices.size() ? "," : "");
    }
    w.indentLevel--;
    w.nl();
    out << "]";
    w.indentLevel--;
    w.nl();
    out << "},";

    // ---- cuda ----
    w.nl();
    out << "\"cuda\": {";
    w.indentLevel++;
    w.bnum("available", info.cuda.available, true);
    if (!info.cuda.available) {
        w.str("unavailable_reason", info.cuda.unavailable_reason, true);
    }
    w.num("driver_version", static_cast<unsigned long long>(info.cuda.driver_version), true);
    w.nl();
    out << "\"devices\": [";
    w.indentLevel++;
    for (size_t i = 0; i < info.cuda.devices.size(); ++i) {
        const auto& d = info.cuda.devices[i];
        w.nl();
        out << "{";
        w.indentLevel++;
        w.num("index", static_cast<unsigned long long>(d.index), true);
        w.str("name", d.name, false);
        w.indentLevel--;
        w.nl();
        out << "}" << (i + 1 < info.cuda.devices.size() ? "," : "");
    }
    w.indentLevel--;
    w.nl();
    out << "]";
    w.indentLevel--;
    w.nl();
    out << "}";

    w.indentLevel--;
    w.nl();
    out << "}";

    return out.str();
}

} // namespace agr
