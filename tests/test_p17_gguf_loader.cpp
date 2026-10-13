// tests/test_p17_gguf_loader.cpp
// Phase 17: Native GGUF parser tests.
// Exits 77 (SKIP) if the model file is not present.

#include "mini_test.h"
#include "hydroxl/GGUFLoader.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace agr::hydroxl;

static const char* MODEL_PATH = "C:\\HeteroAccel\\Models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";

static const char* modelPath() {
    const char* e = std::getenv("HETEROACCEL_MODEL_PATH");
    return (e && *e) ? e : MODEL_PATH;
}
static bool modelAvailable() { return std::filesystem::exists(modelPath()); }

int main() {
    // bad file always works (no skip needed)
    {   GGUFLoader loader;
        bool ok = loader.open("nonexistent_file_zzz.gguf");
        AGR_CHECK(!ok);
        AGR_CHECK(!loader.lastError().empty());
    }

    // Skip the rest if model is absent
    if (!modelAvailable()) {
        std::cout << "[SKIP] Model not found at " << modelPath() << "\n";
        std::cout << "       Set HETEROACCEL_MODEL_PATH to override.\n";
        return 77;
    }

    // Open succeeds
    {   GGUFLoader loader;
        bool ok = loader.open(modelPath());
        if (!ok) std::cerr << "Open error: " << loader.lastError() << "\n";
        AGR_CHECK(ok);
        AGR_CHECK(loader.isOpen());
    }

    // Architecture metadata
    {   GGUFLoader loader;
        AGR_CHECK(loader.open(modelPath()));
        auto arch = loader.kvString("general.architecture", "");
        std::cout << "Architecture: " << arch << "\n";
        AGR_CHECK(arch.find("qwen") != std::string::npos);
    }

    // Has tensors
    {   GGUFLoader loader;
        AGR_CHECK(loader.open(modelPath()));
        AGR_CHECK(!loader.tensors().empty());
        AGR_CHECK(loader.tensors().size() > 100);
        std::cout << "Tensor count: " << loader.tensors().size() << "\n";
    }

    // token_embd.weight is present and non-zero
    {   GGUFLoader loader;
        AGR_CHECK(loader.open(modelPath()));
        const auto* t = loader.findTensor("token_embd.weight");
        AGR_CHECK(t != nullptr);
        AGR_CHECK(t->n_bytes > 0);
        std::cout << "token_embd type=" << toString(t->type)
                  << " n_bytes=" << t->n_bytes << "\n";
    }

    // Hyperparameters match known Qwen2.5-0.5B values
    {   GGUFLoader loader;
        AGR_CHECK(loader.open(modelPath()));
        uint32_t n_embd   = loader.kvUInt32("qwen2.embedding_length",        0);
        uint32_t n_heads  = loader.kvUInt32("qwen2.attention.head_count",    0);
        uint32_t n_layers = loader.kvUInt32("qwen2.block_count",             0);
        uint32_t n_ff     = loader.kvUInt32("qwen2.feed_forward_length",     0);
        std::cout << "n_embd=" << n_embd << " n_heads=" << n_heads
                  << " n_layers=" << n_layers << " n_ff=" << n_ff << "\n";
        AGR_CHECK(n_embd   == 896);
        AGR_CHECK(n_heads  == 14);
        AGR_CHECK(n_layers == 24);
        AGR_CHECK(n_ff     == 4864);
    }

    // data_offset is non-zero
    {   GGUFLoader loader;
        AGR_CHECK(loader.open(modelPath()));
        AGR_CHECK(loader.dataOffset() > 0);
        std::cout << "Data offset: " << loader.dataOffset() << " bytes\n";
    }

    AGR_TEST_MAIN_END()
}
