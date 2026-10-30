// examples/cpp_example.cpp
// Phase 23: Minimal C++ SDK Consumer Example

#include <heteroaccel/HeteroAccel.hpp>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "=== HeteroAccel C++ SDK Example (v" << ha::Runtime::version() << ") ===\n";

    try {
        // 1. Initialize Runtime
        ha::Runtime rt;

        // 2. Query available backends
        auto backends = rt.backends();
        std::cout << "Registered backends: " << backends.size() << "\n";
        for (size_t i = 0; i < backends.size(); ++i) {
            std::cout << "  [" << i << "] " << backends[i].backend_id
                      << " (Available: " << (backends[i].available ? "YES" : "NO")
                      << ", Vulkan: " << (backends[i].requires_vulkan ? "YES" : "NO")
                      << ", CUDA: " << (backends[i].requires_cuda ? "YES" : "NO") << ")\n";
        }

        // 3. Load model if path is provided via env
        const char* model_path = std::getenv("HETEROACCEL_MODEL_PATH");
        if (!model_path) {
            std::cout << "HETEROACCEL_MODEL_PATH not set. Skipping model inference.\n";
            return 0;
        }

        std::cout << "Loading model: " << model_path << "\n";
        ha::Model model = rt.load_model(model_path);

        std::cout << "Model loaded:\n"
                  << "  Architecture:   " << model.arch() << "\n"
                  << "  Context Length: " << model.context_length() << "\n"
                  << "  Vocab Size:     " << model.vocab_size() << "\n";

        // 4. Create inference session
        ha::Session session = model.create_session();

        // 5. Run inference
        ha::InferParams params;
        params.prompt = "The capital of France is";
        params.max_new_tokens = 8;
        params.temperature = 0.0f;

        std::cout << "Running synchronous inference...\n";
        ha::Result res = session.infer(params);

        if (!res.ok()) {
            std::cerr << "Inference error: " << res.error_message << "\n";
            return 1;
        }

        std::cout << "Output: '" << res.output_text << "'\n"
                  << "Tokens generated: " << res.n_output_tokens
                  << " (" << res.n_prompt_tokens << " prompt tokens)\n"
                  << "Throughput:       " << res.tokens_per_second << " tok/s\n"
                  << "Backend:          " << res.backend_used
                  << " (GPU: " << (res.gpu_used ? "YES" : "NO") << ")\n";

        // 6. Test streaming simulation
        std::cout << "Streaming tokens: ";
        session.infer_stream(params, [](const std::string& token_text, int token_id) {
            (void)token_id;
            std::cout << "[" << token_text << "] " << std::flush;
            return true; // continue
        });
        std::cout << "\n";

        std::cout << "Completed successfully. Resources cleaned up by RAII.\n";
        return 0;
    } catch (const ha::Exception& e) {
        std::cerr << "HeteroAccel exception: " << e.what()
                  << " (code: " << ha_error_string(e.code()) << ")\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << "\n";
        return 1;
    }
}
