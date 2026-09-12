// test_llm_availability.cpp
//
// Verifies that the llama.cpp library is linked and the backend init/free
// cycle works. Does NOT require a model file or GPU.
// Always runs — no skip code.

#include "mini_test.h"
#include "llama.h"   // direct check that llama.h is accessible

int main() {
    std::cout << "== test_llm_availability ==\n";

    // If this compiles and links, llama.cpp is properly integrated.
    std::cout << "  llama.cpp linked: YES\n";

    // Basic backend init/free must not crash.
    llama_backend_init();
    std::cout << "  llama_backend_init(): OK\n";

    llama_backend_free();
    std::cout << "  llama_backend_free(): OK\n";

    // Second init/free cycle (should also be safe)
    llama_backend_init();
    llama_backend_free();
    std::cout << "  Second init/free cycle: OK\n";

    AGR_CHECK(true); // If we got here, all good
    AGR_TEST_MAIN_END();
}
