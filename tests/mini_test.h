#pragma once
// Tiny assertion-based test helper. Deliberately dependency-free so the
// test suite never needs to fetch a framework -- each test is a small
// standalone executable registered with CTest via add_test().
#include <iostream>
#include <string>

inline int g_mini_test_failures = 0;

#define AGR_CHECK(cond)                                                     \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "FAIL: " << #cond << " at " << __FILE__ << ":"     \
                      << __LINE__ << "\n";                                  \
            ++g_mini_test_failures;                                         \
        } else {                                                            \
            std::cout << "  ok: " << #cond << "\n";                         \
        }                                                                   \
    } while (0)

#define AGR_TEST_MAIN_END()                                                 \
    return g_mini_test_failures == 0 ? 0 : 1;
