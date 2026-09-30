#include "mini_test.h"
#include "model/ResourceResidencyManager.h"

int main() {
    std::cout << "== test_residency_transitions ==\n";
    agr::ResourceResidencyManager mgr;
    uint64_t resId = 42;

    mgr.setResidency(resId, agr::ResourceResidency::DISK);
    AGR_CHECK(mgr.getResidency(resId) == agr::ResourceResidency::DISK);

    // Invalid transition
    bool r = mgr.transition(resId, agr::ResourceResidency::WARM, agr::ResourceResidency::HOT);
    AGR_CHECK(r == false);
    AGR_CHECK(mgr.getResidency(resId) == agr::ResourceResidency::DISK);

    // Valid begin transition
    r = mgr.tryBeginTransition(resId, agr::ResourceResidency::DISK, agr::ResourceResidency::LOADING);
    AGR_CHECK(r == true);
    AGR_CHECK(mgr.getResidency(resId) == agr::ResourceResidency::LOADING);

    // Finish transition
    r = mgr.transition(resId, agr::ResourceResidency::LOADING, agr::ResourceResidency::WARM);
    AGR_CHECK(r == true);
    AGR_CHECK(mgr.getResidency(resId) == agr::ResourceResidency::WARM);

    AGR_TEST_MAIN_END();
}
