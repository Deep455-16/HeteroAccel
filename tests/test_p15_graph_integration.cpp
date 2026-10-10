// tests/test_p15_graph_integration.cpp

#include "execution/ExecutionGraphBuilder.h"
#include <cassert>
#include <iostream>

using namespace agr;

void test_builder_supported_plan() {
    ModelExecutionPlan plan;
    plan.viable = true;
    plan.mode = ExecutionMode::VULKAN;
    plan.residency = ExecutionStrategy::FULL_RESIDENT;
    plan.engine_key = "llama.cpp";

    std::vector<ModelRegion> regions;
    ModelRegion r1;
    r1.identifier = "chunk0";
    regions.push_back(r1);

    ExecutionGraphBuilder builder;
    ExecutionGraph graph;
    std::string err;

    bool ok = builder.build(plan, "dummy.gguf", regions, "prompt", graph, err);
    assert(ok);

    auto nodes = graph.getAllNodeIds();
    // 1 residency node, 1 sync node, 1 compute node -> 3 nodes total
    assert(nodes.size() == 3);

    bool found_res = false, found_sync = false, found_comp = false;
    for (uint64_t id : nodes) {
        auto* n = graph.getNode(id);
        if (n->getType() == ExecutionNodeType::RESIDENCY) {
            assert(n->getResource() == ExecutionResource::VULKAN);
            found_res = true;
        } else if (n->getType() == ExecutionNodeType::SYNCHRONIZATION) {
            found_sync = true;
        } else if (n->getType() == ExecutionNodeType::COMPUTE) {
            assert(n->getResource() == ExecutionResource::VULKAN);
            found_comp = true;
        }
    }
    assert(found_res && found_sync && found_comp);
}

void test_builder_unsupported_plan() {
    ModelExecutionPlan plan;
    plan.viable = false; // Not viable
    
    ExecutionGraphBuilder builder;
    ExecutionGraph graph;
    std::string err;

    bool ok = builder.build(plan, "dummy.gguf", {}, "prompt", graph, err);
    assert(!ok);
    assert(err.find("unsupported") != std::string::npos || err.find("non-viable") != std::string::npos);
}

int main() {
    test_builder_supported_plan();
    test_builder_unsupported_plan();
    std::cout << "test_p15_graph_integration PASSED\n";
    return 0;
}
