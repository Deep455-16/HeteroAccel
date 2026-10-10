// tests/test_p15_graph_validation.cpp

#include "execution/ExecutionGraph.h"
#include <cassert>
#include <iostream>
#include <string>

using namespace agr;

class DummyNode : public IExecutionNode {
public:
    DummyNode(uint64_t id) : id_(id) {}
    uint64_t getId() const override { return id_; }
    std::string getName() const override { return "Dummy" + std::to_string(id_); }
    ExecutionNodeType getType() const override { return ExecutionNodeType::CUSTOM; }
    ExecutionResource getResource() const override { return ExecutionResource::NONE; }
    GraphNodeResult execute(ExecutionGraphContext&) override { return {true, ""}; }
private:
    uint64_t id_;
};

void test_empty_graph() {
    ExecutionGraph g;
    std::string err;
    assert(g.validate(err)); // empty graph is technically valid DAG
}

void test_single_node() {
    ExecutionGraph g;
    g.addNode(std::make_unique<DummyNode>(1));
    std::string err;
    assert(g.validate(err));
}

void test_multiple_nodes() {
    ExecutionGraph g;
    g.addNode(std::make_unique<DummyNode>(1));
    g.addNode(std::make_unique<DummyNode>(2));
    g.addNode(std::make_unique<DummyNode>(3));
    g.addDependency(1, 2);
    g.addDependency(2, 3);
    std::string err;
    assert(g.validate(err));
    assert(g.getSuccessors(1).size() == 1);
    assert(g.getPredecessors(3).size() == 1);
}

void test_self_cycle() {
    ExecutionGraph g;
    g.addNode(std::make_unique<DummyNode>(1));
    g.addDependency(1, 1);
    std::string err;
    assert(!g.validate(err));
}

void test_two_node_cycle() {
    ExecutionGraph g;
    g.addNode(std::make_unique<DummyNode>(1));
    g.addNode(std::make_unique<DummyNode>(2));
    g.addDependency(1, 2);
    g.addDependency(2, 1);
    std::string err;
    assert(!g.validate(err));
}

void test_multi_node_cycle() {
    ExecutionGraph g;
    g.addNode(std::make_unique<DummyNode>(1));
    g.addNode(std::make_unique<DummyNode>(2));
    g.addNode(std::make_unique<DummyNode>(3));
    g.addDependency(1, 2);
    g.addDependency(2, 3);
    g.addDependency(3, 1);
    std::string err;
    assert(!g.validate(err));
}

void test_disconnected_graph() {
    ExecutionGraph g;
    g.addNode(std::make_unique<DummyNode>(1));
    g.addNode(std::make_unique<DummyNode>(2));
    g.addNode(std::make_unique<DummyNode>(3));
    g.addDependency(1, 2);
    // Node 3 is disconnected
    std::string err;
    assert(g.validate(err));
}

int main() {
    test_empty_graph();
    test_single_node();
    test_multiple_nodes();
    test_self_cycle();
    test_two_node_cycle();
    test_multi_node_cycle();
    test_disconnected_graph();
    std::cout << "test_p15_graph_validation PASSED\n";
    return 0;
}
