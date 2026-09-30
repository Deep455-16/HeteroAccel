#include "mini_test.h"
#include "model/LayerManager.h"

int main() {
    std::cout << "== test_layer_dependencies ==\n";
    agr::LayerManager layerMgr;

    auto layer1 = std::make_shared<agr::ModelLayer>();
    layer1->id = 1;
    layer1->model_id = 10;
    layer1->name = "Layer1";

    auto layer2 = std::make_shared<agr::ModelLayer>();
    layer2->id = 2;
    layer2->model_id = 10;
    layer2->name = "Layer2";
    layer2->depends_on_layers.push_back(1);

    auto layer3 = std::make_shared<agr::ModelLayer>();
    layer3->id = 3;
    layer3->model_id = 10;
    layer3->name = "Layer3";
    layer3->depends_on_layers.push_back(2);

    layerMgr.registerLayer(layer1);
    layerMgr.registerLayer(layer2);
    layerMgr.registerLayer(layer3);

    auto succ1 = layerMgr.getSuccessors(1);
    AGR_CHECK(succ1.size() == 1);
    AGR_CHECK(succ1[0]->id == 2);

    auto succ2 = layerMgr.getSuccessors(2);
    AGR_CHECK(succ2.size() == 1);
    AGR_CHECK(succ2[0]->id == 3);

    AGR_TEST_MAIN_END();
}
