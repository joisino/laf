from torch_geometric.datasets import Planetoid

from utils import TFGNN, LaFLoader, calc_acc

hidden_dim = 32

datasetname = "Cora"
dataset = Planetoid(root=f"/tmp/{datasetname}", name=datasetname)
test_mask = dataset[0]["test_mask"]
for n_layers in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
    test_loader = LaFLoader(dataset[0], [-1] * n_layers, test_mask, dataset.num_classes, shuffle=False)
    model = TFGNN(n_layers, dataset.num_node_features, hidden_dim, dataset.num_classes)
    acc = calc_acc(model, test_loader)
    print(n_layers, acc)
