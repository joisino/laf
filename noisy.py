import torch
import torch.bin
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

from utils import GCN, TFGNN, LaFLoader, calc_acc

n_layers = 3
hidden_dim = 32
epochs = 32
lr = 0.0001
weight_decay = 0.01

res_tfgnn = []
res_gcn = []
for seed in range(3):
    for noise in [1e-2 * 10**i for i in range(8)]:
        torch.manual_seed(seed)

        dataset = Planetoid(root="/tmp/cora", name="Cora")
        dataset[0].x += torch.randn_like(dataset[0].x) * noise

        accs_tfgnn = []
        train_loader = LaFLoader(dataset[0], [-1] * n_layers, dataset[0].train_mask, dataset.num_classes, shuffle=True)
        val_loader = LaFLoader(dataset[0], [-1] * n_layers, dataset[0].val_mask, dataset.num_classes, shuffle=False)
        test_loader = LaFLoader(dataset[0], [-1] * n_layers, dataset[0].test_mask, dataset.num_classes, shuffle=False)

        model = TFGNN(n_layers, dataset.num_node_features, hidden_dim, dataset.num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        val_acc = calc_acc(model, val_loader)
        test_acc = calc_acc(model, test_loader)
        accs_tfgnn.append((val_acc, test_acc))
        print(f"Initial Accuracy: {val_acc}")
        for epoch in range(epochs):
            for data in train_loader:
                model.train()
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out[0], data.y[0])
                loss.backward()
                optimizer.step()

            val_acc = calc_acc(model, val_loader)
            test_acc = calc_acc(model, test_loader)
            accs_tfgnn.append((val_acc, test_acc))
            print(f"Epoch: {epoch}, Accuracy: {val_acc}")
        res_tfgnn.append(max(accs_tfgnn, key=lambda x: x[0])[1])  # model selection by validation accuracy
        print("res_tfgnn", res_tfgnn)

        torch.manual_seed(seed)

        accs_gcn = []
        train_loader = NeighborLoader(dataset[0], [-1] * n_layers, dataset[0].train_mask, shuffle=True)
        val_loader = NeighborLoader(dataset[0], [-1] * n_layers, dataset[0].val_mask, shuffle=False)
        test_loader = NeighborLoader(dataset[0], [-1] * n_layers, dataset[0].test_mask, shuffle=False)

        model = GCN(n_layers, dataset.num_node_features, hidden_dim, dataset.num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        val_acc = calc_acc(model, val_loader)
        test_acc = calc_acc(model, test_loader)
        accs_gcn.append((val_acc, test_acc))
        print(f"Initial Accuracy: {val_acc}")
        for epoch in range(epochs):
            for data in train_loader:
                model.train()
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out[0], data.y[0])
                loss.backward()
                optimizer.step()

            val_acc = calc_acc(model, val_loader)
            test_acc = calc_acc(model, test_loader)
            accs_gcn.append((val_acc, test_acc))
            print(f"Epoch: {epoch}, Accuracy: {val_acc}")
        res_gcn.append(max(accs_gcn, key=lambda x: x[0])[1])  # model selection by validation accuracy
        print("res_gcn", res_gcn)
