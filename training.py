import matplotlib.pyplot as plt
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

dataset = Planetoid(root="/tmp/Cora", name="Cora")

res_tfgnn = []
res_gcn = []
for seed in range(3):
    torch.manual_seed(seed)

    train_loader = LaFLoader(dataset[0], [-1] * n_layers, dataset[0].train_mask, dataset.num_classes, shuffle=True)
    val_loader = LaFLoader(dataset[0], [-1] * n_layers, dataset[0].val_mask, dataset.num_classes, shuffle=False)

    model = TFGNN(n_layers, dataset.num_node_features, hidden_dim, dataset.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    acc = calc_acc(model, val_loader)
    print(f"Initial Accuracy: {acc}")
    accs_tfgnn = [acc]
    it = 0
    for epoch in range(epochs):
        for data in train_loader:
            it += 1
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[0], data.y[0])
            loss.backward()
            optimizer.step()

            if it % 100 == 0:
                acc = calc_acc(model, val_loader)
                accs_tfgnn.append(acc)
                print(f"Iter: {it}, Accuracy: {acc}")

    res_tfgnn.append(accs_tfgnn)

    torch.manual_seed(seed)

    train_loader = NeighborLoader(dataset[0], [-1] * n_layers, dataset[0].train_mask, shuffle=True)
    val_loader = NeighborLoader(dataset[0], [-1] * n_layers, dataset[0].val_mask, shuffle=False)

    model = GCN(n_layers, dataset.num_node_features, hidden_dim, dataset.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    acc = calc_acc(model, val_loader)
    print(f"Initial Accuracy: {acc}")
    accs_gcn = [acc]
    it = 0
    for epoch in range(epochs):
        for data in train_loader:
            it += 1
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[0], data.y[0])
            loss.backward()
            optimizer.step()

            if it % 100 == 0:
                acc = calc_acc(model, val_loader)
                accs_gcn.append(acc)
                print(f"Iter: {it}, Accuracy: {acc}")

    res_gcn.append(accs_gcn)

print(res_tfgnn)
print(res_gcn)

plt.plot(torch.tensor(res_tfgnn).mean(dim=0), label="TFGCNN")
plt.fill_between(
    range(len(res_tfgnn[0])),
    torch.tensor(res_tfgnn).mean(dim=0) - torch.tensor(res_tfgnn).std(dim=0),
    torch.tensor(res_tfgnn).mean(dim=0) + torch.tensor(res_tfgnn).std(dim=0),
    alpha=0.3,
)

plt.plot(torch.tensor(res_gcn).mean(dim=0), label="GCN")
plt.fill_between(
    range(len(res_gcn[0])),
    torch.tensor(res_gcn).mean(dim=0) - torch.tensor(res_gcn).std(dim=0),
    torch.tensor(res_gcn).mean(dim=0) + torch.tensor(res_gcn).std(dim=0),
    alpha=0.3,
)

plt.savefig("training.png")
