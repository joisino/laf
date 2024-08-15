import argparse

import matplotlib.pyplot as plt
import torch
import torch.bin
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

from utils import GCN, TFGNN, LaFLoader, calc_acc

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=256)
args = parser.parse_args()

hidden_dim = args.hidden_dim

res_tfgnn = []
res_gcn = []
for n_train in range(2, 21):
    torch.manual_seed(0)
    dataset = Planetoid(root=f"/tmp/{args.dataset}", name=args.dataset, split="random", num_train_per_class=n_train)

    accs_tfgnn = []
    for n_layers in [2, 4, 8, 16]:
        train_loader = LaFLoader(dataset[0], [-1] * n_layers, dataset[0].train_mask, dataset.num_classes, shuffle=True)
        val_loader = LaFLoader(dataset[0], [-1] * n_layers, dataset[0].val_mask, dataset.num_classes, shuffle=False)
        test_loader = LaFLoader(dataset[0], [-1] * n_layers, dataset[0].test_mask, dataset.num_classes, shuffle=False)

        model = TFGNN(n_layers, dataset.num_node_features, hidden_dim, dataset.num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        val_acc = calc_acc(model, val_loader)
        test_acc = calc_acc(model, test_loader)
        accs_tfgnn.append((val_acc, test_acc))
        print(f"Accuracy: {val_acc}")
        for epoch in range(args.epochs):
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
            print(f"Size: {n_train}, Layer: {n_layers}, Epoch: {epoch}, Accuracy: {val_acc}")
    res_tfgnn.append(max(accs_tfgnn, key=lambda x: x[0])[1])

    accs_gcn = []
    for n_layers in [2, 4, 8, 16]:
        train_loader = NeighborLoader(dataset[0], [-1] * n_layers, dataset[0].train_mask, shuffle=True)
        val_loader = NeighborLoader(dataset[0], [-1] * n_layers, dataset[0].val_mask, shuffle=False)
        test_loader = NeighborLoader(dataset[0], [-1] * n_layers, dataset[0].test_mask, shuffle=False)

        model = GCN(n_layers, dataset.num_node_features, hidden_dim, dataset.num_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        val_acc = calc_acc(model, val_loader)
        test_acc = calc_acc(model, test_loader)
        accs_gcn.append((val_acc, test_acc))
        for epoch in range(args.epochs):
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
            print(f"Size: {n_train}, Layer: {n_layers}, Epoch: {epoch}, Accuracy: {val_acc}")

    res_gcn.append(max(accs_gcn, key=lambda x: x[0])[1])

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

plt.savefig(f"{args.dataset}.png")
