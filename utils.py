import torch
import torch.bin
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, GCNConv, MessagePassing


class TFConv(MessagePassing):
    def __init__(self, in_d, out_d, n_classes, tfinit=True):
        super().__init__(aggr="mean")
        self.in_d = in_d
        self.out_d = out_d
        self.n_classes = n_classes
        self.tfinit = tfinit
        l_in_d = in_d + n_classes + 1
        l_out_d = out_d + n_classes + 1
        self.VL = torch.nn.Linear(l_in_d, l_out_d)
        self.WL = torch.nn.Linear(l_in_d, l_out_d)
        self.VU = torch.nn.Linear(l_in_d, l_out_d)
        self.WU = torch.nn.Linear(l_in_d, l_out_d)

        if self.tfinit:
            label_ind_in = torch.arange(self.in_d, self.in_d + self.n_classes + 1)
            label_ind_out = torch.arange(self.out_d, self.out_d + self.n_classes + 1)
            with torch.no_grad():
                self.VL.weight[label_ind_out, :] = 0
                self.VL.weight[label_ind_out, label_ind_in] = 1
                self.VL.bias[label_ind_out] = 0
                self.WL.weight[label_ind_out, :] = 0
                self.WL.bias[label_ind_out] = 0
                self.VU.weight[label_ind_out, :] = 0
                self.VU.bias[label_ind_out] = 0
                self.WU.weight[label_ind_out, :] = 0
                self.WU.weight[label_ind_out, label_ind_in] = 1
                self.WU.bias[label_ind_out] = 0

    def forward(self, x, edge_index, labelmask):
        propagated_x = self.propagate(edge_index, x=x)

        res = labelmask[:, None] * (self.VL(x) + self.WL(propagated_x))
        res += (1 - labelmask.int())[:, None] * (self.VU(x) + self.WU(propagated_x))
        return res


class TFGNN(torch.nn.Module):
    def __init__(self, n_layers, in_d, mid_d, n_classes, tfinit=True):
        super().__init__()
        self.n_layers = n_layers
        self.mid_d = mid_d
        self.n_classes = n_classes
        self.tfinit = tfinit
        for i in range(n_layers):
            if i == 0:
                setattr(self, f"conv{i}", TFConv(in_d, mid_d, n_classes, tfinit))
            else:
                setattr(self, f"conv{i}", TFConv(mid_d, mid_d, n_classes, tfinit))
        self.classifier = torch.nn.Linear(mid_d + n_classes + 1, n_classes)

        if self.tfinit:
            with torch.no_grad():
                self.classifier.bias[:] = 0
                self.classifier.weight[:, :] = 0
                for i in range(self.n_classes):
                    self.classifier.weight[i, self.mid_d + 1 + i] = 1

    def forward(self, data):
        x, edge_index, labelmask = data.x, data.edge_index, data.labelmask

        for i in range(self.n_layers):
            x = getattr(self, f"conv{i}")(x, edge_index, labelmask)
            x = F.relu(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, n_layers, in_d, mid_d, n_classes):
        super().__init__()
        self.n_layers = n_layers
        self.mid_d = mid_d
        self.n_classes = n_classes
        for i in range(n_layers):
            if i == 0:
                setattr(self, f"conv{i}", GCNConv(in_d, mid_d))
            else:
                setattr(self, f"conv{i}", GCNConv(mid_d, mid_d))
        self.classifier = torch.nn.Linear(mid_d, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.n_layers):
            x = getattr(self, f"conv{i}")(x, edge_index)
            x = F.relu(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, n_layers, in_d, mid_d, n_classes):
        super().__init__()
        self.n_layers = n_layers
        self.mid_d = mid_d
        self.n_classes = n_classes
        for i in range(n_layers):
            if i == 0:
                setattr(self, f"conv{i}", GATConv(in_d, mid_d))
            else:
                setattr(self, f"conv{i}", GATConv(mid_d, mid_d))
        self.classifier = torch.nn.Linear(mid_d, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.n_layers):
            x = getattr(self, f"conv{i}")(x, edge_index)
            x = F.relu(x)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class LaFLoader(NeighborLoader):
    def __init__(self, data, num_neighbors, input_nodes, num_classes, shuffle=None):
        super().__init__(data, num_neighbors, input_nodes, shuffle=shuffle)
        self.num_classes = num_classes

    def filter_fn(self, out):
        data = super().filter_fn(out)
        data.labelmask = data.train_mask.clone().detach()
        data.labelmask[data.n_id == data.input_id] = 0
        label_features = torch.eye(self.num_classes)[data.y] * data.labelmask[:, None]
        data.x = torch.cat([data.x, data.labelmask[:, None], label_features], dim=-1)
        return data


def calc_acc(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        pred = model(data).argmax(dim=1)
        correct += (pred[0] == data.y[0]).sum().item()
    return correct / len(loader)
