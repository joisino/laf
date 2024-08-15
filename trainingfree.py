import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.loader import NeighborLoader

from utils import GAT, GCN, TFGNN, LaFLoader, calc_acc

n_layers = 3
hidden_dim = 32

for datasetname in ["Cora", "CiteSeer", "PubMed", "CS", "Physics", "Computers", "Photo"]:
    if datasetname in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=f"/tmp/{datasetname}", name=datasetname)
    elif datasetname in ["CS", "Physics"]:
        dataset = Coauthor(root=f"/tmp/{datasetname}", name=datasetname)
    elif datasetname in ["Computers", "Photo"]:
        dataset = Amazon(root=f"/tmp/{datasetname}", name=datasetname)

    data = dataset[0]

    if "train_mask" not in dataset[0]:
        n_train_per_class = 20
        n_val_per_class = 30
        n_classes = dataset[0]["y"].max() + 1
        train_mask = []
        val_mask = []
        test_mask = []
        for c in range(n_classes):
            all_c = np.where(dataset[0]["y"].numpy() == c)[0]
            if len(all_c) <= n_train_per_class + n_val_per_class:
                continue
            train, rest = train_test_split(all_c, train_size=n_train_per_class, random_state=0)
            val, test = train_test_split(rest, train_size=n_val_per_class, random_state=0)
            train_mask += train.tolist()
            val_mask += val.tolist()
            test_mask += test.tolist()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).scatter_(0, torch.tensor(train_mask), True)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool).scatter_(0, torch.tensor(val_mask), True)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).scatter_(0, torch.tensor(test_mask), True)

    for loader in ["NeighborLoader", "LaFLoader"]:
        if loader == "NeighborLoader":
            test_loader = NeighborLoader(data, [-1] * n_layers, data.test_mask, shuffle=False)
        elif loader == "LaFLoader":
            test_loader = LaFLoader(data, [-1] * n_layers, data.test_mask, dataset.num_classes, shuffle=False)
        for modelname in ["GCN", "GAT", "NTFGCNN", "TFGCNN"]:
            torch.manual_seed(0)
            if loader == "NeighborLoader" and modelname in ["TFGCNN", "NTFGCNN"]:
                continue
            if modelname == "TFGCNN":
                model = TFGNN(n_layers, dataset.num_node_features, hidden_dim, dataset.num_classes)
            elif modelname == "NTFGCNN":
                model = TFGNN(n_layers, dataset.num_node_features, hidden_dim, dataset.num_classes, False)
            elif modelname == "GCN":
                input_dim_GCN = dataset.num_node_features
                if loader == "LaFLoader":
                    input_dim_GCN = dataset.num_node_features + dataset.num_classes + 1
                model = GCN(n_layers, input_dim_GCN, hidden_dim, dataset.num_classes)
            elif modelname == "GAT":
                input_dim_GAT = dataset.num_node_features
                if loader == "LaFLoader":
                    input_dim_GAT = dataset.num_node_features + dataset.num_classes + 1
                model = GAT(n_layers, input_dim_GAT, hidden_dim, dataset.num_classes)

            acc = calc_acc(model, test_loader)
            print(datasetname, loader, modelname, acc)
