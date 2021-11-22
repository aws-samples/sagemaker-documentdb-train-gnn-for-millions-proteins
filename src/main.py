import argparse
import json
import os
import math
import random
from datetime import datetime
from typing import Optional
from collections.abc import Iterator

from pymongo import MongoClient
import dgl
from dgl.nn import GraphConv, GlobalAttentionPooling
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# amino acid to index mapping
# from Bio.PDB.Polypeptide import d1_to_index
d1_to_index = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}


def setup(args: dict, seed: int = 0) -> dict:
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return args


def collate_protein_graphs(samples: list) -> tuple[dgl.DGLGraph, torch.Tensor]:
    """Batching a list of datapoints for dataloader."""
    graphs, targets = map(list, zip(*samples))
    bg = dgl.batch(graphs)
    return bg, torch.tensor(targets).unsqueeze(1).to(torch.float32)


def convert_to_graph(protein: dict, k: int = 3) -> dgl.DGLGraph:
    """
    Convert a protein (dict) to a dgl graph using kNN.
    """
    coords = torch.tensor(protein["coords"])
    X_ca = coords[:, 1]
    # construct knn graph from C-alpha coordinates
    g = dgl.knn_graph(X_ca, k=k)
    seq = protein["seq"]
    node_features = torch.tensor([d1_to_index[residue] for residue in seq])
    node_features = F.one_hot(node_features, num_classes=len(d1_to_index)).to(
        dtype=torch.float
    )

    # add node features
    g.ndata["h"] = node_features
    return g


class ProteinDataset(data.IterableDataset):
    """
    An iterable-style dataset for proteins in DocumentDB
    Args:
        pipeline: an aggregation pipeline to retrieve data from DocumentDB
        db_uri: URI of the DocumentDB
        db_name: name of the database
        collection_name: name of the collection
        k: k used for kNN when creating a graph from atomic coordinates
    """

    def __init__(
        self,
        pipeline: list,
        db_uri: str = "",
        db_name: str = "",
        collection_name: str = "",
        k: int = 3,
    ) -> None:

        self.db_uri = db_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.k = k

        client = MongoClient(self.db_uri, connect=False)
        collection = client[self.db_name][self.collection_name]
        # pre-fetch the metadata as docs from DocumentDB
        self.docs = [doc for doc in collection.aggregate(pipeline)]
        # mapping document '_id' to label
        self.labels = {doc["_id"]: doc["y"] for doc in self.docs}

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading, return the full iterator
            protein_ids = [doc["_id"] for doc in self.docs]

        else:  # in a worker process
            # split workload
            start = 0
            end = len(self.docs)
            per_worker = int(
                math.ceil((end - start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)

            protein_ids = [
                doc["_id"] for doc in self.docs[iter_start:iter_end]
            ]

        # retrieve a list of proteins by _id from DocDB
        with MongoClient(self.db_uri) as client:
            collection = client[self.db_name][self.collection_name]
            cur = collection.find(
                {"_id": {"$in": protein_ids}},
                projection={"coords": True, "seq": True},
            )
            return (
                (
                    convert_to_graph(protein, k=self.k),
                    self.labels[protein["_id"]],
                )
                for protein in cur
            )

    def __len__(self) -> int:
        return len(self.docs)


class BufferedShuffleDataset(data.IterableDataset):
    """Dataset shuffled from the original dataset.
    This class is useful to shuffle an existing instance of an IterableDataset.
    """

    dataset: data.IterableDataset
    buffer_size: int

    def __init__(
        self, dataset: data.IterableDataset, buffer_size: int
    ) -> None:
        super(BufferedShuffleDataset, self).__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator:
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()


class EarlyStopper(object):
    def __init__(self, patience: int, filename: Optional[str] = None) -> None:
        if filename is None:
            # Name checkpoint based on time
            dt = datetime.now()
            filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
                dt.date(), dt.hour, dt.minute, dt.second
            )
            filename = os.path.join("/opt/ml/model", filename)

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def save_checkpoint(self, model: nn.Module) -> None:
        """Saves model when the metric on the validation set gets improved."""
        torch.save({"model_state_dict": model.state_dict()}, self.filename)

    def load_checkpoint(self, model: nn.Module) -> nn.Module:
        """Load model saved with early stopping."""
        model.load_state_dict(torch.load(self.filename)["model_state_dict"])

    def step(self, score: float, model: nn.Module) -> bool:
        if (self.best_score is None) or (score > self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                "EarlyStopping counter: {:d} out of {:d}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""

    def __init__(self) -> None:
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())

    def roc_auc_score(self) -> list:
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_y_true = y_true[:, task].numpy()
            task_y_pred = y_pred[:, task].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores


class GCN(nn.Module):
    """A two layer Graph Conv net with Global Attention Pooling over the
    nodes.
    Args:
        in_feats: int, dim of input node features
        h_feats: int, dim of hidden layers
        num_classes: int, number of output units
    """

    def __init__(self, in_feats: int, h_feats: int, num_classes: int) -> None:
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        # the gate layer that maps node feature to outputs
        self.gate_nn = nn.Linear(h_feats, num_classes)
        self.gap = GlobalAttentionPooling(self.gate_nn)
        # the output layer making predictions
        self.output = nn.Linear(h_feats, num_classes)

    def _conv_forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """forward pass through the GraphConv layers"""
        in_feat = g.ndata["h"]
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        return h

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        h = self._conv_forward(g)
        h = self.gap(g, h)
        return self.output(h)

    def attention_scores(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Calculate attention scores"""
        h = self._conv_forward(g)
        with g.local_scope():
            gate = self.gap.gate_nn(h)
            g.ndata["gate"] = gate
            gate = dgl.softmax_nodes(g, "gate")
            g.ndata.pop("gate")
            return gate


def run_a_train_epoch(
    args: dict,
    epoch: int,
    model: nn.Module,
    data_loader: data.DataLoader,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        bg, labels = batch_data
        bg = bg.to(args["device"])
        labels = labels.to(args["device"])
        logits = model(bg)
        # Mask non-existing labels
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "epoch {:d}/{:d}, batch {:d}, loss {:.4f}".format(
                epoch + 1,
                args["n_epochs"],
                batch_id + 1,
                loss.item(),
            )
        )
        train_meter.update(logits, labels)
    train_score = np.mean(train_meter.roc_auc_score())
    print(
        "epoch {:d}/{:d}, training roc-auc {:.4f}".format(
            epoch + 1, args["n_epochs"], train_score
        )
    )


def run_an_eval_epoch(
    args: dict, model: nn.Module, data_loader: data.DataLoader
) -> None:
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_data in data_loader:
            bg, labels = batch_data
            bg = bg.to(args["device"])
            labels = labels.to(args["device"])
            logits = model(bg)
            eval_meter.update(logits, labels)
    return np.mean(eval_meter.roc_auc_score())


def load_sagemaker_config(args: dict) -> dict:
    file_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            new_args = json.load(f)
            for k, v in new_args.items():
                if k not in args:
                    continue
                if isinstance(args[k], int):
                    v = int(v)
                if isinstance(args[k], float):
                    v = float(v)
                args[k] = v
    return args


def match_by_split(split: str) -> dict:
    """Get the $match query by split one of ['train', 'valid', 'test']."""
    return {"$and": [{"is_AF": {"$exists": True}}, {"split": split}]}


def main(args: dict) -> None:
    args = setup(args)
    uri = "mongodb://{}:{}@{}:{}/?tls=true&tlsCAFile=rds-combined-ca-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false".format(
        args["db_username"],
        args["db_password"],
        args["db_host"],
        args["db_port"],
    )

    project = {"y": "$is_AF"}

    datasets = [
        ProteinDataset(
            [
                {"$match": match_by_split(split)},
                {"$project": project},
            ],
            db_uri=uri,
            db_name="proteins",
            collection_name="proteins",
            k=args["knn"],
        )
        for split in ("train", "valid", "test")
    ]
    train_loader = data.DataLoader(
        BufferedShuffleDataset(datasets[0], buffer_size=256),
        batch_size=args["batch_size"],
        collate_fn=collate_protein_graphs,
        num_workers=8,
    )

    valid_loader = data.DataLoader(
        datasets[1],
        batch_size=args["batch_size"],
        collate_fn=collate_protein_graphs,
    )
    test_loader = data.DataLoader(
        datasets[2],
        batch_size=args["batch_size"],
        collate_fn=collate_protein_graphs,
    )

    # Create the model with given dimensions
    dim_nfeats = len(d1_to_index)
    n_classes = 1
    model = GCN(dim_nfeats, 16, n_classes).to(args["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    stopper = EarlyStopper(args["patience"])

    for epoch in range(args["n_epochs"]):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, valid_loader)
        early_stop = stopper.step(val_score, model)
        print(
            "epoch {:d}/{:d}, validation roc-auc {:.4f}, ".format(
                epoch + 1, args["n_epochs"], val_score
            )
        )
        print("best validation roc-auc {:.4f}".format(stopper.best_score))
        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch(args, model, test_loader)
    print("Best validation score {:.4f}".format(stopper.best_score))
    print("Test score {:.4f}".format(test_score))


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="GCN for Tox21")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of graphs (molecules) per batch",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=3,
        help="k used in kNN when creating protein graphs",
    )
    parser.add_argument(
        "--db-host",
        type=str,
        help="Host of DocumentDB",
    )
    parser.add_argument(
        "--db-port",
        type=str,
        help="Port of DocumentDB",
    )
    parser.add_argument(
        "--db-username",
        type=str,
        help="Username of DocumentDB",
    )
    parser.add_argument(
        "--db-password",
        type=str,
        help="Password of DocumentDB",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs to wait before early stop",
    )
    return parser.parse_args().__dict__


if __name__ == "__main__":
    args = parse_args()
    print(args)
    args = load_sagemaker_config(args)
    main(args)
