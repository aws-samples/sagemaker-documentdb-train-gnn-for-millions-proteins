# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functions and classes for the dataset and data loader
"""

import math
import random
from typing import Tuple, Iterator

from pymongo import MongoClient
import dgl
import torch
import torch.utils.data as data
import torch.nn.functional as F

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


def collate_protein_graphs(samples: list) -> Tuple[dgl.DGLGraph, torch.Tensor]:
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


def match_by_split(split: str) -> dict:
    """Get the $match query by split one of ['train', 'valid', 'test']."""
    return {"$and": [{"is_AF": {"$exists": True}}, {"split": split}]}
