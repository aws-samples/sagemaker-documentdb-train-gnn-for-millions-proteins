import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

# custom imports
from model import GCN
from dataset import (
    d1_to_index,
    collate_protein_graphs,
    ProteinDataset,
    BufferedShuffleDataset,
    match_by_split,
)
from helpers import setup, Meter, EarlyStopper, load_sagemaker_config
import config


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


def main(args: dict) -> None:
    args = setup(args, seed=config.seed)
    uri = "mongodb://{}:{}@{}:{}/?tls=true&tlsCAFile=rds-combined-ca-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false".format(
        args["db_username"],
        args["db_password"],
        args["db_host"],
        args["db_port"],
    )

    datasets = [
        ProteinDataset(
            [
                {"$match": match_by_split(split)},
                {"$project": config.projection},
            ],
            db_uri=uri,
            db_name="proteins",
            collection_name="proteins",
            k=args["knn"],
        )
        for split in ("train", "valid", "test")
    ]
    train_loader = data.DataLoader(
        BufferedShuffleDataset(datasets[0], buffer_size=config.buffer_size),
        batch_size=args["batch_size"],
        collate_fn=collate_protein_graphs,
        num_workers=config.num_workers,
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
    model = GCN(dim_nfeats, config.h_feats, config.n_classes).to(
        args["device"]
    )

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
    parser = argparse.ArgumentParser(
        description="GCN for proteins in DocumentDB"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.batch_size,
        help="Number of graphs (molecules) per batch",
    )
    parser.add_argument(
        "--lr", type=float, default=config.lr, help="Learning rate"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=config.n_epochs,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=config.knn,
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
        default=config.patience,
        help="Number of epochs to wait before early stop",
    )
    return parser.parse_args().__dict__


if __name__ == "__main__":
    args = parse_args()
    print(args)
    args = load_sagemaker_config(args)
    main(args)
