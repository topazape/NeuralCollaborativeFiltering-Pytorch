import argparse
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
from neural_collaborative_filtering import NeuMF, Trainer
from neural_collaborative_filtering.data import MFDatasetWithNegativeSampling
from neural_collaborative_filtering.utils import get_device
from torch.utils.data import DataLoader

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="seed for initializing training"
    )

    return parser.parse_args()


def create_dataset(config: dict) -> tuple[Any, Any, Any, Any]:
    header = config["dataset"].get("header")
    if config["dataset"].get("engine") is not None:
        engine = config["dataset"]["engine"]
    else:
        engine = "c"

    dataset = MFDatasetWithNegativeSampling(
        config["dataset"]["filename"],
        sep=config["dataset"]["sep"],
        header=header,
        num_negatives=config["dataset"]["num_negatives"],
        engine=engine,
    )
    tr_set, va_set, test_set = dataset.train_test_split()

    return (tr_set, va_set, test_set, dataset)


if __name__ == "__main__":
    args = make_parser()
    device = get_device()

    config_file = args.config_file
    config_file_dir = str(Path(config_file).resolve().parent)

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    tr_set, va_set, test_set, org_set = create_dataset(config)
    tr_loader = DataLoader(
        tr_set,
        batch_size=config["dataloader"]["train"]["batch_size"],
        shuffle=config["dataloader"]["train"]["shuffle"],
    )
    va_loader = DataLoader(
        va_set,
        # If `num_negative=4`, the `valid_dataset` is sorted such that the score is `[1, 0, 0, 0, 0]`.
        # `batch_size=5` means `POS:NEG = 1:4`. Therefore, this is a per-user evaluation.
        batch_size=100,
        shuffle=False,
    )

    model = NeuMF(
        user_num=org_set.user_num,
        item_num=org_set.item_num,
        gmf_dim=config["model"]["gmf"]["latent_dim"],
        mlp_dim=config["model"]["mlp"]["latent_dim"],
        layer_nums_list=config["model"]["mlp"]["layer_nums_list"],
    )
    print(model)
    torchinfo.summary(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["trainer"]["learning_rate"])

    trainer = Trainer(
        epochs=config["trainer"]["epochs"],
        train_loader=tr_loader,
        valid_loader=va_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=config_file_dir,
    )
    model.to(device)
    trainer.fit(model)
