import argparse
import random
from pathlib import Path

import torch
from neural_collaborative_filtering import NeuMF, Trainer
from neural_collaborative_filtering.utils import get_device
from neural_collaborative_filtering.data import MFDatasetWithNegativeSampling
from torch.utils.data import DataLoader, Dataset, random_split

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


def create_dataset(config: dict) -> tuple[Dataset, Dataset, Dataset, Dataset]:
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
    tr_set, test_set = dataset.train_test_split()
    tr_set, va_set = random_split(tr_set, lengths=[0.8, 0.2])

    return (tr_set, va_set, test_set, dataset)


if __name__ == "__main__":
    args = make_parser()
    device = get_device()

    config_file = args.config_file
    config_file_dir = str(Path(config_file).resolve().parent)

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    tr_set, va_set, test_set, org_set = create_dataset(config)

    model = NeuMF(user_num=org_set.user_num, item_num=org_set.user_num, gmf_dim=config["model"]["gmf"]["latent_dim"], mlp_dim=config["model"]["gmf"]["latent_dim"])
