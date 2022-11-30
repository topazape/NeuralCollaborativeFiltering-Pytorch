from typing import Any, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .base_mf_dataset import BaseMFDataset


class MFDatasetWithNegativeSampling(BaseMFDataset):
    def __init__(
        self,
        filename: str,
        sep: str,
        header: Optional[int],
        num_negatives: int,
        engine: Any = "c",
    ) -> None:
        super().__init__(filename=filename, sep=sep, header=header, engine=engine)

        self.data["score"] = self.data["score"].astype(bool).astype(float)

        if isinstance(num_negatives, int):
            if num_negatives > 0:
                self.num_negatives = num_negatives
                self._create_data_with_negatives()
            else:
                raise ValueError
        else:
            raise TypeError

    def _create_data_with_negatives(self) -> None:
        all_items = np.array(list(self.item_indices.values()))

        negatives = (
            self.data.groupby("user_index")["item_index"].apply(np.unique).reset_index()
        )
        negatives["negative_items"] = negatives["item_index"].apply(
            lambda x: np.random.choice(np.setdiff1d(all_items, x), 99)
        )
        negatives = negatives.drop(["item_index"], axis=1)

        self.data = self.data.merge(negatives, on="user_index")

    def train_test_split(
        self,
    ) -> tuple["CreateDataset", "CreateDataset", "CreateDataset"]:
        self.data["rank"] = self.data.groupby("user_index")["ts"].rank(
            method="first", ascending=False
        )
        train = self.data[self.data["rank"] > 2].copy()
        train["negative_samples"] = train["negative_items"].apply(
            lambda x: np.random.choice(x, self.num_negatives)
        )
        valid = self.data[self.data["rank"] == 2].copy()
        valid = valid.rename(columns={"negative_items": "negative_samples"})
        test = self.data[self.data["rank"] == 1].copy()
        test = test.rename(columns={"negative_items": "negative_samples"})

        train = self.__explode_negative_items(train)
        valid = self.__explode_negative_items(valid)
        valid = valid.sort_values(by=["user_index", "score"], ascending=[True, False])
        test = self.__explode_negative_items(test)
        test = test.sort_values(by=["user_index", "score"], ascending=[True, False])

        train_dataset = CreateDataset(train)
        valid_dataset = CreateDataset(valid)
        test_dataset = CreateDataset(test)

        return (train_dataset, valid_dataset, test_dataset)

    def __explode_negative_items(self, df: pd.DataFrame) -> pd.DataFrame:
        pos = df[["user_index", "item_index", "rank", "score"]]

        neg = df[["user_index", "negative_samples", "rank"]]
        neg = neg.explode("negative_samples").rename(
            columns={"negative_samples": "item_index"}
        )
        neg["score"] = 0.0

        ret = pd.concat([pos, neg], axis=0).reset_index(drop=True)
        return ret


class CreateDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[int, int, float]:
        data = self.data.iloc[idx]
        indices = data[["user_index", "item_index"]].to_numpy()
        user_idx, movie_idx = indices[0], indices[1]
        score = data["score"]

        return (user_idx, movie_idx, score)
