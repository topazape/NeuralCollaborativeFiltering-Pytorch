from pathlib import Path
from typing import Any, Optional

import pandas as pd


class BaseMFDataset:
    def __init__(
        self,
        filename: str,
        sep: str,
        header: Optional[int],
        engine: Any = "c",
    ) -> None:
        filepath = Path(filename)

        if filepath.exists():
            self.data = pd.read_csv(
                filepath,
                sep=sep,
                header=header,
                engine=engine,
                names=["userId", "itemId", "score", "ts"],
            )
        else:
            raise FileNotFoundError

        self.__add_indicies()

    def __add_indicies(self) -> None:
        self.user_indices = {
            userid: idx for idx, userid in enumerate(self.data["userId"].unique())
        }
        self.item_indices = {
            itemid: idx for idx, itemid in enumerate(self.data["itemId"].unique())
        }
        self.data["user_index"] = self.data["userId"].map(self.user_indices)
        self.data["item_index"] = self.data["itemId"].map(self.item_indices)

    @property
    def user_num(self) -> int:
        return len(self.user_indices)

    @property
    def item_num(self) -> int:
        return len(self.item_indices)
