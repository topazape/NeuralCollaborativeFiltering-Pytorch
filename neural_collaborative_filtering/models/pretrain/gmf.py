import torch
import torch.nn as nn

from ..base_gmf import BaseGeneralizedMatrixFactorization


class PretrainGeneralizedMatrixFactorization(BaseGeneralizedMatrixFactorization):
    def __init__(self, user_num: int, item_num: int, dim: int) -> None:
        super().__init__(user_num=user_num, item_num=item_num, dim=dim)

        self._init_weight()

        self.linear = nn.Linear(in_features=dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        elem_product = torch.mul(user_emb, item_emb)
        out = self.linear(elem_product)
        out = self.sigmoid(out)
        return out.squeeze(-1)
