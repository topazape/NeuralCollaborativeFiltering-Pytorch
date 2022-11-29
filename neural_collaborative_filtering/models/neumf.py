import torch
import torch.nn as nn

from .base_gmf import BaseGeneralizedMatrixFactorization
from .base_mlp import BaseMultiLayerPerceptron


class GeneralizedMatrixFactorization(BaseGeneralizedMatrixFactorization):
    def __init__(self, user_num: int, item_num: int, dim: int) -> None:
        super().__init__(user_num=user_num, item_num=item_num, dim=dim)
        self._init_weight()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)

        out = torch.mul(user_emb, item_emb)

        return out


class MultiLayerPerceptron(BaseMultiLayerPerceptron):
    def __init__(
        self, user_num: int, item_num: int, dim: int, layer_nums_list: list[int]
    ) -> None:
        super().__init__(
            user_num=user_num,
            item_num=item_num,
            dim=dim,
            layer_nums_list=layer_nums_list,
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        vec = torch.cat([user_emb, item_emb], dim=-1)
        out = self.fc_layers(vec)

        return out


class NeuMF(nn.Module):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        gmf_dim: int,
        mlp_dim: int,
        layer_nums_list: list[int],
    ) -> None:
        super().__init__()

        self.gmf = GeneralizedMatrixFactorization(
            user_num=user_num, item_num=item_num, dim=gmf_dim
        )
        self.mlp = MultiLayerPerceptron(
            user_num=user_num,
            item_num=item_num,
            dim=mlp_dim,
            layer_nums_list=layer_nums_list,
        )

        self.linear = nn.Linear(
            in_features=(gmf_dim + layer_nums_list[-1]), out_features=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        out_gmf = self.gmf(users, items)
        out_mlp = self.mlp(users, items)
        vec = torch.cat([out_gmf, out_mlp], dim=-1)
        out = self.linear(vec)
        out = self.sigmoid(out)
        return out.squeeze(-1)
