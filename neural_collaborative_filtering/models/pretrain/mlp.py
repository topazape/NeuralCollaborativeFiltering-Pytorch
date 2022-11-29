import torch
import torch.nn as nn

from ..base_mlp import BaseMultiLayerPerceptron


class PretrainMultiLayerPerceptron(BaseMultiLayerPerceptron):
    def __init__(
        self, user_num: int, item_num: int, dim: int, layer_nums_list: list[int]
    ) -> None:
        super().__init__(
            user_num=user_num,
            item_num=item_num,
            dim=dim,
            layer_nums_list=layer_nums_list,
        )
        self._init_weight()

        self.linear = nn.Linear(in_features=layer_nums_list[-1], out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        vec = torch.cat([user_emb, item_emb], dim=-1)
        out = self.fc_layers(vec)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out.squeeze(-1)
