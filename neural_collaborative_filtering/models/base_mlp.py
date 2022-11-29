import torch
import torch.nn as nn


class BaseMultiLayerPerceptron(nn.Module):
    def __init__(
        self, user_num: int, item_num: int, dim: int, layer_nums_list: list[int]
    ) -> None:
        super().__init__()

        if (dim * 2) != layer_nums_list[0]:
            raise ValueError(
                "dim * 2 should be the same as first item in layer_nums_list"
            )

        self.user_emb = nn.Embedding(num_embeddings=user_num, embedding_dim=dim)
        self.item_emb = nn.Embedding(num_embeddings=item_num, embedding_dim=dim)

        layers = []
        for i, in_features in enumerate(layer_nums_list):
            if (len(layer_nums_list) - 1) == i:
                break
            out_features = layer_nums_list[i + 1]
            layer = nn.Linear(in_features=in_features, out_features=out_features)
            layers.append(layer)
            layers.append(nn.ReLU())

        self.fc_layers = nn.Sequential(*layers)

    def _init_weight(self) -> None:
        nn.init.uniform_(self.user_emb.weight.data, a=0.0, b=1.0)
        nn.init.uniform_(self.item_emb.weight.data, a=0.0, b=1.0)

    def forward(self) -> torch.Tensor:
        raise NotImplementedError
