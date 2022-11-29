import torch
import torch.nn as nn


class BaseGeneralizedMatrixFactorization(nn.Module):
    def __init__(self, user_num: int, item_num: int, dim: int) -> None:
        super().__init__()

        self.user_emb = nn.Embedding(num_embeddings=user_num, embedding_dim=dim)
        self.item_emb = nn.Embedding(num_embeddings=item_num, embedding_dim=dim)

    def _init_weight(self) -> None:
        nn.init.uniform_(self.user_emb.weight.data, a=0.0, b=1.0)
        nn.init.uniform_(self.item_emb.weight.data, a=0.0, b=1.0)

    def forward(self) -> torch.Tensor:
        raise NotImplementedError
