import torch
import torch.nn as nn


class BaseGeneralizedMatrixFactorization(nn.Module):
    def __init__(self, user_num: int, item_num: int, dim: int) -> None:
        super().__init__()

        self.user_emb = nn.Embedding(num_embeddings=user_num, embedding_dim=dim)
        self.item_emb = nn.Embedding(num_embeddings=item_num, embedding_dim=dim)

    def _init_weight(self) -> None:
        self.user_emb.weight.data.normal_(mean=0, std=0.01)
        self.item_emb.weight.data.normal_(mean=0, std=0.01)

    def forward(self) -> torch.Tensor:
        raise NotImplementedError
