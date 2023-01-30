import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 num_items: int,
                 hidden_dim: int,
                 max_seq_len: int) -> None:
        super().__init__()

        self.item_emb_matrix = nn.Embedding(num_embeddings=num_items,
                                            embedding_dim=hidden_dim)
        self.positional_emb = nn.Parameter(data=torch.rand(size=(max_seq_len, hidden_dim)))
        nn.init.normal_(self.positional_emb)

    def forward(self, x):
        x = self.item_emb_matrix(x)
        x += self.positional_emb
        return x