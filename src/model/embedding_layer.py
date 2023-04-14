import math

import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_items: int, hidden_dim: int, max_seq_len: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.item_emb_matrix = nn.Embedding(
            num_embeddings=num_items + 1, embedding_dim=hidden_dim
        )
        self.positional_emb = nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=hidden_dim
        )

    def forward(self, x):
        x = self.item_emb_matrix(x)
        x *= math.sqrt(self.hidden_dim)

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        device = x.device.type

        seq_len_range = torch.tensor(range(seq_len))
        positions = torch.tile(input=seq_len_range, dims=(batch_size, 1))
        positions = positions.to(device)

        positional_embs = self.positional_emb(positions)
        x += positional_embs

        return x
