import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 num_items: int,
                 hidden_dim: int,
                 max_seq_len: int) -> None:
        super().__init__()

        self.item_emb_matrix = nn.Embedding(num_embeddings=num_items + 1,
                                            embedding_dim=hidden_dim)
        self.positional_emb = nn.Embedding(num_embeddings=max_seq_len,
                                           embedding_dim=hidden_dim)

        self.max_seq_len = max_seq_len

    def forward(self, x):
        x = self.item_emb_matrix(x)

        max_seq_len_range = range(max_seq_len)
        batch_size = x.shape[0]
        device = x.device.type
        positions = torch.tile(input=max_seq_len_range, dims=(batch_size, 1)).to(device)
        positional_embs = self.positional_emb(positions)

        x += positional_embs

        return x