import torch
from torch import nn

from embedding_layer import EmbeddingLayer
from self_attn import SelfAttn


class SASRec(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.embedding_layer = EmbeddingLayer(num_items=args.num_items,
                                              hidden_dim=args.hidden_dim,
                                              max_seq_len=args.max_seq_len)

    def forward(self, x):
        pass