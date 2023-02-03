import argparse

import torch
from torch import nn

from model import EmbeddingLayer, SelfAttnBlock


class SASRec(nn.Module):
    def __init__(self,
                 num_items: int,
                 num_blocks: int,
                 hidden_dim: int,
                 max_seq_len: int,
                 dropout_p: float,
                 share_item_emb: bool) -> None:
        super().__init__()

        self.embedding_layer = EmbeddingLayer(num_items=num_items,
                                              hidden_dim=hidden_dim,
                                              max_seq_len=max_seq_len)

        self_attn_blocks = [SelfAttnBlock(hidden_dim=hidden_dim,
                                          dropout_p=dropout_p)
                            for _ in range(num_blocks)]
        self.self_attn_blocks = nn.Sequential(*self_attn_blocks)

        self.dropout = nn.Dropout(p=dropout_p)

        self.share_item_emb = share_item_emb
        if share_item_emb:
            self.classifier = nn.Linear(in_features=hidden_dim,
                                        out_features=num_items)

    def forward(self, x):
        x_embedded = self.dropout(self.embedding_layer(x))
        x_attn = self.self_attn_blocks(x_embedded)
        return x_attn
