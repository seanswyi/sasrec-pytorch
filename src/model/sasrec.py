import argparse

import torch
from torch import nn

from model import EmbeddingLayer, SelfAttnBlock


class SASRec(nn.Module):
    def __init__(self, model_args: argparse.Namespace) -> None:
        super().__init__()

        self.embedding_layer = EmbeddingLayer(num_items=model_args.num_items,
                                              hidden_dim=model_args.hidden_dim,
                                              max_seq_len=model_args.max_seq_len)

        self_attn_blocks = [SelfAttnBlock(hidden_dim=model_args.hidden_dim,
                                          dropout_p=model_args.dropout_p)
                            for _ in range(model_args.num_blocks)]
        self.self_attn_blocks = nn.Sequential(*self_attn_blocks)

        self.dropout = nn.Dropout(p=model_args.dropout_p)

        self.share_item_emb = model_args.share_item_emb
        if model_args.share_item_emb:
            self.classifier = nn.Linear(in_features=model_args.hidden_dim,
                                        out_features=model_args.num_items)

    def forward(self, x):
        x_embedded = self.dropout(self.embedding_layer(x))
        x_attn = self.self_attn_blocks(x_embedded)
        return x_attn
