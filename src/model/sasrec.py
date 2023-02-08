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
        if not share_item_emb:
            self.classifier = nn.Linear(in_features=hidden_dim,
                                        out_features=num_items)

    def forward(self,
                input_seqs: torch.Tensor,
                positive_seqs: torch.Tensor=None,
                negative_seqs: torch.Tensor=None) -> torch.Tensor:
        input_emb = self.dropout(self.embedding_layer(input_seqs))
        input_attn = self.self_attn_blocks(input_emb)

        if not self.share_item_emb:
            output_logits = self.classifier(input_attn)
        else:
            output_logits = input_attn @ self.embedding_layer.item_emb_matrix.weight.transpose(1, 0)

        outputs = (output_logits,)

        if (positive_seqs is not None) and (negative_seqs is not None):
            positive_emb = self.dropout(self.embedding_layer(positive_seqs))
            negative_emb = self.dropout(self.embedding_layer(negative_seqs))

            positive_logits = input_attn * positive_emb
            negative_logits = input_attn * negative_emb

            outputs += positive_logits
            outputs += negative_logits

        return outputs
