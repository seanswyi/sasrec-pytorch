import torch
from torch import nn

from .embedding_layer import EmbeddingLayer
from .self_attn_block import SelfAttnBlock


class SASRec(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.embedding_layer = EmbeddingLayer(num_items=args.num_items,
                                              hidden_dim=args.hidden_dim,
                                              max_seq_len=args.max_seq_len)

        self_attn_blocks = [SelfAttnBlock(hidden_dim=args.hidden_dim,
                                          dropout_p=args.dropout_p)
                            for _ in range(args.num_blocks)]
        self.self_attn_blocks = nn.Sequential(*self_attn_blocks)

        self.share_item_emb = args.share_item_emb
        if args.share_item_emb:
            self.classifier = nn.Linear(in_features=args.hidden_dim,
                                        out_features=args.num_items)

    def forward(self, x):
        x_embedded = self.dropout(self.embedding_layer(x))
        x_attn = self.self_attn_blocks(x_embedded)
        import pdb; pdb.set_trace()