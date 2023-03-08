import torch
from torch import nn

from model import PointWiseFFNN, SelfAttn


class SelfAttnBlock(nn.Module):
    def __init__(self,
                 max_seq_len: int,
                 hidden_dim: int,
                 dropout_p: float,
                 device: str) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=1,
                                               dropout=dropout_p,
                                               batch_first=True)
        self.ffnn = PointWiseFFNN(hidden_dim=hidden_dim)

    def dropout_layernorm(self, x: torch.Tensor) -> torch.Tensor:
        layer_norm_output = self.layer_norm(x)
        dropout_output = self.dropout(layer_norm_output)

        return dropout_output

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        attention_mask = ~torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.bool))
        device = x.device.type
        attention_mask = attention_mask.to(device)

        x_attn, _ = self.self_attn(key=x,
                                   query=x,
                                   value=x,
                                   attn_mask=attention_mask)
        x_attn_output = x + self.dropout_layernorm(x_attn)

        x_ffnn = self.ffnn(x_attn_output)
        x_ffnn_output = x_attn_output + self.dropout_layernorm(x_ffnn)

        output = x_ffnn_output * padding_mask.unsqueeze(-1)
        return output
