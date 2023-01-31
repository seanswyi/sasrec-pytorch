import torch
from torch import nn

from .pointwise_ffnn import PointWiseFFNN
from .self_attn import SelfAttn


class SelfAttnBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_p: float) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.self_attn = SelfAttn(hidden_dim=hidden_dim)
        self.ffnn = PointWiseFFNN(hidden_dim=hidden_dim)

    def dropout_and_ln(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.layer_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn = x + self.dropout_and_ln(self.self_attn(x))
        x_ffnn = x_attn + self.dropout_and_ln(x_attn)
        return x_ffnn
