import torch
from torch import nn

from model import scaled_dotprod_attn


class SelfAttn(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.W_q = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_k = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W_v = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.W_q(x)
        x_k = self.W_k(x)
        x_v = self.W_v(x)

        attended_x = scaled_dotprod_attn(q=x_q,
                                         k=x_k,
                                         v=x_v,
                                         d=self.hidden_dim)

        return attended_x
