import torch
from torch import nn


class PointWiseFFNN(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()

        self.W1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.W2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.relu(self.W1(x))
        x_2 = self.W2(x_1)

        return x_2
