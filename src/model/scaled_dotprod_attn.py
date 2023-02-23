import math

import torch
from torch.nn import functional as F


def scaled_dotprod_attn(q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        d: int) -> torch.Tensor:
    """
    Perform scaled dot-product attention as described in the paper.
    """
    qk = q @ k
    qk /= math.sqrt(d)

    # To prevent peaking.
    ones = torch.ones(size=qk.shape)
    mask_matrix = torch.triu(ones, diagonal=1) * (-1e9)

    device = qk.device.type
    mask_matrix = mask_matrix.to(device)
    qk += mask_matrix

    softmax_qk = F.softmax(qk, dim=-1)
    attn = softmax_qk @ v

    return attn
