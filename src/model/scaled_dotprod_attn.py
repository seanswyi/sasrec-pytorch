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
    scaled_qk = qk / torch.sqrt(d)
    softmax_qk = F.softmax(scaled_qk, dim=-1)

    attn = softmax_qk @ v

    return attn
