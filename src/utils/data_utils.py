import random

import torch
from torch.nn import functional as F


InputSequences = torch.Tensor
PositiveSamples = torch.Tensor
NegativeSamples = torch.Tensor


def get_negative_samples(positive_labels: PositiveSamples,
                         num_items: int,
                         num_samples: int=1) -> NegativeSamples:
    """
    `seen` refers to the positive labels.
    `candidates` is a list of item IDs excluding the positive labels.
    `random.sample` is used to sample the negative labels.
    """
    negative_samples = []
    for positive_sample in positive_samples:
        seen = [positive_sample]
        candidates = [idx for idx in range(1, num_items + 1) if idx not in seen]
        negative_sample = random.sample(population=candidates, k=num_samples)

        assert len(negative_sample) == len(set(negative_sample))

        negative_samples.append(negative_sample)

    negative_samples = torch.tensor(negative_samples)
    return negative_samples


def pad_or_truncate_seq(sequence: list[int],
                        max_seq_len: int) -> InputSequences:
    """Pads or truncates sequences depending on max_seq_len."""
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)

    if len(sequence) > max_seq_len:
        sequence = sequence[-max_seq_len:]
    else:
        diff = max_seq_len - len(sequence)
        sequence = F.pad(sequence, pad=(diff, 0))

    return sequence
