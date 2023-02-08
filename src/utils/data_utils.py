import random

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


InputSequences = torch.Tensor
PositiveSamples = torch.Tensor
NegativeSamples = torch.Tensor


def get_positive2negatives(num_items: int,
                           num_samples: int=100) -> list[int]:
    """
    Creates a dictionary that maps an integer to an array of
      negative integers. This dictionary will be used later
      when we create negative samples for each positive sample.
    """
    all_samples = np.arange(1, num_items + 1)
    positive2negatives = {}
    for positive_sample in tqdm(iterable=all_samples, desc="Negs", total=all_samples.shape[0]):
        candidates = np.concatenate((np.arange(positive_sample),
                                     np.arange(positive_sample + 1, num_items + 1)),
                                    axis=0)
        negative_samples = np.random.choice(candidates,
                                            size=(num_samples,),
                                            replace=False)

        positive2negatives[positive_sample] = negative_samples

    return positive2negatives


def get_negative_samples(positive2negatives: dict[int, list[int]],
                         positive_seqs: torch.Tensor,
                         num_samples=1) -> torch.Tensor:
    negative_seqs = torch.zeros(size=positive_seqs.shape, dtype=torch.long)
    for row_idx in range(positive_seqs.shape[0]):
        for col_idx in range(positive_seqs[row_idx].shape[0]):
            positive_sample = positive_seqs[row_idx][col_idx].item()

            if positive_sample == 0:
                continue

            negative_samples = positive2negatives[positive_sample]
            negative_sample = np.random.choice(a=negative_samples,
                                               size=(num_samples,),
                                               replace=False)
            negative_seqs[row_idx][col_idx] = negative_sample[0]

    return negative_seqs


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
