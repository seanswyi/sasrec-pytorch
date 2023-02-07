import numpy as np
import random

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

def get_negative_samples(positive_samples: PositiveSamples,
                         num_items: int,
                         num_samples: int=1) -> NegativeSamples:
    """
    `seen` refers to the positive labels.
    `candidates` is a list of item IDs excluding the positive labels.
    `random.sample` is used to sample the negative labels.
    """
    import pdb; pdb.set_trace()
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
