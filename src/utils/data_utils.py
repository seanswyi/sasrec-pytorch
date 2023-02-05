import torch
from torch.nn import functional as F


PositiveLabels = torch.Tensor
NegativeLabels = torch.Tensor


def get_negative_labels(positive_labels: PositiveLabels,
                        num_items: int,
                        num_samples: int=1) -> NegativeLabels:
    """
    `seen` refers to the positive label. Sampling happens as following:
        1. Until we've reached our desired number of samples:
        1.1 Get sample candidates by excluding `seen` from the range.
        1.2 Once a new negative label is sampled, update `seen` and continue.
        2. Return negative sample sequences once all are done.
    """
    negative_labels = []
    for positive_label in tqdm(positive_labels):
        seen = [positive_label]
        count = 0
        while count < num_samples:
            candidates = [idx for idx in range(1, num_items + 1) if idx not in seen]
            negative_label = random.choice(candidates)

            while negative_label in seen:
                negative_label = random.choice(candidates)

            seen.append(negative_label)
            count += 1
        negative_labels.append(seen[1:])

    negative_labels = torch.tensor(negative_labels)
    return negative_labels


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
