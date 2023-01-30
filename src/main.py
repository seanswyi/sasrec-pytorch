import argparse
import os
import time

import torch
from torch.nn import functional as F

from arguments import get_args
from dataset import Dataset
from model.sasrec import SASRec


def main() -> None:
    args = get_args()
    print(args)

    dataset = Dataset(args)

    args.num_users = len(dataset.user2items)
    args.num_items = len(dataset.item2users)

    model = SASRec(args)

    train_data = dataset.user2items_train
    sample = torch.tensor(train_data[1])
    sample = F.pad(input=sample, pad=(0, args.max_seq_len - sample.shape[0]))
    model(sample)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
