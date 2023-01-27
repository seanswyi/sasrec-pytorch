import argparse
import os
import time

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

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
