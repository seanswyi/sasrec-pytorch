import argparse
import os
import time

from arguments import get_args
from dataset import Dataset


def main() -> None:
    args = get_args()
    print(args)

    dataset = Dataset(args)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
