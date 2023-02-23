import argparse
import os
import time

import torch
from torch.nn import functional as F
from torch import optim

from utils import (get_args,
                   get_device,
                   DatasetArgs,
                   ModelArgs,
                   OptimizerArgs,
                   TrainerArgs)
from dataset import Dataset
from model import SASRec
from trainer import Trainer


def main() -> None:
    args = get_args()
    args.device = get_device()

    if args.debug:
        args.num_epochs = 1

    dataset_args = DatasetArgs(args)
    dataset = Dataset(**vars(dataset_args))

    args.num_items = dataset.num_items
    model_args = ModelArgs(args)
    model = SASRec(**vars(model_args))
    model = model.to(args.device)

    optimizer_args = OptimizerArgs(args)
    optimizer = optim.Adam(params=model.parameters(),
                           **vars(optimizer_args))

    trainer_args = TrainerArgs(args)
    trainer = Trainer(dataset=dataset,
                      model=model,
                      optimizer=optimizer,
                      **vars(trainer_args))
    bn, bne, bh, bhe = trainer.train()

    print(f"Best nDCG@10 was {bn} at epoch {bne + 1}")
    print(f"Best Hit@10 was {bh} at epoch {bhe + 1}")


if __name__ == '__main__':
    main()
