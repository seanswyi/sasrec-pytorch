import argparse

import torch
from torch.optim import Optimizer
from tqdm import tqdm, trange

from dataset import Dataset
from model import SASRec
from utils import get_scheduler

class Trainer:
    def __init__(self,
                 dataset: Dataset,
                 model: SASRec,
                 optimizer: Optimizer,
                 num_epochs: int,
                 warmup_ratio: float,
                 scheduler_type: str) -> None:
        self.train_data = dataset.user2items_train
        self.valid_data = dataset.user2items_valid
        self.test_data = dataset.user2items_test

        self.train_dataloader = dataset.get_dataloader(data=self.train_data)
        self.valid_dataloader = dataset.get_dataloader(data=self.valid_data, shuffle=False)
        self.test_dataloader = dataset.get_dataloader(data=self.test_data, shuffle=False)

        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.scheduler_type = scheduler_type

        self.model = model
        self.optimizer = optimizer

    def train(self) -> None:
        loss_func = nn.CrossEntropyLoss(ignore_idx=0)
        scheduler = get_scheduler(optimizer=self.optimizer,
                                  scheduler_type=self.scheduler_type,
                                  num_batches=len(self.train_dataloader),
                                  num_epochs=self.num_epochs,
                                  warmup_ratio=self.warmup_ratio)

        epoch_pbar = trange(iterable=self.num_epochs,
                            desc="Epochs: ",
                            total=self.num_epochs)

    def evaluate(self):
        pass

    def predict(self):
        pass