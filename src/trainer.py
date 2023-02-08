import argparse

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm, trange

from dataset import Dataset
from model import SASRec
from utils import get_negative_samples, get_scheduler

class Trainer:
    def __init__(self,
                 dataset: Dataset,
                 model: SASRec,
                 optimizer: Optimizer,
                 max_lr: float,
                 num_epochs: int,
                 warmup_ratio: float,
                 scheduler_type: str) -> None:
        self.train_data = dataset.user2items_train
        self.valid_data = dataset.user2items_valid
        self.test_data = dataset.user2items_test

        self.train_dataloader = dataset.get_dataloader(data=self.train_data)
        self.valid_dataloader = dataset.get_dataloader(data=self.valid_data, shuffle=False)
        self.test_dataloader = dataset.get_dataloader(data=self.test_data, shuffle=False)

        self.positive2negatives = dataset.positive2negatives

        self.max_lr = max_lr
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.scheduler_type = scheduler_type

        self.model = model
        self.optimizer = optimizer

    def train(self) -> None:
        # loss_func = nn.CrossEntropyLoss(ignore_idx=0)
        scheduler = get_scheduler(optimizer=self.optimizer,
                                  scheduler_type=self.scheduler_type,
                                  max_lr=self.max_lr,
                                  num_batches=len(self.train_dataloader),
                                  num_epochs=self.num_epochs,
                                  warmup_ratio=self.warmup_ratio)

        epoch_pbar = trange(self.num_epochs,
                            desc="Epochs: ",
                            total=self.num_epochs)
        for epoch in epoch_pbar:
            self.model.train()
            self.model.zero_grad()
            self.optimizer.zero_grad()

            train_pbar = tqdm(iterable=self.train_dataloader,
                              desc="Training",
                              total=len(self.train_dataloader))
            for batch in train_pbar:
                positive_seqs = batch.clone()

                batch[:, -1] = 0
                input_seqs = batch.roll(shifts=1)
                negative_seqs = get_negative_samples(self.positive2negatives, positive_seqs)

                inputs = {'input_seqs': input_seqs,
                          'positive_seqs': positive_seqs,
                          'negative_seqs': negative_seqs}
                output = self.model(**inputs)
                import pdb; pdb.set_trace()

    def evaluate(self):
        pass

    def predict(self):
        pass