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
        self.scheduler = get_scheduler(optimizer=optimizer,
                                       scheduler_type=scheduler_type,
                                       max_lr=max_lr,
                                       num_batches=len(self.train_dataloader),
                                       num_epochs=num_epochs,
                                       warmup_ratio=warmup_ratio)

    def calculate_bce_loss(self,
                           positive_idxs: torch.Tensor,
                           negative_idxs: torch.Tensor,
                           positive_logits: torch.Tensor,
                           negative_logits: torch.Tensor) -> torch.Tensor:
        loss_func = nn.BCEWithLogitsLoss()

        positive_logits = positive_logits[positive_idxs]
        positive_labels = torch.ones(size=positive_logits.shape)

        negative_logits = negative_logits[negative_idxs]
        negative_labels = torch.zeros(size=negative_logits.shape)

        positive_loss = loss_func(positive_logits, positive_labels)
        negative_loss = loss_func(negative_logits, negative_labels)

        return positive_loss + negative_loss

    def train(self) -> None:
        epoch_pbar = trange(self.num_epochs,
                            desc="Epochs: ",
                            total=self.num_epochs)
        for epoch in epoch_pbar:
            self.model.train()
            self.model.zero_grad()

            train_pbar = tqdm(iterable=self.train_dataloader,
                              desc="Training",
                              total=len(self.train_dataloader))
            for batch in train_pbar:
                self.optimizer.zero_grad()

                positive_seqs = batch.clone()
                positive_idxs = torch.where(positive_seqs != 0)

                batch[:, -1] = 0
                input_seqs = batch.roll(shifts=1)
                negative_seqs = get_negative_samples(self.positive2negatives, positive_seqs)
                negative_idxs = torch.where(negative_seqs != 0)

                inputs = {'input_seqs': input_seqs,
                          'positive_seqs': positive_seqs,
                          'negative_seqs': negative_seqs}

                output = self.model(**inputs)
                assert len(output) == 3, f"Wrong number of outputs ({len(output)})"

                positive_logits = output[1]
                negative_logits = output[2]

                loss = self.calculate_bce_loss(positive_idxs=positive_idxs,
                                               negative_idxs=negative_idxs,
                                               positive_logits=positive_logits,
                                               negative_logits=negative_logits)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def evaluate(self):
        pass

    def predict(self):
        pass