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
                 scheduler_type: str,
                 device: str) -> None:
        self.device = device

        self.train_data = dataset.user2items_train
        self.valid_data = dataset.user2items_valid
        self.test_data = dataset.user2items_test

        self.train_dataloader = dataset.get_dataloader(data=self.train_data)
        self.valid_dataloader = dataset.get_dataloader(data=self.valid_data, split='valid')
        self.test_dataloader = dataset.get_dataloader(data=self.test_data, split='test')

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
        positive_labels = torch.ones(size=positive_logits.shape).to(self.device)

        negative_logits = negative_logits[negative_idxs]
        negative_labels = torch.zeros(size=negative_logits.shape).to(self.device)

        positive_loss = loss_func(positive_logits, positive_labels)
        negative_loss = loss_func(negative_logits, negative_labels)

        return positive_loss + negative_loss

    def train(self) -> None:
        num_steps = 0
        epoch_pbar = trange(self.num_epochs,
                            desc="Epochs: ",
                            total=self.num_epochs)
        for epoch in epoch_pbar:
            self.model.train()

            epoch_loss = 0

            train_pbar = tqdm(iterable=self.train_dataloader,
                              desc="Training",
                              total=len(self.train_dataloader))
            for batch in train_pbar:
                self.model.zero_grad()

                positive_seqs = batch.clone()
                positive_idxs = torch.where(positive_seqs != 0)

                batch[:, -1] = 0
                input_seqs = batch.roll(shifts=1)
                negative_seqs = get_negative_samples(self.positive2negatives, positive_seqs)
                negative_idxs = torch.where(negative_seqs != 0)

                inputs = {'input_seqs': input_seqs.to(self.device),
                          'positive_seqs': positive_seqs.to(self.device),
                          'negative_seqs': negative_seqs.to(self.device)}
                output = self.model(**inputs)

                positive_logits = output[0]
                negative_logits = output[1]

                loss = self.calculate_bce_loss(positive_idxs=positive_idxs,
                                               negative_idxs=negative_idxs,
                                               positive_logits=positive_logits,
                                               negative_logits=negative_logits)
                loss.backward()
                epoch_loss += loss.item()
                self.optimizer.step()
                self.scheduler.step()

                num_steps += 1

            self.evaluate()

            print(f"Epoch {epoch}, loss: {epoch_loss: 0.6f}")
        import pdb; pdb.set_trace()

    def evaluate(self, mode='valid'):
        if mode == 'valid':
            dataloader = self.valid_dataloader
        else:
            dataloader = self.test_dataloader

        self.model.eval()
        eval_pbar = tqdm(iterable=dataloader,
                         desc=f"Evaluating for {mode}",
                         total=len(dataloader))
        for batch in eval_pbar:
            input_seqs, item_idxs = batch

            inputs = {'input_seqs': input_seqs.to(self.device),
                      'item_idxs': item_idxs.to(self.device)}
            outputs = self.model(**inputs)
            import pdb; pdb.set_trace()

    def predict(self):
        pass