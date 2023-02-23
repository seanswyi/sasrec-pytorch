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
        # self.valid_dataloader = dataset.get_dataloader(data=self.train_data, split='valid')
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
        best_ndcg = 0
        best_hit_rate = 0
        best_ndcg_epoch = 0
        best_hit_epoch = 0

        num_steps = 0
        epoch_pbar = trange(self.num_epochs,
                            desc="Epochs: ",
                            total=self.num_epochs)
        for epoch in epoch_pbar:
            self.model.train()

            epoch_loss = 0
            loss_func = nn.BCEWithLogitsLoss()

            # train_pbar = tqdm(iterable=self.train_dataloader,
            #                   desc="Training",
            #                   total=len(self.train_dataloader))
            # for batch in train_pbar:
            for batch in self.train_dataloader:
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

            ndcg, hit = self.evaluate()

            if ndcg >= best_ndcg:
                best_ndcg = ndcg
                best_ndcg_epoch = epoch

            if hit >= best_hit_rate:
                best_hit_rate = hit
                best_hit_epoch = epoch

            print(f"Epoch {epoch}, loss: {epoch_loss: 0.6f}")

        return best_ndcg, best_ndcg_epoch, best_hit_rate, best_hit_epoch

    def evaluate(self, mode='valid'):
        if mode == 'valid':
            dataloader = self.valid_dataloader
        else:
            dataloader = self.test_dataloader

        ndcg = 0
        hit = 0
        num_users = 0

        self.model.eval()
        # eval_pbar = tqdm(iterable=dataloader,
        #                  desc=f"Evaluating for {mode}",
        #                  total=len(dataloader))
        # for batch in eval_pbar:
        for batch in dataloader:
            input_seqs, item_idxs = batch
            num_users += input_seqs.shape[0]

            inputs = {'input_seqs': input_seqs.to(self.device),
                      'item_idxs': item_idxs.to(self.device)}
            outputs = self.model(**inputs)

            # inputs = {'log_seqs': input_seqs.to(self.device),
            #           'item_indices': item_idxs.to(self.device)}
            # outputs = self.model.predict(**inputs)

            logits = -outputs[0]

            if logits.device.type == 'mps':
                logits = logits.detach().cpu()

            ranks = logits.argsort().argsort()

            # rank = ranks[0].item()
            # if rank < 10:
            #     ndcg += (1 / np.log2(rank + 2))
            #     hit += 1
            # import pdb; pdb.set_trace()

            # MINE. #############################
            ranks = [r[0].item() for r in ranks]

            for rank in ranks:
                if rank < 10:
                    ndcg += (1 / np.log2(rank + 2))
                    hit += 1
            #####################################

        ndcg /= num_users
        hit /= num_users

        print(f"\nnDCG@10: {ndcg: 0.6f}, Hit@10: {hit: 0.6f}\n")

        return ndcg, hit
