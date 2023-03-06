import argparse
from collections import OrderedDict
import copy
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm, trange

from dataset import Dataset
from model import SASRec
from utils import get_negative_samples, get_scheduler


logger = logging.getLogger()


class Trainer:
    def __init__(self,
                 dataset: Dataset,
                 evaluate_k: int,
                 model: SASRec,
                 optimizer: Optimizer,
                 max_lr: float,
                 num_epochs: int,
                 warmup_ratio: float,
                 use_scheduler: bool,
                 scheduler_type: str,
                 output_dir: str,
                 save_dir: str,
                 resume_training: bool=False,
                 device: str='cpu') -> None:
        self.device = device
        self.evaluate_k = evaluate_k
        self.save_dir = os.path.join(output_dir, save_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

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

        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler = get_scheduler(optimizer=optimizer,
                                           scheduler_type=scheduler_type,
                                           max_lr=max_lr,
                                           num_batches=len(self.train_dataloader),
                                           num_epochs=num_epochs,
                                           warmup_ratio=warmup_ratio)
        else:
            self.scheduler = None

        self.resume_training = resume_training
        if self.resume_training:
            checkpoint_file = os.path.join(self.save_dir, 'most_recent_checkpoint.pt')
            checkpoint = torch.load(f=checkpoint_file)

            most_recent_epoch = checkpoint['most_recent_epoch']
            most_recent_model = checkpoint['most_recent_model_state_dict']
            most_recent_optimizer = checkpoint['most_recent_optim_state_dict']
            most_recent_scheduler = checkpoint['most_recent_scheduler_state_dict']

            self.model.load_state_dict(most_recent_model)
            self.optimizer.load_state_dict(most_recent_optimizer)
            self.num_epochs = num_epochs - most_recent_epoch

            if self.scheduler:
                self.scheduler.load_state_dict(most_recent_scheduler)

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

    def save_results(self,
                     epoch: int,
                     ndcg: float,
                     model_state_dict: OrderedDict[str, torch.Tensor],
                     optim_state_dict: OrderedDict[str, torch.Tensor],
                     scheduler_state_dict: OrderedDict[str, torch.Tensor]=None,
                     save_name: str='best') -> None:
        checkpoint = {f'{save_name}_epoch': epoch,
                      f'{save_name}_ndcg': ndcg,
                      f'{save_name}_model_state_dict': model_state_dict,
                      f'{save_name}_optim_state_dict': optim_state_dict,
                      f'{save_name}_scheduler_state_dict': scheduler_state_dict}
        save_dir = os.path.join(self.save_dir, f'{save_name}_checkpoint.pt')
        torch.save(obj=checkpoint, f=save_dir)

    def train(self) -> (int,
                        OrderedDict[str, torch.Tensor],
                        OrderedDict[str, torch.Tensor]):
        best_ndcg = 0
        best_hit_rate = 0
        best_ndcg_epoch = 0
        best_hit_epoch = 0
        best_model_state_dict = None
        best_optim_state_dict = None
        best_scheduler_state_dict = None

        last_epoch = 0
        num_steps = 0
        epoch_pbar = trange(self.num_epochs,
                            desc="Epochs: ",
                            total=self.num_epochs)
        for epoch in epoch_pbar:
            self.model.train()

            epoch_loss = 0
            loss_func = nn.BCEWithLogitsLoss()

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

                if self.use_scheduler:
                    self.scheduler.step()

                num_steps += 1

            ndcg, hit = self.evaluate()

            if ndcg >= best_ndcg:
                best_ndcg = ndcg
                best_ndcg_epoch = epoch
                best_model_state_dict = copy.deepcopy(x=self.model.state_dict())
                best_optim_state_dict = copy.deepcopy(x=self.optimizer.state_dict())

                self.save_results(epoch=best_ndcg_epoch,
                                  ndcg=best_ndcg,
                                  model_state_dict=best_model_state_dict,
                                  optim_state_dict=best_optim_state_dict,
                                  scheduler_state_dict=best_scheduler_state_dict)

            if hit >= best_hit_rate:
                best_hit_rate = hit
                best_hit_epoch = epoch

            epoch_result_msg = f"Epoch {epoch}, \
                loss: {epoch_loss: 0.6f}, \
                    nDCG@{self.evaluate_k}: {ndcg: 0.4f}, \
                        Hit@{self.evaluate_k}: {hit: 0.4f}"
            logger.info(epoch_result_msg)

            most_recent_model = self.model.state_dict()
            most_recent_optim = self.optimizer.state_dict()
            most_recent_scheduler = None

            if self.use_scheduler:
                most_recent_scheduler = self.scheduler.state_dict()

            self.save_results(epoch=epoch,
                              ndcg=best_ndcg,
                              model_state_dict=most_recent_model,
                              optim_state_dict=most_recent_optim,
                              scheduler_state_dict=most_recent_scheduler,
                              save_name='most_recent')

        best_ndcg_msg = f"Best nDCG@{self.evaluate_k} was {best_ndcg: 0.6f} at epoch {best_ndcg_epoch}."
        best_hit_msg = f"Best Hit@{self.evaluate_k} was {best_hit_rate: 0.6f} at epoch {best_hit_epoch}."
        best_results_msg = '\n'.join([best_ndcg_msg, best_hit_msg])
        logger.info(f"Best results:\n{best_results_msg}")

        return (best_ndcg_epoch, best_model_state_dict, best_optim_state_dict)

    def evaluate(self, mode: str='valid', model: SASRec=None) -> tuple[float, float]:
        if mode == 'valid':
            dataloader = self.valid_dataloader
        else:
            dataloader = self.test_dataloader

        if model:
            self.model = model

        ndcg = 0
        hit = 0
        num_users = 0

        self.model.eval()

        with torch.no_grad():
            eval_pbar = tqdm(iterable=dataloader,
                             desc=f"Evaluating for {mode}",
                             total=len(dataloader))
            for batch in eval_pbar:
                input_seqs, item_idxs = batch
                num_users += input_seqs.shape[0]

                inputs = {'input_seqs': input_seqs.to(self.device),
                          'item_idxs': item_idxs.to(self.device)}
                outputs = self.model(**inputs)

                logits = -outputs[0]

                if logits.device.type == 'mps': # torch.argsort isn't implemented for MPS.
                    logits = logits.detach().cpu()

                ranks = logits.argsort().argsort()
                ranks = [r[0].item() for r in ranks]

                for rank in ranks:
                    if rank < self.evaluate_k:
                        ndcg += (1 / np.log2(rank + 2))
                        hit += 1

        ndcg /= num_users
        hit /= num_users

        return ndcg, hit
