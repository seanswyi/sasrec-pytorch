import os
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_positive2negatives, pad_or_truncate_seq


User = str
Item = str
InputSequences = torch.Tensor
PositiveSamples = torch.Tensor
NegativeSamples = torch.Tensor
ItemIdxs = torch.Tensor


class Dataset:
    def __init__(self,
                 batch_size: int,
                 max_seq_len: int,
                 data_root: str,
                 data_filepath: str,
                 debug: bool):
        self.debug = debug
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.data_root = data_root
        self.data_filepath = data_filepath

        self.data = self.load_data(data_filepath=self.data_filepath)

        if self.debug:
            self.data = self.data[:100]

        self.user2items, self.item2users = self.create_mappings(data=self.data)
        self.num_users = len(self.user2items)

        if self.debug:
            self.num_items = 57289
        else:
            self.num_items = len(self.item2users)

        self.positive2negatives = get_positive2negatives(self.num_items)

        splits = self.create_train_valid_test(user2items=self.user2items)
        self.user2items_train, self.user2items_valid, self.user2items_test = splits

    def load_data(self, data_filepath: str) -> list[list[User, Item]]:
        """Load and format data."""
        with open(file=data_filepath) as f:
            user_item_pairs = f.readlines()

        user_item_pairs = [pair.strip().split() for pair in user_item_pairs]
        user_item_pairs = [list(map(int, pair)) for pair in user_item_pairs]

        return user_item_pairs

    def create_mappings(self, data: list[list[User, Item]]) -> (dict[User, list[Item]],
                                                                dict[Item, list[User]]):
        """
        Convert the list of [user, item] pairs to a mapping where the users are keys \
            mapped to a list of items.
        """
        user2items = {}
        item2users = {}
        pbar = tqdm(iterable=data,
                    desc="Creating user2items",
                    total=len(data))
        for user, item in pbar:
            try:
                user2items[user].append(item)
            except KeyError:
                user2items[user] = [item]

            try:
                item2users[item].append(user)
            except KeyError:
                item2users[item] = [user]

        return user2items, item2users

    def create_train_valid_test(self,
                                user2items: dict[User, list[Item]]) -> (dict[User, list[Item]],
                                                                        dict[User, list[Item]],
                                                                        dict[User, list[Item]]):
        """
        Makes train/valid/test splits for users and items.
        If a user has interacted with less than three items, we only use that for training.
        Otherwise, the second to last and last items are each used for valid and test sets.
        """
        user2items_train = {}
        user2items_valid = {}
        user2items_test = {}

        pbar = tqdm(iterable=user2items.items(),
                    desc="Getting train/valid/test splits",
                    total=len(user2items))
        for user, items in pbar:
            num_items = len(items)

            if num_items < 3:
                user2items_train[user] = items
                user2items_valid[user] = []
                user2items_test[user] = []
            else:
                user2items_train[user] = items[:-2]

                valid_input_seq = items[:-2]
                valid_label = items[-2]
                user2items_valid[user] = (valid_input_seq, valid_label)

                test_input_seq = valid_input_seq + [valid_label]
                test_label = items[-1]
                user2items_test[user] = (test_input_seq, test_label)

        return user2items_train, user2items_valid, user2items_test

    def collate_fn_train(self, batch: list[list[int]]) -> InputSequences:
        """
        Simple collate function for the DataLoader.
          1. Truncate input seqs that are longer than max_seq_len from the front.
          2. Pad input seqs that are shorter from the front.
          3. Slice the seqs so that the last element is used as the label.
        """
        seq_tensors = []
        for idx, seq in enumerate(batch):
            seq = pad_or_truncate_seq(seq, max_seq_len=self.max_seq_len)
            seq_tensors.append(seq)

        input_seqs = torch.stack(seq_tensors)

        return input_seqs

    def collate_fn_eval(self, batch: list[list[int]]) -> (InputSequences, ItemIdxs):
        """
        Essentially the same thing as collate_fn_train except for evaluation
          we have to take into consideration the positive and negative samples
          we'll be getting the logits for.

        The hidden representations of these samples are matrix multiplied with
          the hidden representations of the input sequence in order to get
          predictions.
        """
        input_seqs = [x[0] for x in batch]
        seq_tensors = []
        for idx, seq in enumerate(input_seqs):
            seq = pad_or_truncate_seq(seq, max_seq_len=self.max_seq_len)
            seq_tensors.append(seq)

        input_seqs = torch.stack(seq_tensors)

        item_idxs = [x[1] for x in batch]
        item_idxs = torch.tensor(item_idxs, dtype=torch.long)

        return (input_seqs, item_idxs)

    def get_dataloader(self,
                       data: dict[User, list[Item]],
                       split: str='train') -> DataLoader:
        """
        Create and return a DataLoader. Not considering users in this setting.

        1. If split == 'train':
             dataset -> list[list[int]]
        2. Elif split in ['valid', 'test']:
             dataset -> list[tuple[list[int], int]]
        """
        dataset = list(data.values())

        if split in ['valid', 'test']:
            shuffle = False
            collate_fn = self.collate_fn_eval

            input_seqs = [x[0] for x in dataset if x != []]
            all_pred_item_idxs = []

            # Get negative samples and append validation to
            #   input sequence for test phase.
            if split == 'valid':
                for user, items in self.user2items_valid.items():
                    if items == []:
                        continue

                    positive_sample = items[1]
                    negative_samples = self.positive2negatives[positive_sample]
                    pred_item_idxs = [positive_sample] + negative_samples
                    all_pred_item_idxs.append(pred_item_idxs)
            elif split == 'test':
                for user, items in self.user2items_test.items():
                    if items == []:
                        continue

                    positive_sample = items[1]
                    negative_samples = self.positive2negatives[positive_sample]
                    pred_item_idxs = [positive_sample] + negative_samples
                    all_pred_item_idxs.append(pred_item_idxs)

            assert len(input_seqs) == len(all_pred_item_idxs)
            dataset = list(zip(input_seqs, all_pred_item_idxs))
        else:
            shuffle = True
            collate_fn = self.collate_fn_train

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                collate_fn=collate_fn)
        return dataloader
