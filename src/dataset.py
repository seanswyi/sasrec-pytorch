import os
import random

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


User = str
Item = str
InputSequences = torch.Tensor
PositiveLabels = torch.Tensor
NegativeLabels = torch.Tensor


class Dataset:
    def __init__(self,
                 batch_size: int,
                 max_seq_len: int,
                 data_root: str,
                 data_filepath: str):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.data_root = data_root
        self.data_filepath = data_filepath

        self.data = self.load_data(data_filepath=self.data_filepath)
        self.user2items, self.item2users = self.create_mappings(data=self.data)
        self.num_users = len(self.user2items)
        self.num_items = len(self.item2users)

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
                user2items_valid[user] = [items[-2]]
                user2items_test[user] = [items[-1]]

        return user2items_train, user2items_valid, user2items_test

    def get_negative_labels(self,
                            positive_labels: PositiveLabels,
                            num_samples: int=1) -> NegativeLabels:
        """
        `seen` refers to the positive label. Sampling happens as following:
          1. Until we've reached our desired number of samples:
            1.1 Get sample candidates by excluding `seen` from the range.
            1.2 Once a new negative label is sampled, update `seen` and continue.
          2. Return negative sample sequences once all are done.
        """
        negative_labels = []
        for positive_label in tqdm(positive_labels):
            seen = [positive_label]
            count = 0
            while count < num_samples:
                candidates = [idx for idx in range(1, self.num_items + 1) if idx not in seen]
                negative_label = random.choice(candidates)

                while negative_label in seen:
                    negative_label = random.choice(candidates)

                seen.append(negative_label)
                count += 1
            negative_labels.append(seen[1:])

        negative_labels = torch.tensor(negative_labels)
        return negative_labels

    def collate_fn_train(self, batch: list[list[int]]) -> (InputSequences,
                                                           PositiveLabels,
                                                           NegativeLabels):
        """
        Simple collate function for the DataLoader.
          1. Truncate input sequences that are longer than max_seq_len from the front.
          2. Pad input sequences that are shorter from the front.
          3. Slice the sequences so that the last element is used as the label.
        """
        sequence_tensors = []
        for idx, sequence in enumerate(batch):
            sequence = torch.tensor(sequence)
            if len(sequence) > self.max_seq_len:
                sequence = sequence[-self.max_seq_len:]
            else:
                diff = self.max_seq_len - len(sequence)
                sequence = F.pad(sequence, pad=(diff, 0))

            sequence_tensors.append(sequence)

        sequences = torch.stack(sequence_tensors)

        inputs = sequences[:, :-1]
        positive_labels = sequences[:, -1]
        negative_labels = self.get_negative_labels(positive_labels)

        return (inputs, positive_labels, negative_labels)

    def get_dataloader(self,
                       data: dict[User, list[Item]],
                       shuffle: bool=True) -> DataLoader:
        """Create and return a DataLoader. Not considering users in this setting."""
        item_sequences = list(data.values())

        batch = item_sequences[:4]
        sequence_tensors = []
        for idx, sequence in enumerate(batch):
            sequence = torch.tensor(sequence)
            if len(sequence) > self.max_seq_len:
                sequence = sequence[-self.max_seq_len:]
            else:
                diff = self.max_seq_len - len(sequence)
                sequence = F.pad(sequence, pad=(diff, 0))

            sequence_tensors.append(sequence)

        sequences = torch.stack(sequence_tensors)

        import pdb; pdb.set_trace()
        dataloader = DataLoader(dataset=item_sequences,
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                collate_fn=self.collate_fn_train)
        return dataloader
