import os

from tqdm import tqdm

from utils.custom_type_hints import Item, User


class Dataset:
    def __init__(self, args):
        self.args = args

        args.data_root = os.path.expanduser(path=args.data_root)
        self.data_root = os.path.join(args.data_root, args.data_name)
        self.data_filepath = os.path.join(self.data_root, args.data_filename)

        self.data = self.load_data(data_filepath=self.data_filepath)
        self.user2items, self.item2users = self.create_mappings(data=self.data)
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
