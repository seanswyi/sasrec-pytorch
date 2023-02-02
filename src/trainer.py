class Trainer:
    def __init__(self, args, dataset):
        self.args = args

        self.train_data = dataset.user2items_train
        self.valid_data = dataset.user2items_valid
        self.test_data = dataset.user2items_test

    def collate_fn(self, batch):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass