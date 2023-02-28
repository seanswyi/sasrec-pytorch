import argparse
from datetime import datetime
import logging
import os
import time

import torch
from torch.nn import functional as F
from torch import optim

from utils import (get_args,
                   get_device,
                   DatasetArgs,
                   ModelArgs,
                   OptimizerArgs,
                   TrainerArgs)
from dataset import Dataset
from model import SASRec
from trainer import Trainer


logger = logging.getLogger()


def main() -> None:
    args = get_args()
    args.device = get_device()

    # Get timestamp.
    time_right_now = time.time()
    timestamp = datetime.fromtimestamp(timestamp=time_right_now).strftime(format='%m-%d-%Y-%H%M')
    args.timestamp = timestamp

    # Get log file information.
    data_name = args.data_filename.split('.txt')[0]
    log_filename = f"sasrec-{data_name}_lr-{args.lr}_batch-size-{args.batch_size}\
        _num-epochs-{args.num_epochs}_{timestamp}.log"
    args.log_filename = log_filename
    if not os.path.exists('../logs'):
        os.makedirs('../logs', exist_ok=True)
    args.log_filename = os.path.join('../logs', args.log_filename)

    # Logging basic configuration.
    log_msg_format = '[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d] %(message)s'
    handlers = [logging.FileHandler(filename=args.log_filename), logging.StreamHandler()]
    logging.basicConfig(format=log_msg_format,
                        level=logging.INFO,
                        handlers=handlers)

    logger.info(f"Starting main process with {data_name}...")

    if args.debug:
        args.num_epochs = 1

    dataset_args = DatasetArgs(args)
    dataset = Dataset(**vars(dataset_args))

    args.num_items = dataset.num_items
    model_args = ModelArgs(args)
    model = SASRec(**vars(model_args))
    model = model.to(args.device)

    optimizer_args = OptimizerArgs(args)
    optimizer = optim.Adam(params=model.parameters(),
                           **vars(optimizer_args))

    trainer_args = TrainerArgs(args)
    trainer = Trainer(dataset=dataset,
                      model=model,
                      optimizer=optimizer,
                      **vars(trainer_args))

    best_model_state_dict = trainer.train()


if __name__ == '__main__':
    main()
