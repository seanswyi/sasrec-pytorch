import argparse
from datetime import datetime
import json
import logging
import os
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import init
from torch import optim

from utils import (get_args,
                   get_device,
                   get_output_name,
                   DatasetArgs,
                   DotDict,
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

    torch.manual_seed(args.random_seed)

    # Get timestamp.
    time_right_now = time.time()
    timestamp = datetime.fromtimestamp(timestamp=time_right_now).strftime(format='%m-%d-%Y-%H%M')
    args.timestamp = timestamp

    data_name = args.data_filename.split('.txt')[0]

    # If we're resuming training, we have to set up our args and log file accordingly.
    if args.resume_training:
        if args.resume_dir:
            resume_dir = args.resume_dir

            args_save_filename = os.path.join(args.output_dir, args.resume_dir, 'args.json')
            with open(file=args_save_filename) as f:
                args = json.load(fp=f)

            args = DotDict(args)

            args.log_filename = os.path.join(args.log_dir, f'{args.resume_dir}.log')
            args.save_dir = resume_dir
            args.resume_dir = resume_dir
        else:
            relevant_files = [f for f in os.listdir(args.output_dir) if data_name in f]
            timestamps = [f.split('_')[-1] for f in relevant_files]
            timestamp_objs = [datetime.strptime(ts, '%m-%d-%Y-%H%M').timestamp() for ts in timestamps]

            most_recent_ts_idx = np.argmax(timestamp_objs)
            resume_dir = relevant_files[most_recent_ts_idx]

            args_save_filename = os.path.join(args.output_dir, resume_dir, 'args.json')
            with open(file=args_save_filename) as f:
                args = json.load(fp=f)

            args = DotDict(args)

            args.log_filename = os.path.join(args.log_dir, f'{resume_dir}.log')
            args.save_dir = os.path.join(args.output_dir, resume_dir)
            args.resume_dir = resume_dir

    # Get log file information.
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    output_name = get_output_name(args, timestamp)
    log_filename = f'{output_name}.log'
    args.log_filename = os.path.join(args.log_dir, log_filename)

    # Create save file.
    args.save_name = output_name
    args.save_dir = os.path.join(args.output_dir, output_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    args_save_filename = os.path.join(args.save_dir, 'args.json')
    with open(file=args_save_filename, mode='w') as f:
        json.dump(obj=vars(args), fp=f, indent=2)

    # Logging basic configuration.
    log_msg_format = '[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d] %(message)s'
    handlers = [logging.FileHandler(filename=args.log_filename), logging.StreamHandler()]
    logging.basicConfig(format=log_msg_format,
                        level=logging.INFO,
                        handlers=handlers)
    logger.info(f"Starting main process with {data_name}...")
    logger.info(f"Output directory set to {args.save_dir}")

    if args.debug:
        args.num_epochs = 1

    dataset_args = DatasetArgs(args)
    dataset = Dataset(**vars(dataset_args))

    args.num_items = dataset.num_items
    model_args = ModelArgs(args)
    model = SASRec(**vars(model_args))

    for name, param in model.named_parameters():
        try:
            init.xavier_uniform_(param.data)
        except ValueError:
            continue

    model = model.to(args.device)

    optimizer_args = OptimizerArgs(args)
    optimizer = optim.Adam(params=model.parameters(),
                           **vars(optimizer_args))

    trainer_args = TrainerArgs(args)
    trainer = Trainer(dataset=dataset,
                      model=model,
                      optimizer=optimizer,
                      **vars(trainer_args))

    best_results = trainer.train()
    best_ndcg_epoch, best_model_state_dict, best_optim_state_dict = best_results

    # Perform test.
    model.load_state_dict(best_model_state_dict)
    logger.info(f"Testing with model checkpoint from epoch {best_ndcg_epoch}...")
    test_ndcg, test_hit_rate = trainer.evaluate(mode='test', model=model)

    test_ndcg_msg = f"Test nDCG@{trainer_args.evaluate_k} is {test_ndcg: 0.6f}."
    test_hit_msg = f"Test Hit@{trainer_args.evaluate_k} is {test_hit_rate: 0.6f}."
    test_result_msg = '\n'.join([test_ndcg_msg, test_hit_msg])
    logger.info(f"\n{test_result_msg}")


if __name__ == '__main__':
    main()
