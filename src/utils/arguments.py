import argparse
import os

from utils import get_device


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Determines whether we save our results or not.",
    )
    parser.add_argument(
        "--log_dir",
        default="../logs",
        type=str,
        help="Directory to save logging files.",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed for deterministic training.",
    )
    parser.add_argument(
        "--resume_dir",
        default="",
        type=str,
        help="Output name from which to resume training.",
    )

    # Dataset arguments.
    parser.add_argument(
        "--max_seq_len",
        default=50,
        type=int,
        help="Maximum number of items to see. Denoted by $n$ in the paper.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size for batching data.",
    )
    parser.add_argument(
        "--data_root",
        default="../data",
        type=str,
        help="Root directory containing _all_ datasets.",
    )
    parser.add_argument(
        "--data_filename",
        default="amazon_beauty.txt",
        type=str,
        choices=[
            "amazon_beauty.txt",
            "amazon_games.txt",
            "steam.txt",
            "movie-lens_1m.txt",
        ],
        help="Name of data file.",
    )

    # Model arguments.
    parser.add_argument(
        "--hidden_dim",
        default=50,
        type=int,
        help="Dimensionality of embedding matrix.",
    )
    parser.add_argument(
        "--num_blocks",
        default=2,
        help="Number of self-attention -> FFNN blocks to stack.",
    )
    parser.add_argument(
        "--dropout_p",
        default=0.5,
        type=float,
        help="Dropout rate applied to embedding layer and FFNN.",
    )
    parser.add_argument(
        "--share_item_emb",
        action="store_true",
        default=False,
        help="Whether or not to use item matrix for prediction layer.",
    )

    # Optimizer arguments.
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        type=float,
        help="Beta1 argument for Adam optimizer.",
    )
    parser.add_argument(
        "--beta2",
        default=0.999,
        type=float,
        help="Beta2 argument for Adam optimizer.",
    )
    parser.add_argument(
        "--eps",
        default=1e-8,
        type=float,
        help="Epsilon value for Adam optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay rate for Adam optimizer.",
    )

    # Trainer arguments.
    parser.add_argument(
        "--device",
        default="",
        type=str,
        help="Device to use.",
    )
    parser.add_argument(
        "--evaluate_k",
        default=10,
        type=int,
        help="nDCG@k, Hit@k, etc.",
    )
    parser.add_argument(
        "--num_epochs",
        default=2000,
        type=int,
        help="Number of epochs to train the model.",
    )
    parser.add_argument(
        "--early_stop_epoch",
        default=20,
        type=int,
        help="Number of epochs to stop early after.",
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        default=False,
        help="Using the scheduler doesn't always help.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.05,
        type=float,
        help="Ratio to determine number of warmup steps.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="onecycle",
        type=str,
        help="Determines the type of scheduler to use.",
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        default=False,
        help="Option on whether or not to resume training.",
    )
    parser.add_argument(
        "--output_dir",
        default="../outputs",
        type=str,
        help="Directory to save all results and artifacts.",
    )

    # MLflow arguments.
    parser.add_argument(
        "--mlflow_experiment",
        default="sasrec-pytorch-experiments",
        type=str,
        help="Name of MLflow experiment.",
    )
    parser.add_argument(
        "--mlflow_run_name",
        default="",
        type=str,
        help="Name for MLflow run within the experiment.",
    )

    args = parser.parse_args()

    args.device = get_device()

    return args


class DatasetArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        args.data_root = os.path.expanduser(path=args.data_root)

        if not os.getcwd().endswith("src") and os.getcwd().endswith("sasrec-pytorch"):
            args.data_root = "./data"

        self.data_filepath = os.path.join(args.data_root, args.data_filename)

        assert os.path.exists(
            self.data_filepath
        ), f"{self.data_filepath} does not exist!"

        self.batch_size = args.batch_size
        self.max_seq_len = args.max_seq_len
        self.debug = args.debug


class ModelArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = args.device
        self.num_items = args.num_items
        self.num_blocks = args.num_blocks
        self.max_seq_len = args.max_seq_len
        self.hidden_dim = args.hidden_dim
        self.dropout_p = args.dropout_p
        self.share_item_emb = args.share_item_emb


class OptimizerArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.eps = args.eps
        self.weight_decay = args.weight_decay


class TrainerArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = args.device
        self.evaluate_k = args.evaluate_k
        self.max_lr = args.lr
        self.num_epochs = args.num_epochs
        self.early_stop_epoch = args.early_stop_epoch
        self.use_scheduler = args.use_scheduler
        self.warmup_ratio = args.warmup_ratio
        self.scheduler_type = args.scheduler_type
        self.resume_training = args.resume_training
        self.save_dir = args.save_dir
