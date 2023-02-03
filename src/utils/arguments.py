import argparse
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Dataset arguments.
    parser.add_argument(
        '--data_root',
        default='~/projects/datasets',
        type=str,
        help="Root directory containing _all_ datasets."
    )
    parser.add_argument(
        '--data_name',
        default='amazon',
        type=str,
        help="Specific type of dataset we want to use."
    )
    parser.add_argument(
        '--data_filename',
        default='amazon_beauty.txt',
        type=str,
        help="Name of data file."
    )

    # Model arguments.
    parser.add_argument(
        '--hidden_dim',
        default=50,
        type=int,
        help="Dimensionality of embedding matrix."
    )
    parser.add_argument(
        '--num_blocks',
        default=1,
        help="Number of self-attention -> FFNN blocks to stack."
    )
    parser.add_argument(
        '--dropout_p',
        default=0.5,
        type=float,
        help="Dropout rate applied to embedding layer and FFNN."
    )
    parser.add_argument(
        '--share_item_emb',
        action='store_true',
        default=False,
        help="Whether or not to use item matrix for prediction layer."
    )

    # Trainer arguments.
    parser.add_argument(
        '--max_seq_len',
        default=50,
        type=int,
        help="Maximum number of items to see. Denoted by $n$ in the paper."
    )
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help="Batch size for batching data."
    )

    # Optimizer arguments.
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help="Learning rate."
    )
    parser.add_argument(
        '--beta1',
        default=0.9,
        type=float,
        help="Beta1 argument for Adam optimizer."
    )
    parser.add_argument(
        '--beta2',
        default=0.999,
        type=float,
        help="Beta2 argument for Adam optimizer."
    )
    parser.add_argument(
        '--eps',
        default=1e-8,
        type=float,
        help="Epsilon value for Adam optimizer."
    )
    parser.add_argument(
        '--weight_decay',
        default=0.0,
        type=float,
        help="Weight decay rate for Adam optimizer."
    )

    args = parser.parse_args()
    return args


class DatasetArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        args.data_root = os.path.expanduser(path=args.data_root)
        self.data_root = os.path.join(args.data_root, args.data_name)
        self.data_filepath = os.path.join(self.data_root, args.data_filename)


class ModelArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.num_items = args.num_items
        self.num_blocks = args.num_blocks
        self.max_seq_len = args.max_seq_len
        self.hidden_dim = args.hidden_dim
        self.dropout_p = args.dropout_p
        self.share_item_emb = args.share_item_emb


class OptimizerArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.eps = args.eps
        self.weight_decay = args.weight_decay


class TrainerArgs:
    def __init__(self, args: argparse.Namespace) -> None:
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.warmup_steps = args.warmup_steps
        self.scheduler = args.scheduler
