import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # File arguments.
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

    # Data arguments.
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

    # Training arguments.
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help="Learning rate."
    )

    args = parser.parse_args()
    return args
