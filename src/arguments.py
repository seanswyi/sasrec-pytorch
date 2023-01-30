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
        '--num_layer',
        default=1,
    )

    # Data arguments.
    parser.add_argument(
        '--max_seq_len',
        default=50,
        type=int,
        help="Maximum number of items to see. Denoted by $n$ in the paper."
    )

    args = parser.parse_args()
    return args
