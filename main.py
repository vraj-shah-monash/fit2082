"""
Entry point for training a model on a MONSTER dataset.

Usage:
    python main.py --model gru --dataset AudioMNIST --fold 0
    python main.py --model informer --dataset AudioMNIST-DS --fold 0 --epochs 15

Run from this directory (the parent of models/ and monster/) so the
package imports resolve correctly.
"""

import argparse

from models import gru, informer, ms4n

MODEL_RUNNERS = {
    "gru": gru.run,
    "informer": informer.run,
    #"mamba": mamba.run,
    "ms4n": ms4n.run,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on a MONSTER dataset.")
    parser.add_argument("--model", choices=MODEL_RUNNERS.keys(), required=True,
                         help="Which architecture to train.")
    parser.add_argument("--dataset", default="AudioMNIST",
                         help="MONSTER dataset name, e.g. AudioMNIST, FruitFlies, LenDB.")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold (0-4).")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for this dataset.")
    parser.add_argument("--input_size", type=int, default=1, help="Number of channels for this dataset.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--seq_len", type=int, default=None,
                         help="Sequence length (timepoints per example). Required for --model ms4n.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_fn = MODEL_RUNNERS[args.model]

    kwargs = dict(
        dataset_name=args.dataset,
        fold=args.fold,
        input_size=args.input_size,
        num_classes=args.num_classes,
        num_epochs=args.epochs,
    )

    if args.model == "ms4n":
        if args.seq_len is None:
            raise SystemExit("--seq_len is required for --model ms4n, e.g. --seq_len 600 for InsectSound.")
        kwargs["seq_len"] = args.seq_len

    run_fn(**kwargs)


if __name__ == "__main__":
    main()