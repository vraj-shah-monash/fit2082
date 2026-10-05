"""
Runs a model multiple times (different random seeds, same fold) and computes
an empirical bias/variance decomposition from the per-example predictions.

Usage:
    python bias_variance.py --model gru --dataset InsectSound --fold 0 --seeds 5 --epochs 20

This is the project's actual core measurement -- not just accuracy.
"""

import argparse
import numpy as np
import torch

from models import gru, informer, ms4n #, mamba
from monster.monster_utils import get_dataloaders, predict_all

MODEL_MODULES = {"gru": gru, "informer": informer, "mamba": mamba, "ms4n": ms4n}

# Each model module needs a matching *_build(...) helper below, since run()
# does load+train+evaluate all at once and we need the trained model object
# back out to call predict_all() ourselves.


def build_and_train(model_name, dataset_name, fold, input_size, num_classes, num_epochs, device):
    """Loads data, builds the requested model, trains it, and returns the
    trained model plus the test loader (so we can extract per-example
    predictions afterwards).
    """
    train_loader, test_loader = get_dataloaders(dataset_name, fold=fold)

    if model_name == "gru":
        model = gru.GRUClassifier(input_size=input_size, hidden_size=64, num_classes=num_classes).to(device)
    elif model_name == "informer":
        model = informer.InformerClassifier(input_size=input_size, num_classes=num_classes).to(device)
    elif model_name == "mamba":
        model = mamba.MambaClassifier(input_size=input_size, num_classes=num_classes).to(device)
    elif model_name == "ms4n":
        model = ms4n.MS4NClassifier(input_size=input_size, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    from monster.monster_utils import run_training
    run_training(model, train_loader, test_loader, device, num_epochs=num_epochs)
    return model, test_loader


def bias_variance_decomposition(all_run_preds, y_true):
    """
    all_run_preds: array of shape (num_seeds, num_test_examples) -- each row
                   is one seed's predicted labels, in fixed test-set order.
    y_true: array of shape (num_test_examples,) -- the true labels.

    Returns (bias, variance, mean_accuracy) using the standard practical
    decomposition: for each example, the "main prediction" is the most
    common label predicted across seeds. Bias = how often that main
    prediction is wrong. Variance = how often individual runs disagree
    with the main prediction (regardless of correctness).
    """
    num_seeds, num_examples = all_run_preds.shape

    main_preds = np.zeros(num_examples, dtype=int)
    for i in range(num_examples):
        values, counts = np.unique(all_run_preds[:, i], return_counts=True)
        main_preds[i] = values[np.argmax(counts)]

    bias_per_example = (main_preds != y_true).astype(float)
    bias = bias_per_example.mean()

    variance_per_example = (all_run_preds != main_preds[None, :]).mean(axis=0)
    variance = variance_per_example.mean()

    per_run_accuracy = (all_run_preds == y_true[None, :]).mean(axis=1)
    mean_accuracy = per_run_accuracy.mean()
    std_accuracy = per_run_accuracy.std()

    return bias, variance, mean_accuracy, std_accuracy, per_run_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_MODULES.keys(), required=True)
    parser.add_argument("--dataset", default="InsectSound")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5, help="Number of repeated runs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {args.seeds} seeds of {args.model} on {args.dataset} fold {args.fold}...")

    all_run_preds = []
    y_true = None

    for seed in range(args.seeds):
        print(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed)

        model, test_loader = build_and_train(
            args.model, args.dataset, args.fold,
            args.input_size, args.num_classes, args.epochs, device,
        )
        preds = predict_all(model, test_loader, device)
        all_run_preds.append(preds)

        if y_true is None:
            y_true = np.concatenate([y.numpy() for _, y in test_loader])

    all_run_preds = np.stack(all_run_preds)  # (num_seeds, num_examples)

    bias, variance, mean_acc, std_acc, per_run_acc = bias_variance_decomposition(all_run_preds, y_true)

    print("\n" + "=" * 50)
    print(f"Results over {args.seeds} seeds, {args.model} on {args.dataset} fold {args.fold}")
    print("=" * 50)
    print(f"Per-run accuracies: {np.round(per_run_acc, 4)}")
    print(f"Mean accuracy:      {mean_acc:.4f}")
    print(f"Std of accuracy:    {std_acc:.4f}  (quick-and-dirty variance signal)")
    print(f"Bias:               {bias:.4f}  (fraction of examples the 'main' prediction gets wrong)")
    print(f"Variance:           {variance:.4f}  (fraction of individual runs disagreeing with the main prediction)")


if __name__ == "__main__":
    main()