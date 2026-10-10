"""
Runs a model across ALL folds, repeating each fold with multiple random
seeds, and computes an empirical bias/variance decomposition from the
per-example predictions -- implementing the "5 seeds x 5 folds = 25 runs
per architecture" protocol.

Usage:
    python bias_variance.py --model gru --dataset FordChallenge --input_size 30 --num_classes 2 --epochs 200 --seeds 5
    python bias_variance.py --model ms4n --dataset InsectSound --seq_len 600 --epochs 200 --seeds 5 --folds 0,1,2,3,4

Why this matters: bias and variance CANNOT be computed from accuracy alone
(two models can have identical accuracy via completely different mixes of
bias vs. variance). This script saves every test example's PREDICTED LABEL
on every run, not just the final accuracy -- that per-example detail is
the actual ingredient the decomposition formula needs. See the comments in
bias_variance_decomposition() for the exact formula.
"""

import argparse
import os
import numpy as np
import torch

from models import gru, informer, mamba, ms4n
from monster.monster_utils import get_dataloaders, predict_all, run_training

MODEL_MODULES = {"gru": gru, "informer": informer, "mamba": mamba, "ms4n": ms4n}


def build_model(model_name, input_size, num_classes, seq_len, device):
    """Builds the requested (untrained) model. seq_len is only used by MS4N,
    which needs it up front to construct its config dict.
    """
    if model_name == "gru":
        model = gru.GRUClassifier(input_size=input_size, hidden_size=64, num_classes=num_classes)
    elif model_name == "informer":
        model = informer.InformerClassifier(input_size=input_size, num_classes=num_classes)
    elif model_name == "mamba":
        model = mamba.MambaClassifier(input_size=input_size, num_classes=num_classes)
    elif model_name == "ms4n":
        if seq_len is None:
            raise ValueError("--seq_len is required for --model ms4n.")
        model = ms4n.MS4NClassifier(input_size=input_size, seq_len=seq_len, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def bias_variance_decomposition(all_run_preds, y_true):
    """
    all_run_preds: array of shape (num_seeds, num_test_examples) -- each row
                   is one seed's predicted labels, in fixed test-set order.
    y_true: array of shape (num_test_examples,) -- the true labels.

    Returns per-example bias and variance arrays (not yet averaged), plus
    per-run accuracies -- the per-example arrays are what let us correctly
    pool results across folds afterward (see run_full_protocol below),
    rather than averaging an already-averaged number.

    Formula (standard practical bias/variance decomposition for 0-1 loss):
    - main_pred(x) = the most common predicted label for x, across seeds
      (this approximates the model's "expected" prediction)
    - bias(x)     = 1 if main_pred(x) != true_label(x), else 0
    - variance(x) = fraction of seeds whose prediction for x disagreed
                    with main_pred(x)
    """
    num_seeds, num_examples = all_run_preds.shape

    main_preds = np.zeros(num_examples, dtype=int)
    for i in range(num_examples):
        values, counts = np.unique(all_run_preds[:, i], return_counts=True)
        main_preds[i] = values[np.argmax(counts)]

    bias_per_example = (main_preds != y_true).astype(float)
    variance_per_example = (all_run_preds != main_preds[None, :]).mean(axis=0)

    per_run_accuracy = (all_run_preds == y_true[None, :]).mean(axis=1)

    return bias_per_example, variance_per_example, per_run_accuracy


def run_one_fold(model_name, dataset_name, fold, input_size, num_classes, seq_len, num_epochs, num_seeds, device):
    """Trains `num_seeds` independently-seeded models on one fold, and
    returns that fold's per-example bias/variance arrays plus the raw
    per-run accuracies (for the quick mean/std sanity check).
    """
    all_run_preds = []
    y_true = None

    for seed in range(num_seeds):
        print(f"\n  --- Fold {fold}, seed {seed} ---")
        torch.manual_seed(seed)

        train_loader, test_loader = get_dataloaders(dataset_name, fold=fold)
        model = build_model(model_name, input_size, num_classes, seq_len, device)
        run_training(model, train_loader, test_loader, device, num_epochs=num_epochs)

        preds = predict_all(model, test_loader, device)
        all_run_preds.append(preds)

        if y_true is None:
            y_true = np.concatenate([y.numpy() for _, y in test_loader])

    all_run_preds = np.stack(all_run_preds)  # (num_seeds, num_examples_in_this_fold)
    bias_pe, variance_pe, per_run_acc = bias_variance_decomposition(all_run_preds, y_true)
    return bias_pe, variance_pe, per_run_acc


def fold_path(results_dir, model_name, dataset_name, fold):
    return os.path.join(results_dir, f"{model_name}_{dataset_name}_fold{fold}.npz")


def save_fold(path, bias_pe, variance_pe, per_run_acc, num_epochs, num_seeds):
    """Saves one finished fold so it survives a crash or a Colab disconnect."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, bias_pe=bias_pe, variance_pe=variance_pe, per_run_acc=per_run_acc,
             epochs=num_epochs, seeds=num_seeds)


def load_fold(path, num_epochs, num_seeds):
    """Returns a fold's saved results, or None if there is no file or it was
    produced with different epochs/seeds (so results from different settings
    are never silently mixed together)."""
    if not os.path.exists(path):
        return None
    d = np.load(path)
    if int(d["epochs"]) != num_epochs or int(d["seeds"]) != num_seeds:
        print(f"  (ignoring {path}: saved with epochs={int(d['epochs'])}, seeds={int(d['seeds'])}; "
              f"this run uses epochs={num_epochs}, seeds={num_seeds})")
        return None
    return d["bias_pe"], d["variance_pe"], d["per_run_acc"]


def run_full_protocol(model_name, dataset_name, folds, input_size, num_classes, seq_len, num_epochs, num_seeds, device,
                      results_dir="results", force=False):
    """Runs every fold x seed combination (e.g. 5 folds x 5 seeds = 25 runs)
    and pools the per-example bias/variance arrays across ALL folds into a
    single architecture-level bias and variance score. Pooling is valid
    here because every test example across the 5 folds is a DIFFERENT
    example (k-fold CV splits the dataset, it doesn't resample it), so
    concatenating per-example results across folds is equivalent to
    computing bias/variance over the full dataset.
    """
    per_fold_results = {}
    all_bias_pe = []
    all_variance_pe = []
    all_run_accs = {}

    for fold in folds:
        print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
        path = fold_path(results_dir, model_name, dataset_name, fold)
        saved = None if force else load_fold(path, num_epochs, num_seeds)
        if saved is not None:
            bias_pe, variance_pe, per_run_acc = saved
            print(f"  Fold {fold} already finished -- loaded from {path} (use --force to retrain)")
        else:
            bias_pe, variance_pe, per_run_acc = run_one_fold(
                model_name, dataset_name, fold, input_size, num_classes, seq_len, num_epochs, num_seeds, device,
            )
            save_fold(path, bias_pe, variance_pe, per_run_acc, num_epochs, num_seeds)
        print(f"\n  >>> FOLD {fold} DONE: mean_acc={per_run_acc.mean():.4f}  std_acc={per_run_acc.std():.4f}  "
              f"bias={bias_pe.mean():.4f}  variance={variance_pe.mean():.4f}")
        per_fold_results[fold] = {
            "bias": bias_pe.mean(),
            "variance": variance_pe.mean(),
            "mean_acc": per_run_acc.mean(),
            "std_acc": per_run_acc.std(),
            "per_run_acc": per_run_acc,
        }
        all_bias_pe.append(bias_pe)
        all_variance_pe.append(variance_pe)
        all_run_accs[fold] = per_run_acc

    pooled_bias = np.concatenate(all_bias_pe).mean()
    pooled_variance = np.concatenate(all_variance_pe).mean()
    all_25_accuracies = np.concatenate(list(all_run_accs.values()))

    return per_fold_results, pooled_bias, pooled_variance, all_25_accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_MODULES.keys(), required=True)
    parser.add_argument("--dataset", default="InsectSound")
    parser.add_argument("--folds", default="0,1,2,3,4",
                         help="Comma-separated fold indices, e.g. '0,1,2,3,4' (default: all 5) or '0' for just one.")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=None, help="Required for --model ms4n.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5, help="Number of repeated runs PER FOLD.")
    parser.add_argument("--results_dir", default="results",
                         help="Where finished folds are saved. Re-running skips folds already saved here.")
    parser.add_argument("--force", action="store_true", help="Retrain folds even if a saved result exists.")
    args = parser.parse_args()

    folds = [int(f) for f in args.folds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_runs = len(folds) * args.seeds
    print(f"Running {args.model} on {args.dataset}: {len(folds)} folds x {args.seeds} seeds "
          f"= {total_runs} total training runs.")

    per_fold_results, pooled_bias, pooled_variance, all_accuracies = run_full_protocol(
        args.model, args.dataset, folds, args.input_size, args.num_classes,
        args.seq_len, args.epochs, args.seeds, device,
        results_dir=args.results_dir, force=args.force,
    )

    print("\n" + "=" * 60)
    print(f"PER-FOLD RESULTS -- {args.model} on {args.dataset}")
    print("=" * 60)
    for fold, r in per_fold_results.items():
        print(f"Fold {fold}: mean_acc={r['mean_acc']:.4f}  std_acc={r['std_acc']:.4f}  "
              f"bias={r['bias']:.4f}  variance={r['variance']:.4f}")

    print("\n" + "=" * 60)
    print(f"OVERALL (pooled across {len(folds)} folds, {total_runs} total runs)")
    print("=" * 60)
    print(f"Mean accuracy:  {all_accuracies.mean():.4f}")
    print(f"Std of accuracy (across all {total_runs} runs): {all_accuracies.std():.4f}")
    print(f"Bias:           {pooled_bias:.4f}  (fraction of examples the 'main' prediction gets wrong)")
    print(f"Variance:       {pooled_variance:.4f}  (fraction of individual runs disagreeing with the main prediction)")


if __name__ == "__main__":
    main()