"""
Wrapper around the OFFICIAL MS4N implementation (Saadatmand et al.), cloned
into external/ms4n_official/. This supersedes models/ms4n_reconstructed.py
(my earlier from-the-paper reconstruction) now that the real code is available.

This wrapper keeps MS4N plugged into the SAME pipeline as gru.py, informer.py,
and mamba.py -- your own aeon-based data loading and training loop -- rather
than switching to the official repo's own loader/trainer. See the note at the
bottom of this file for the one important trade-off that choice involves.
"""

import os
import sys

import torch
import torch.nn as nn

from monster.monster_utils import get_dataloaders, run_training

# Make the cloned repo's `model` package importable as `ms4n_model` without
# colliding with this project's own `models` package (note: singular vs
# plural -- the official repo's folder is genuinely named `model`).
#
# The repo may be cloned under either name depending on how you ran `git
# clone` -- `external/ms4n_official` (the name used in setup instructions)
# or `external/MS4N` (git's default, taken from the repo name itself). This
# checks both so a plain `git clone ... external/MS4N` (no rename) still
# works without editing this file.
_EXTERNAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external")
_CANDIDATE_NAMES = ["ms4n_official", "MS4N"]
_MS4N_REPO_PATH = None
for _name in _CANDIDATE_NAMES:
    _candidate = os.path.join(_EXTERNAL_DIR, _name)
    if os.path.isdir(os.path.join(_candidate, "model")):
        _MS4N_REPO_PATH = _candidate
        break

if _MS4N_REPO_PATH is None:
    raise ImportError(
        f"Could not find the official MS4N repo under {_EXTERNAL_DIR} "
        f"(looked for: {', '.join(_CANDIDATE_NAMES)}). Clone it with:\n"
        "    mkdir -p external\n"
        "    git clone https://github.com/h-saadatmand/MS4N.git external/ms4n_official\n"
        "(or external/MS4N, either name is fine) and install its one extra "
        "dependency:\n"
        "    pip install einops"
    )

if _MS4N_REPO_PATH not in sys.path:
    sys.path.insert(0, _MS4N_REPO_PATH)

try:
    from model.MS4N import MS4NClassifier as _OfficialMS4NClassifier
except ImportError as e:
    raise ImportError(
        f"Found the repo at {_MS4N_REPO_PATH} but couldn't import model.MS4N "
        "from it. Make sure you've installed its one extra dependency:\n"
        "    pip install einops"
    ) from e


class MS4NClassifier(nn.Module):
    """Thin adapter: builds the official MS4NClassifier from the plain
    (input_size, num_classes) arguments the rest of this project's models
    use, instead of the official repo's config-dict convention.
    """

    def __init__(self, input_size, seq_len, num_classes, d_model=64, d_state=64, num_layers=1, dropout=0.1):
        super().__init__()
        # The official MS4NClassifier expects a config dict shaped like what
        # their own utils.trainer.trainer() builds: config['Data_shape'] is
        # the *training data's* full shape (N, C, L), and config['num_labels']
        # is the class count. We only need C and L out of Data_shape, so N
        # (the first entry) is set to 1 -- it is never read by the model.
        config = {
            "Data_shape": (1, input_size, seq_len),
            "num_labels": num_classes,
            "dropout": dropout,
        }
        self.model = _OfficialMS4NClassifier(
            config,
            num_classes=num_classes,
            num_layers=num_layers,
            d_model=d_model,
            d_state=d_state,
        )

    def forward(self, x):
        # x arrives as (batch, channels, timepoints) from aeon, matching what
        # the official MS4NClassifier.forward() already expects and permutes
        # internally -- no reshaping needed here.
        return self.model(x)


def run(dataset_name="AudioMNIST", fold=0, input_size=1, num_classes=10,
        seq_len=None, d_model=64, d_state=64, num_layers=1, num_epochs=200, device=None):
    """Loads a MONSTER dataset/fold, builds the OFFICIAL MS4N classifier, and
    trains it via this project's shared training loop (monster_utils.run_training).

    seq_len is required (the official model needs it up front to build its
    config dict) -- main.py passes it through, or infer it once by checking
    a dataset's documented sequence length (see Table 1 in your proposal).

    num_epochs defaults to 200, matching the official repo's own MONSTER
    setting (see external/ms4n_official/main.py) -- override if your
    bias-variance protocol needs a smaller number for faster iteration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MS4N-official] Using device: {device}")

    if seq_len is None:
        raise ValueError(
            "seq_len is required to build the official MS4NClassifier's config. "
            "Pass it explicitly, e.g. run(..., seq_len=600) for InsectSound."
        )

    train_loader, test_loader = get_dataloaders(dataset_name, fold=fold)

    model = MS4NClassifier(
        input_size=input_size,
        seq_len=seq_len,
        num_classes=num_classes,
        d_model=d_model,
        d_state=d_state,
        num_layers=num_layers,
    ).to(device)

    return run_training(model, train_loader, test_loader, device, num_epochs=num_epochs)


# ------------------------------------------------------------------------
# IMPORTANT TRADE-OFF, READ BEFORE TRUSTING RESULTS AGAINST THE PAPER:
#
# The official repo's own utils/data_loader.py does per-channel normalization
# and carves out a validation split for early stopping (see split_data() and
# trainer() in external/ms4n_official/utils/). This wrapper does NOT do that
# -- it reuses your own aeon-based get_dataloaders(), which has no
# normalization step and no validation split, to keep MS4N on the exact same
# footing as your GRU/Informer/Mamba comparisons.
#
# This is the right choice for a FAIR comparison across your four
# architectures (nobody gets preferential data treatment), but it means you
# should NOT expect to reproduce the paper's exact reported accuracy numbers
# with this wrapper -- those numbers were produced with normalization + early
# stopping this pipeline intentionally omits. If you want to check your MS4N
# implementation matches the paper's, run their own main.py directly instead
# (see external/ms4n_official/main.py), separately from your comparison.
# ------------------------------------------------------------------------