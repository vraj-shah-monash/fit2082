"""
Mamba classifier for time series classification on MONSTER datasets.

Uses the `mambapy` package (pip install mambapy) -- a pure-PyTorch
reimplementation (alxndrTL/mamba.py), NOT the official mamba_ssm package.

Why the switch: mamba_ssm requires compiling custom CUDA kernels
(causal-conv1d + mamba-ssm) at install time, which failed outright on a
CPU-only laptop and took 45+ minutes with no success even on a Colab GPU
runtime (likely a version mismatch between Colab's torch/CUDA/Python combo
and the prebuilt wheels causal-conv1d ships). mambapy needs no compilation
at all -- it's plain PyTorch ops -- so it installs in seconds and runs
identically on CPU or GPU.

Trade-off: mambapy will likely be slower per training step than the
official CUDA-kernel version, since it doesn't use Mamba's specialised
selective-scan kernel. That's an acceptable cost for actually being able
to run it reliably, on your laptop or on Colab/Kaggle/M3 without fighting
a build every time.

Setup:
    pip install mambapy
"""

import torch
import torch.nn as nn

from monster.monster_utils import get_dataloaders, run_training

try:
    from mambapy.mamba import Mamba, MambaConfig
except ImportError as e:
    raise ImportError(
        "mambapy is not installed. Run:\n"
        "    pip install mambapy\n"
        "No compilation needed -- this is pure PyTorch and works on CPU or GPU."
    ) from e


class MambaClassifier(nn.Module):
    def __init__(self, input_size, d_model=64, num_layers=4, num_classes=10, d_state=16):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        config = MambaConfig(d_model=d_model, n_layers=num_layers, d_state=d_state)
        self.mamba = Mamba(config)  # mambapy's Mamba already stacks n_layers
                                     # internally with its own residual + norm,
                                     # so no manual block-stacking needed here.

        self.norm_out = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x comes in as (batch, channels, timepoints) from aeon
        x = x.permute(0, 2, 1)          # -> (batch, timepoints, channels)
        x = self.input_proj(x)          # -> (batch, timepoints, d_model)
        x = self.mamba(x)               # -> (batch, timepoints, d_model)
        x = self.norm_out(x)
        pooled = x.mean(dim=1)          # average over time -> (batch, d_model)
        return self.classifier(pooled)


def run(dataset_name="AudioMNIST", fold=0, input_size=1, num_classes=10,
        d_model=64, num_layers=4, d_state=16, num_epochs=10, device=None):
    """Loads a MONSTER dataset/fold, builds a Mamba classifier, and trains it.
    Called from main.py; keeps config as explicit arguments so main.py can
    override per dataset without editing this file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Mamba] Using device: {device}")

    train_loader, test_loader = get_dataloaders(dataset_name, fold=fold)

    model = MambaClassifier(
        input_size=input_size,
        d_model=d_model,
        num_layers=num_layers,
        num_classes=num_classes,
        d_state=d_state,
    ).to(device)

    return run_training(model, train_loader, test_loader, device, num_epochs=num_epochs)