"""
Mamba classifier for time series classification on MONSTER datasets.

Uses the official mamba_ssm package (Gu & Dao, 2023) rather than a
reimplementation, since Mamba's selective-scan kernels are custom CUDA code
that isn't practical to reproduce from scratch.

Setup (requires an NVIDIA GPU -- the official kernels do not run on CPU):
    pip install causal-conv1d>=1.4.0
    pip install mamba-ssm

Note: per your project scope, Mamba itself is NOT one of your three compared
architectures (that's MS4N, Informer, GRU) -- MS4N's own paper uses Mamba as
a baseline it's shown to outperform. This file exists so you can reproduce
that baseline comparison yourself if useful, not as one of your core results.
"""

import torch
import torch.nn as nn

from monster.monster_utils import get_dataloaders, run_training

try:
    from mamba_ssm import Mamba
except ImportError as e:
    raise ImportError(
        "mamba_ssm is not installed. On a CUDA-enabled machine, run:\n"
        "    pip install causal-conv1d>=1.4.0\n"
        "    pip install mamba-ssm\n"
        "Mamba's kernels require an NVIDIA GPU and will not run on CPU."
    ) from e


class MambaBlock(nn.Module):
    """A single Mamba layer wrapped with residual connection + LayerNorm,
    following the standard pre-norm pattern used in the Mamba paper."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        # x: (batch, length, d_model)
        return x + self.mamba(self.norm(x))


class MambaClassifier(nn.Module):
    def __init__(self, input_size, d_model=64, num_layers=4, num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x comes in as (batch, channels, timepoints) from aeon
        x = x.permute(0, 2, 1)          # -> (batch, timepoints, channels)
        x = self.input_proj(x)          # -> (batch, timepoints, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        pooled = x.mean(dim=1)          # average over time -> (batch, d_model)
        return self.classifier(pooled)


def run(dataset_name="AudioMNIST", fold=0, input_size=1, num_classes=10,
        d_model=64, num_layers=4, num_epochs=10, device=None):
    """Loads a MONSTER dataset/fold, builds a Mamba classifier, and trains it.
    Called from main.py; keeps config as explicit arguments so main.py can
    override per dataset without editing this file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[Mamba] WARNING: no CUDA GPU detected -- mamba_ssm's kernels "
              "require CUDA and this will likely fail or be extremely slow.")
    print(f"[Mamba] Using device: {device}")

    train_loader, test_loader = get_dataloaders(dataset_name, fold=fold)

    model = MambaClassifier(
        input_size=input_size,
        d_model=d_model,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)

    return run_training(model, train_loader, test_loader, device, num_epochs=num_epochs)