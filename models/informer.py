"""
Informer-style transformer for time series classification on MONSTER datasets.

NOTE: this uses standard multi-head self-attention (nn.TransformerEncoderLayer)
as a stand-in for Informer's ProbSparse attention, which is its actual
architectural contribution (a sparse approximation of attention, designed for
efficiency on long sequences). Standard attention has quadratic memory cost
in sequence length, so this will struggle on long series (e.g. AudioMNIST's
47,998 timepoints) without downsampling first.

To use the literal ProbSparse attention instead, clone the official repo:
    git clone https://github.com/zhouhaoyi/Informer2020
and import from it:
    from models.attn import ProbAttention, AttentionLayer
then swap it in for nn.MultiheadAttention inside the encoder below.
Worth checking with your supervisor whether this simplified version is
acceptable for your project's scope, or whether the literal module is expected.
"""

import math

import torch
import torch.nn as nn

from monster.monster_utils import get_dataloaders, run_training


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, since transformers have no
    built-in sense of sequence order (unlike GRU, which processes step by step)."""

    def __init__(self, d_model, max_len=50000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class InformerClassifier(nn.Module):
    def __init__(self, input_size, d_model=64, n_heads=4, num_layers=2, num_classes=10, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)   # project raw channels -> d_model
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x comes in as (batch, channels, timepoints) from aeon
        x = x.permute(0, 2, 1)          # -> (batch, timepoints, channels)
        x = self.input_proj(x)          # -> (batch, timepoints, d_model)
        x = self.pos_encoding(x)
        out = self.encoder(x)           # -> (batch, timepoints, d_model)
        pooled = out.mean(dim=1)        # average over time -> (batch, d_model)
        return self.classifier(pooled)


def run(dataset_name="AudioMNIST-DS", fold=0, input_size=1, num_classes=10,
        d_model=64, n_heads=4, num_layers=2, num_epochs=10, device=None):
    """Loads a MONSTER dataset/fold, builds an Informer-style classifier, and
    trains it. Called from main.py; keeps config as explicit arguments so
    main.py can override per dataset without editing this file.

    Defaults to the downsampled 'AudioMNIST-DS' variant rather than full
    'AudioMNIST', since full self-attention is memory-hungry on long series.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Informer] Using device: {device}")

    train_loader, test_loader = get_dataloaders(dataset_name, fold=fold)

    model = InformerClassifier(
        input_size=input_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)

    return run_training(model, train_loader, test_loader, device, num_epochs=num_epochs)