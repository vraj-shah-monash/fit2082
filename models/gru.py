"""
GRU baseline for time series classification on MONSTER datasets.

This is the simplest of the three architectures (GRU, Informer, MS4N) since
nn.GRU ships built into PyTorch -- no extra install or repo clone needed.
Get this one working end-to-end first before moving to Informer or MS4N.
"""

import torch
import torch.nn as nn

from monster.monster_utils import get_dataloaders, run_training


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # x comes in as (batch, channels, timepoints) from aeon
        x = x.permute(0, 2, 1)          # -> (batch, timepoints, channels) = what GRU expects
        out, _ = self.gru(x)            # out: (batch, timepoints, hidden*2)
        pooled = out.mean(dim=1)        # average over time -> (batch, hidden*2)
        return self.classifier(pooled)  # -> (batch, num_classes)


def run(dataset_name="AudioMNIST", fold=0, input_size=1, num_classes=10,
        hidden_size=64, num_epochs=10, device=None):
    """Loads a MONSTER dataset/fold, builds a GRU classifier, and trains it.
    Called from main.py; keeps config as explicit arguments so main.py can
    override per dataset without editing this file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GRU] Using device: {device}")

    train_loader, test_loader = get_dataloaders(dataset_name, fold=fold)

    model = GRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=num_classes,
    ).to(device)

    return run_training(model, train_loader, test_loader, device, num_epochs=num_epochs)