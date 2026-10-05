#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader with configurable validation ratio.

Normalization modes:
1) 'none'        : no normalization

2) 'per_channel' : per-channel non-zero mean/std normalization
"""

import os
import numpy as np
import logging
from sklearn import model_selection
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# ==============================
# Cache directory
# ==============================
# Override with MONSTER_CACHE_DIR; falls back to the standard HF cache
# location instead of a machine-specific path.
CACHE_DIR = os.environ.get(
    'MONSTER_CACHE_DIR',
    os.path.expanduser("~/.cache/huggingface")
)

#CACHE_DIR = os.path.expanduser("~/hz18_scratch2/hassan/huggingface_cache")

# ==============================
# Normalization utilities
# ==============================

def compute_normalization_stats(data):
    """Compute global mean/std from non-zero values only."""
    mask = data != 0
    non_zero = data[mask]

    if len(non_zero) == 0:
        logger.warning("No non-zero values found for global normalization")
        return 0.0, 1.0

    mean = non_zero.mean()
    std = non_zero.std()
    if std < 1e-8:
        std = 1.0

    logger.info(f"[Global norm] mean={mean:.6f}, std={std:.6f}")
    return mean, std


def normalize_data(data, mean, std):
    """Apply global normalization, keep zeros unchanged."""
    data_norm = data.copy()
    mask = data != 0
    data_norm[mask] = (data[mask] - mean) / std
    return data_norm


def compute_per_channel_stats(data):
    """Compute per-channel mean/std from non-zero values only.
    data shape: (N, C, T)
    """
    C = data.shape[1]
    means = np.zeros(C)
    stds = np.ones(C)

    for c in range(C):
        channel_data = data[:, c, :]
        mask = channel_data != 0
        non_zero = channel_data[mask]

        if len(non_zero) > 0:
            means[c] = non_zero.mean()
            stds[c] = non_zero.std()
            if stds[c] < 1e-8:
                stds[c] = 1.0
        else:
            logger.warning(f"Channel {c}: no non-zero values")

    logger.info("[Per-channel norm] stats computed")
    return means, stds


def normalize_per_channel(data, means, stds):
    """Apply per-channel normalization, keep zeros unchanged."""
    data_norm = data.copy()
    for c in range(data.shape[1]):
        mask = data[:, c, :] != 0
        data_norm[:, c, :][mask] = (
            data[:, c, :][mask] - means[c]
        ) / stds[c]
    return data_norm


# ==============================
# Public loader API
# ==============================

def _hf_download(repo_id, filename):
    """Thin wrapper around hf_hub_download so the repeated repo_type/cache_dir
    boilerplate isn't duplicated at every call site in load()."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=CACHE_DIR,
    )


def load(config, fold):
    dataset           = config['dataset']
    normalization     = config.get('normalization', 'global')
    val_ratio         = config.get('val_ratio', 0.1)

    repo_id = f"monster-monash/{dataset}"

    # -------- Load features --------
    X_path = _hf_download(repo_id, f"{dataset}_X.npy")
    Data_npy = np.load(X_path, mmap_mode="r")

    # -------- Load labels (filename casing varies by dataset) --------
    try:
        y_path = _hf_download(repo_id, f"{dataset}_Y.npy")
    except Exception:
        y_path = _hf_download(repo_id, f"{dataset}_y.npy")

    Label_npy = np.load(y_path)

    # -------- Load test indices --------
    test_index_path = _hf_download(repo_id, f"test_indices_fold_{fold}.txt")
    test_index = np.loadtxt(test_index_path, dtype=int)

    # -------- Split + normalize --------
    Data = split_data(
        Data_npy,
        Label_npy,
        test_index,
        normalization=normalization,
        val_ratio=val_ratio,
    )

    return Data


# ==============================
# Splitting + normalization
# ==============================

def split_data(Data_npy, Label_npy, test_index, normalization='global',
               val_ratio=0.1):

    test_mask = np.zeros(len(Label_npy), dtype=bool)
    test_mask[test_index] = True

    Data = {
        'test_data':        Data_npy[test_index],
        'test_label':       Label_npy[test_index],
        'All_train_data':   Data_npy[~test_mask],
        'All_train_label':  Label_npy[~test_mask]
    }

    # -------- Train / Val split --------
    splitter = model_selection.StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=1234
    )
    (train_idx, val_idx), = splitter.split(
        X=np.zeros(len(Data['All_train_label'])),
        y=Data['All_train_label']
    )

    Data['train_data']  = Data['All_train_data'][train_idx]
    Data['train_label'] = Data['All_train_label'][train_idx]
    Data['val_data']    = Data['All_train_data'][val_idx]
    Data['val_label']   = Data['All_train_label'][val_idx]

    logger.info(
        f"[Split] Train={len(Data['train_label'])} | "
        f"Val={len(Data['val_label'])} | "
        f"Test={len(Data['test_label'])} | "
        f"val_ratio={val_ratio}"
    )

    # -------- Normalization --------
    if normalization == 'none':
        logger.info("Normalization: NONE")
    elif normalization == 'global':
        logger.info("Normalization: GLOBAL")
        mean, std = compute_normalization_stats(Data['train_data'])
        Data['train_data'] = normalize_data(Data['train_data'], mean, std)
        Data['val_data']   = normalize_data(Data['val_data'],   mean, std)
        Data['test_data']  = normalize_data(Data['test_data'],  mean, std)
    elif normalization == 'per_channel':
        logger.info("Normalization: PER-CHANNEL")
        means, stds = compute_per_channel_stats(Data['train_data'])
        Data['train_data'] = normalize_per_channel(Data['train_data'], means, stds)
        Data['val_data']   = normalize_per_channel(Data['val_data'],   means, stds)
        Data['test_data']  = normalize_per_channel(Data['test_data'],  means, stds)
    else:
        raise ValueError(f"Unknown normalization mode: {normalization}")

    return Data


# ==============================
# PyTorch Dataset
# ==============================

class dataset_class(Dataset):
    def __init__(self, data, label):
        self.feature = data
        self.labels = label.astype(np.int32)

    def __getitem__(self, idx):
        x = self.feature[idx].astype(np.float32)
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y), idx

    def __len__(self):
        return len(self.labels)