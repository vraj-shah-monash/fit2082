#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal supervised training loop.

Tracks only loss and accuracy (train/val/test) -- no confusion matrices,
precision, balanced accuracy, FLOPs, memory tracking, or TensorBoard.
Saves the best checkpoint by validation accuracy and plots loss/accuracy
curves with matplotlib when training finishes.

Only two external dependencies remain, since they're how the model and
data actually get built (not "extra" bookkeeping):
    - utils.model_factory.Model_factory -> builds the model from config
    - utils.data_loader.dataset_class   -> wraps numpy arrays as a Dataset
"""

import os
import copy
import time

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import model_factory
from utils.data_loader import dataset_class


# ==============================
# Epoch loop
# ==============================

def run_epoch(model, loader, device, loss_fn, optimizer=None):
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device).long()
            preds = model(X)
            loss = loss_fn(preds, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


# ==============================
# Plotting
# ==============================

def plot_curves(history, save_path=None):
    """Side-by-side loss and accuracy curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    plt.close(fig)


# ==============================
# Public trainer API
# ==============================

def trainer(config, Data):
    """
    Train a model built from `config` on `Data`, tracking loss + accuracy only.

    Args:
        config: dict with 'device', 'batch_size', 'lr', 'epochs', 'save_dir',
                'fold', 'Model_Type'; optional 'lr_step_size', 'lr_gamma',
                'early_stopping_patience' (default 20).
        Data: dict with 'train_data'/'train_label', 'val_data'/'val_label',
              'test_data'/'test_label' (as produced by utils.data_loader.load).

    Returns:
        A flat dict: {'accuracy', 'val_accuracy', 'train_time_sec'}.
    """
    device = config['device']

    # -------- Data --------
    train_loader = DataLoader(dataset_class(Data['train_data'], Data['train_label']),
                               batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset_class(Data['val_data'], Data['val_label']),
                             batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_loader = DataLoader(dataset_class(Data['test_data'], Data['test_label']),
                              batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # -------- Model --------
    config['Data_shape'] = Data['train_data'].shape
    config['num_labels'] = int(max(Data['train_label'])) + 1
    input_shape = tuple(config['Data_shape'][1:])
    model = model_factory.Model_factory(config, input_shape)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['Model_Type']} | Total parameters: {total_params:,}")

    # -------- Optimizer / loss --------
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 50),
        gamma=config.get('lr_gamma', 0.9),
    )
    loss_fn = config.get('loss_fn', torch.nn.CrossEntropyLoss())

    # -------- Training loop --------
    epochs = config['epochs']
    patience = config.get('early_stopping_patience', 30)
    fold_dir = os.path.join(config['save_dir'], f'fold_{config["fold"]}')
    best_path = os.path.join(fold_dir, 'model_best.pth')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, loss_fn)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % config.get('print_interval', 10) == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(best_state, best_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no val accuracy improvement for {patience} epochs).")
            break

    train_time_sec = time.time() - start_time
    print(f"Training finished in {train_time_sec/60:.1f} min. Best val_accuracy = {best_val_acc:.4f}")

    # -------- Final test evaluation with the best checkpoint --------
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, device, loss_fn)
    print(f"Test accuracy: {test_acc:.4f}")

    # -------- Plot --------
    plot_path = os.path.join(fold_dir, f'{config["Model_Type"]}_{config["dataset"]}_curves.png')
    plot_curves(history, save_path=plot_path)

    return {
        'accuracy': test_acc,
        'val_accuracy': best_val_acc,
        'train_time_sec': train_time_sec,
    }