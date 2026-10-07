"""
Shared utilities for training/evaluating architectures on MONSTER datasets.
Used by gru.py, informer.py, and (later) ms4n.py so the data pipeline and
training loop stay identical across architectures -- only the model differs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from aeon.datasets import load_monster_dataset
from tqdm import tqdm


class MonsterDataset(Dataset):
    """Wraps a MONSTER dataset's X, y arrays as a PyTorch Dataset."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(dataset_name, fold=0, batch_size=64):
    """Loads a MONSTER dataset/fold and returns train and test DataLoaders."""
    X_train, y_train, X_test, y_test = load_monster_dataset(dataset_name, fold=fold)

    train_loader = DataLoader(MonsterDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MonsterDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_num=None, num_epochs=None):
    model.train()
    total_loss = 0
    seen = 0

    desc = f"Epoch {epoch_num}/{num_epochs}" if epoch_num is not None else "Training"
    progress = tqdm(loader, desc=desc, unit="batch", leave=True)

    for X_batch, y_batch in progress:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size

        # Running average loss shown live in the progress bar, not just at the end
        progress.set_postfix(loss=f"{total_loss / seen:.4f}")

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    progress = tqdm(loader, desc="Evaluating", unit="batch", leave=True)
    with torch.no_grad():
        for X_batch, y_batch in progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            progress.set_postfix(acc=f"{correct / total:.4f}")
    return correct / total


def predict_all(model, loader, device):
    """Returns every test-set prediction as a single array, in the loader's
    fixed order (test_loader uses shuffle=False, so this order is identical
    across repeated runs -- required for aligning predictions per example
    when computing bias/variance across seeds).
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds).numpy()


def run_training(model, train_loader, test_loader, device, num_epochs=10, lr=1e-3):
    """Standard training + evaluation run, shared across all architectures."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch_num=epoch + 1, num_epochs=num_epochs,
        )
        print(f"Epoch {epoch + 1}/{num_epochs}: train loss = {train_loss:.4f}")

    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    return test_acc