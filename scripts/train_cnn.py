"""
train_cnn.py

Complete CNN training pipeline with:
  - Train + validation loops every epoch
  - Adam optimiser with learning rate scheduling
  - Early stopping to prevent overfitting
  - Best model checkpoint saving
  - Training history saved as JSON (for dashboard graphs)
  - Training curve plots saved as PNG

Run with:
    python scripts/train_cnn.py

Expected runtime on CPU: 15–40 minutes (depending on dataset size)
Expected runtime on GPU: 2–5 minutes

IMPORTANT: Start this script and then open a second terminal
to write inference.py while this runs in the background.
"""

import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.ml.cnn_model  import AudioCNN, count_parameters
from backend.ml.dataset    import AudioSpectrogramDataset

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_CSV    = PROJECT_ROOT / "datasets" / "train.csv"
VAL_CSV      = PROJECT_ROOT / "datasets" / "val.csv"
SAVE_DIR     = PROJECT_ROOT / "saved_models"
PLOTS_DIR    = PROJECT_ROOT / "assets" / "plots"
MODEL_PATH   = SAVE_DIR / "best_cnn.pt"
HISTORY_PATH = SAVE_DIR / "training_history.json"

# ── Hyperparameters ────────────────────────────────────────────────────────────
# These are the knobs you can tune. For the demo, defaults work well.
BATCH_SIZE      = 16     # samples per gradient update
                         # smaller = more updates per epoch, slower
                         # larger = faster but needs more RAM
                         # 16 is a safe default for CPU
LEARNING_RATE   = 0.001  # how big a step to take each update
                         # too high = training diverges
                         # too low = training is very slow
MAX_EPOCHS      = 40     # maximum training epochs
                         # early stopping will stop before this if val_loss plateaus
PATIENCE        = 8      # early stopping: stop if val_loss doesn't improve
                         # for PATIENCE consecutive epochs
LR_PATIENCE     = 4      # reduce LR if val_loss doesn't improve for 4 epochs
NUM_WORKERS     = 0      # parallel data loading workers
                         # 0 = use main thread (safest for Windows/Mac)
                         # set to 2 or 4 on Linux for speed


def get_device() -> torch.device:
    """
    Returns the best available compute device.
    GPU is ~10x faster than CPU for CNN training.
    If no GPU is available, falls back to CPU automatically.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  GPU found: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU (M1/M2/M3 Mac)
        device = torch.device('mps')
        print("  Apple Silicon GPU found (MPS)")
    else:
        device = torch.device('cpu')
        print("  No GPU found — using CPU")
        print("  Tip: training will take 20–40 min. Start this script")
        print("       then work on inference.py in a second terminal.")
    return device


def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device
) -> tuple:
    """
    Runs one complete pass through the training data.

    What happens each iteration:
    1. Load a batch of spectrograms + labels from the DataLoader
    2. Move them to the GPU/CPU device
    3. Forward pass: feed spectrograms through the model → get predictions
    4. Compute loss: BCELoss(predictions, true_labels)
    5. Backward pass: compute gradients (how much to adjust each weight)
    6. Optimizer step: adjust weights by -gradient × learning_rate
    7. Zero gradients: reset for next batch (PyTorch accumulates by default)

    Returns:
        avg_loss (float): average loss over all batches
        accuracy (float): fraction of correct predictions (0–1)
    """
    model.train()   # sets model to training mode (enables Dropout)

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for batch_tensors, batch_labels in loader:
        # ── Move data to device (GPU or CPU) ──────────────────────────────
        batch_tensors = batch_tensors.to(device)
        batch_labels  = batch_labels.to(device)

        # ── Step 3: Forward pass ──────────────────────────────────────────
        # model(batch_tensors) calls AudioCNN.forward() automatically
        # predictions shape: (batch_size, 1) — each value 0.0–1.0
        predictions = model(batch_tensors)

        # ── Step 4: Compute loss ──────────────────────────────────────────
        # BCELoss = Binary Cross-Entropy Loss
        # Measures how wrong the predictions are:
        #   loss = -[y * log(p) + (1-y) * log(1-p)]
        # where y = true label (0 or 1), p = predicted probability
        # Perfect prediction → loss = 0
        # Worst prediction   → loss = large positive number
        loss = criterion(predictions, batch_labels)

        # ── Steps 5 & 6: Backward + update ───────────────────────────────
        optimizer.zero_grad()   # reset gradients from last batch
        loss.backward()         # compute gradients via backpropagation

        # Gradient clipping: prevents rare "exploding gradient" problem
        # Clips any gradient larger than 1.0 to exactly 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()        # update all weights using the gradients

        # ── Track metrics ─────────────────────────────────────────────────
        total_loss    += loss.item() * batch_tensors.size(0)
        # predictions > 0.5 → predict FAKE (1), else REAL (0)
        predicted      = (predictions > 0.5).float()
        correct       += (predicted == batch_labels).sum().item()
        total_samples += batch_tensors.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device
) -> tuple:
    """
    Evaluates the model on validation data WITHOUT updating weights.

    The key differences from train_one_epoch:
    - model.eval() disables Dropout (use all neurons for evaluation)
    - torch.no_grad() skips gradient computation (faster, less memory)
    - We do NOT call optimizer.step() — weights don't change

    Returns:
        avg_loss (float), accuracy (float)
    """
    model.eval()   # disables Dropout layers
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():   # no gradient tracking needed
        for batch_tensors, batch_labels in loader:
            batch_tensors = batch_tensors.to(device)
            batch_labels  = batch_labels.to(device)

            predictions = model(batch_tensors)
            loss        = criterion(predictions, batch_labels)

            total_loss    += loss.item() * batch_tensors.size(0)
            predicted      = (predictions > 0.5).float()
            correct       += (predicted == batch_labels).sum().item()
            total_samples += batch_tensors.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Why do we need this?
    Without early stopping, the model keeps training until MAX_EPOCHS.
    Past a certain point, it starts memorising the training data
    (overfitting) — val_loss stops decreasing and starts increasing.
    Early stopping halts before that damage is done.

    Parameters:
        patience (int):   how many epochs to wait for improvement
        min_delta (float): minimum improvement to count as "better"
        save_path (str):  where to save the best model checkpoint
    """

    def __init__(self, patience: int = 8, min_delta: float = 0.001,
                 save_path: str = None):
        self.patience       = patience
        self.min_delta      = min_delta
        self.save_path      = save_path
        self.best_loss      = float('inf')
        self.epochs_no_imp  = 0        # how many epochs since last improvement
        self.should_stop    = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Call this after each epoch with the validation loss.

        Returns True if training should stop, False to continue.
        Also saves the model if this is the best epoch so far.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improved! Save checkpoint and reset counter
            self.best_loss     = val_loss
            self.epochs_no_imp = 0

            if self.save_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_loss':         val_loss,
                }, self.save_path)
                # Note: we save state_dict (weights only), not the whole model
                # This is smaller, more portable, and best practice

            return False   # don't stop

        else:
            # No improvement
            self.epochs_no_imp += 1
            if self.epochs_no_imp >= self.patience:
                self.should_stop = True
                return True   # stop training

        return False


def save_training_plots(history: dict):
    """
    Saves training + validation loss and accuracy curves.
    These appear on your Dashboard page in the frontend.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Loss curves ────────────────────────────────────────────────────────
    ax1.plot(epochs, history['train_loss'], 'o-',
             color='#185FA5', linewidth=2, markersize=4, label='Training loss')
    ax1.plot(epochs, history['val_loss'], 's--',
             color='#E24B4A', linewidth=2, markersize=4, label='Validation loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (BCELoss)', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Mark the best epoch
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    ax1.axvline(x=best_epoch, color='#639922', linestyle=':', linewidth=1.5,
                label=f'Best epoch ({best_epoch})')
    ax1.legend(fontsize=10)

    # ── Accuracy curves ────────────────────────────────────────────────────
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'o-',
             color='#185FA5', linewidth=2, markersize=4, label='Training accuracy')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 's--',
             color='#E24B4A', linewidth=2, markersize=4, label='Validation accuracy')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training & Validation Accuracy', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([40, 101])

    plt.tight_layout()
    save_path = PLOTS_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Training curves saved: {save_path}")


def main():
    print("═" * 55)
    print("  Audio Deepfake Detector — CNN Training")
    print("═" * 55)

    # ── Verify prerequisites ───────────────────────────────────────────────
    for csv in [TRAIN_CSV, VAL_CSV]:
        if not csv.exists():
            print(f"\nERROR: {csv} not found!")
            print("       Run: python scripts/preprocess.py first")
            sys.exit(1)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device setup ──────────────────────────────────────────────────────
    print("\n[1/5] Setting up device...")
    device = get_device()

    # ── Data loading ──────────────────────────────────────────────────────
    print("\n[2/5] Loading datasets...")
    train_dataset = AudioSpectrogramDataset(str(TRAIN_CSV), augment=True)
    val_dataset   = AudioSpectrogramDataset(str(VAL_CSV),   augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,      # shuffle every epoch
        num_workers = NUM_WORKERS,
        pin_memory  = device.type == 'cuda'  # speeds up GPU transfers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE * 2,   # can use bigger batch for val (no gradients)
        shuffle     = False,
        num_workers = NUM_WORKERS
    )

    print(f"\n  Training batches per epoch: {len(train_loader)}")
    print(f"  Validation batches:         {len(val_loader)}")

    # ── Model setup ───────────────────────────────────────────────────────
    print("\n[3/5] Building model...")
    model = AudioCNN().to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    # ── Loss function and optimiser ───────────────────────────────────────
    # BCELoss: Binary Cross-Entropy — standard loss for binary classification
    criterion = nn.BCELoss()

    # Adam: Adaptive Moment Estimation
    # Better than plain gradient descent — adapts the learning rate
    # individually for each parameter based on past gradients
    # weight_decay: L2 regularisation — penalises large weights (prevents overfitting)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = 1e-4
    )

    # ReduceLROnPlateau: if val_loss doesn't improve for LR_PATIENCE epochs,
    # multiply the learning rate by 0.5. This helps escape plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'min',      # monitor val_loss (lower = better)
        patience = LR_PATIENCE,
        factor   = 0.5,        # multiply LR by 0.5
        verbose  = False
    )

    early_stopping = EarlyStopping(
        patience  = PATIENCE,
        save_path = str(MODEL_PATH)
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[4/5] Training for up to {MAX_EPOCHS} epochs...")
    print(f"      Early stopping patience: {PATIENCE} epochs")
    print(f"      Best model → {MODEL_PATH}")
    print()
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8} | {'Status'}")
    print("  " + "─" * 73)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        # One full training pass
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # One full validation pass
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Check early stopping
        stop = early_stopping.step(val_loss, model)

        # Record history
        history['train_loss'].append(round(train_loss, 5))
        history['val_loss'].append(round(val_loss, 5))
        history['train_acc'].append(round(train_acc, 5))
        history['val_acc'].append(round(val_acc, 5))

        # Status indicator
        epoch_time = time.time() - epoch_start
        is_best    = early_stopping.epochs_no_imp == 0
        status     = "✓ best" if is_best else f"  ({early_stopping.epochs_no_imp}/{PATIENCE})"

        print(f"  {epoch:>5} | {train_loss:>10.4f} | {train_acc*100:>8.1f}% | "
              f"{val_loss:>8.4f} | {val_acc*100:>6.1f}% | "
              f"{current_lr:>8.6f} | {status}")

        if stop:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            print(f"  Best val_loss: {early_stopping.best_loss:.4f}")
            break

    total_time = time.time() - start_time
    print(f"\n  Total training time: {total_time/60:.1f} minutes")

    # ── Save history JSON ─────────────────────────────────────────────────
    print("\n[5/5] Saving results...")

    history['best_val_loss'] = early_stopping.best_loss
    history['best_val_acc']  = max(history['val_acc'])
    history['epochs_trained'] = len(history['train_loss'])
    history['hyperparams'] = {
        'batch_size':    BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'max_epochs':    MAX_EPOCHS,
        'patience':      PATIENCE
    }

    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ Training history saved: {HISTORY_PATH}")

    # Save training plots
    save_training_plots(history)

    # ── Final summary ─────────────────────────────────────────────────────
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    print("\n" + "═" * 55)
    print("  Training Complete!")
    print(f"  Best epoch:       {best_epoch}")
    print(f"  Best val_loss:    {min(history['val_loss']):.4f}")
    print(f"  Best val_acc:     {max(history['val_acc'])*100:.1f}%")
    print(f"  Model saved at:   {MODEL_PATH}")
    print("═" * 55)
    print("\n  Next: python scripts/evaluate.py")


if __name__ == "__main__":
    main()