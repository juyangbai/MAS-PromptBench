# Vendored from MASPOB, https://github.com/HZ1008/MASPOB, commit 0be3af42da83b9b5ab187c101ec6d2ce87a5fe04,
# file scripts/utils/training.py. Z. Hong, Q. Zhang, J. Sun, Z. Shang, M. Kong, X. Wang, Y. Shu and Z. Dai,
# "MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks",
# arXiv:2603.02630, 2026. Upstream terms: "This project is released for research purposes. Please refer
# to the paper for details." The pinned upstream tree contains no LICENSE, COPYING or NOTICE file.
# Everything below the end-of-header line is byte-identical to the upstream file; its SHA-256 is pinned
# in ../regime.py (CORE_SOURCE_SHA256) and verified before the file is executed.
# --- end of vendoring header ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_with_early_stopping(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.Tensor,
    targets: torch.Tensor,
    max_epochs: int = 500,
    patience: int = 30,
    min_delta: float = 1e-5,
    verbose: bool = False,
    use_scheduler: bool = True,
) -> float:
    """Train with early stopping and optional learning-rate scheduling."""
    import copy
    model.train()
    best_loss, patience_counter = float('inf'), 0
    best_state = None

    # ReduceLROnPlateau: lower lr when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=30, min_lr=1e-5
    ) if use_scheduler else None

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        preds = model(data)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if scheduler is not None:
            scheduler.step(current_loss)

        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1} (best loss: {best_loss:.6f})")
            break

    # Restore best model state
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_loss

