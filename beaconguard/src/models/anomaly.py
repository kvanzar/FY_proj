"""FR-4 — Anomaly detection branch: an autoencoder trained exclusively on
benign flows, so it can flag C2 families absent from supervised training
data (GAP-3) purely by their deviation from normal behaviour.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ..config import get_logger

logger = get_logger(__name__)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


class Autoencoder(nn.Module if nn is not None else object):
    """Small symmetric autoencoder. hidden_dims e.g. [32,16,8,16,32] is
    read as encoder [in->32->16->8] then decoder [8->16->32->in]."""

    def __init__(self, n_features: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        mid = len(hidden_dims) // 2
        encoder_dims = [n_features] + hidden_dims[: mid + 1]
        decoder_dims = hidden_dims[mid:] + [n_features]

        def block(in_d, out_d, activate=True):
            layers = [nn.Linear(in_d, out_d)]
            if activate:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
            return layers

        enc_layers = []
        for i in range(len(encoder_dims) - 1):
            enc_layers += block(encoder_dims[i], encoder_dims[i + 1])
        dec_layers = []
        for i in range(len(decoder_dims) - 1):
            is_last = i == len(decoder_dims) - 2
            dec_layers += block(decoder_dims[i], decoder_dims[i + 1], activate=not is_last)

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X_benign_train: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[Any, StandardScaler, dict[str, Any]]:
    """FR-4.1: fit exclusively on benign flows. No malicious data may
    touch this branch during training — this is enforced by the caller
    (evaluate.py / run_pipeline.py) filtering the training split to
    y_train == 0 before this function is ever called.

    Features span wildly different scales (e.g. resp_port up to ~65535
    vs. iat_cv typically 0-2) — an autoencoder's MSE reconstruction loss
    is dominated by whichever raw feature has the largest magnitude, so
    without scaling the loss is driven almost entirely by ports/byte
    counts and the model never learns the timing/ratio features this
    branch actually exists to catch. The scaler is fit here (on benign
    training data only, consistent with FR-2.6.5) and must be reused by
    every later call to `score()` on this model.
    """
    if torch is None:
        raise ImportError("PyTorch is required for the anomaly branch; see requirements.txt")

    ae_cfg = cfg["anomaly"]["autoencoder"]
    torch.manual_seed(cfg["seed"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_benign_train.to_numpy(dtype=np.float64)).astype(np.float32)
    X_tensor = torch.tensor(X_scaled)
    model = Autoencoder(X_tensor.shape[1], ae_cfg["hidden_dims"], ae_cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg["learning_rate"])
    loss_fn = nn.MSELoss()

    n_val = max(1, int(0.1 * len(X_tensor)))
    perm = torch.randperm(len(X_tensor))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train_t, X_val_t = X_tensor[train_idx], X_tensor[val_idx]

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(ae_cfg["epochs"]):
        model.train()
        perm_epoch = torch.randperm(len(X_train_t))
        for start in range(0, len(X_train_t), ae_cfg["batch_size"]):
            batch_idx = perm_epoch[start: start + ae_cfg["batch_size"]]
            batch = X_train_t[batch_idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), X_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= ae_cfg["early_stopping_patience"]:
                logger.info("Autoencoder early stop at epoch %d (val_loss=%.5f)", epoch, best_val_loss)
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_errors = _reconstruction_error(model, X_tensor)
    threshold = float(np.percentile(train_errors, cfg["anomaly"]["threshold_percentile"]))  # FR-4.3

    metadata = {
        "final_val_loss": best_val_loss,
        "threshold_percentile": cfg["anomaly"]["threshold_percentile"],
        "threshold_raw_error": threshold,
        "n_benign_train_rows": len(X_benign_train),
    }
    logger.info(
        "Autoencoder trained: val_loss=%.5f, threshold(p%d)=%.5f",
        best_val_loss, cfg["anomaly"]["threshold_percentile"], threshold,
    )
    return model, scaler, metadata


def _reconstruction_error(model, X_tensor) -> np.ndarray:
    with torch.no_grad():
        recon = model(X_tensor)
        error = torch.mean((recon - X_tensor) ** 2, dim=1)
    return error.numpy()


def score(model, scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    """FR-4.2: reconstruction error per flow. `scaler` must be the one
    returned by `train_autoencoder` for this model — scoring with raw,
    unscaled features would produce errors on a completely different
    scale than the threshold this model's autoencoder was calibrated on."""
    X_scaled = scaler.transform(X.to_numpy(dtype=np.float64)).astype(np.float32)
    X_tensor = torch.tensor(X_scaled)
    return _reconstruction_error(model, X_tensor)


def normalise_score(raw_errors: np.ndarray, threshold: float) -> np.ndarray:
    """FR-4.6: normalise reconstruction error to [0,1] for fusion, using
    the benign-derived threshold as the midpoint (error==threshold -> 0.5)
    via a logistic squashing so scores stay bounded for outlier errors."""
    scaled = (raw_errors - threshold) / (threshold + 1e-9)
    return 1 / (1 + np.exp(-scaled))


def train_baseline(name: str, X_benign_train: pd.DataFrame, cfg: dict[str, Any]):
    """FR-4.4: Isolation Forest / One-Class SVM baselines for comparison."""
    seed = cfg["seed"]
    if name == "isolation_forest":
        model = IsolationForest(random_state=seed, contamination="auto")
    elif name == "one_class_svm":
        model = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    else:
        raise ValueError(f"Unknown anomaly baseline: {name}")
    model.fit(X_benign_train)
    return model


def baseline_score(model, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous, matching the autoencoder's convention
    (sklearn's decision_function is the opposite sign, so negate it)."""
    return -model.decision_function(X)
