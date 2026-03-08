from __future__ import annotations

"""
Transformer model for trajectory temporal dynamics analysis.

Uses self-attention to learn temporal patterns in residue motion, identifying
key transition frames and per-residue dynamic importance over time.
"""

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.trajectory_utils import collect_ca_positions, select_ca_atoms
from ..utils.ml_feature_utils import set_global_seed

logger = logging.getLogger("md_ai_analyzer")


# ── Model classes (module-level) ─────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input sequences.

    Adds a fixed positional signal to each timestep embedding so the
    transformer can reason about temporal ordering.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length supported.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # div_term has length ceil(d_model / 2), which matches pe[:, 0::2].
        # pe[:, 1::2] has length floor(d_model / 2).  When d_model is odd
        # the sin slice is one element longer than the cos slice, so we
        # must trim div_term for the cosine assignment.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Same shape as *x*, with positional signal added.
        """
        return x + self.pe[:, : x.size(1)]


class TrajectoryTransformer(nn.Module):
    """Transformer encoder for per-frame trajectory analysis.

    Projects each frame's per-residue displacement vector into a latent
    space, applies multi-head self-attention, then produces per-frame
    importance scores and reconstructed displacement vectors.

    Parameters
    ----------
    n_residues : int
        Number of input residue features per frame.
    d_model : int
        Internal embedding dimension (default 64).
    nhead : int
        Number of attention heads (default 4).
    n_layers : int
        Number of transformer encoder layers (default 3).
    dim_ff : int
        Feed-forward hidden dimension (default 128).
    max_len : int
        Maximum sequence length for positional encoding (default 5000).
    """

    def __init__(
        self,
        n_residues: int,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 3,
        dim_ff: int = 128,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_residues, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.frame_head = nn.Linear(d_model, 1)        # per-frame importance
        self.reconstruct_head = nn.Linear(d_model, n_residues)  # reconstruction

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(1, T, n_residues)`` — per-frame displacement magnitudes.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(hidden_states, frame_importance, reconstruction)`` with shapes
            ``(1, T, d_model)``, ``(1, T)``, ``(1, T, n_residues)``.
        """
        h = self.input_proj(x)          # (1, T, d_model)
        h = self.pos_encoding(h)
        h = self.transformer(h)         # (1, T, d_model)
        frame_importance = self.frame_head(h).squeeze(-1)  # (1, T)
        reconstruction = self.reconstruct_head(h)          # (1, T, n_res)
        return h, frame_importance, reconstruction


# ── Main analysis entry point ────────────────────────────────────

def run_transformer_analysis(
    universe: Any,
    max_frames: int = 500,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply a Transformer to learn temporal patterns in residue dynamics.

    Treats the trajectory as a time-series of frames with per-residue
    displacement features.  Uses masked self-supervised reconstruction to
    learn frame representations, then identifies:

    * Key transition frames (abrupt hidden-state changes)
    * Temporal residue importance (gradient-based attribution)
    * Structural transition events

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    max_frames : int
        Maximum number of frames to analyse; longer trajectories are
        sub-sampled with a uniform stride.
    **kwargs : Any
        Reserved for future options.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``temporal_importance`` -- per-frame normalised importance
        - ``transition_frames`` -- detected structural transitions
        - ``residue_temporal_scores`` -- per-residue dynamic importance
        - ``resids`` -- residue IDs
        - ``n_frames_analyzed`` -- number of frames after sub-sampling
        - ``n_transitions_detected`` -- count of detected transitions
        - ``model_type`` -- architecture description
        - ``training_losses`` -- per-epoch loss curve
        - ``reconstruction_error`` -- final reconstruction MSE
        - ``stride_used`` -- sub-sampling stride applied
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Transformer analysis starting on device=%s", device)

        # ── Select CA atoms and collect positions ────────────────
        ca = select_ca_atoms(universe)
        n_res: int = len(ca)
        resids: list[int] = ca.resids.tolist()

        positions: np.ndarray = collect_ca_positions(universe, atoms=ca)
        n_frames_total: int = positions.shape[0]

        # Sub-sample if too many frames
        actual_stride: int = 1
        if n_frames_total > max_frames:
            actual_stride = n_frames_total // max_frames
            positions = positions[::actual_stride]

        n_frames: int = len(positions)
        logger.info(
            "Transformer: %d residues, %d frames (stride=%d)",
            n_res, n_frames, actual_stride,
        )

        mean_pos: np.ndarray = positions.mean(axis=0)
        delta: np.ndarray = positions - mean_pos  # (n_frames, n_res, 3)

        # Feature: displacement magnitude per residue per frame
        displacement: np.ndarray = np.sqrt(np.sum(delta ** 2, axis=2))  # (n_frames, n_res)

        # Normalise to zero-mean, unit-variance
        displacement = (displacement - displacement.mean()) / (displacement.std() + 1e-8)

        # ── Prepare data ─────────────────────────────────────────
        X: torch.Tensor = (
            torch.tensor(displacement, dtype=torch.float32).unsqueeze(0).to(device)
        )

        # ── Train: self-supervised masked reconstruction ─────────
        set_global_seed(42)

        model = TrajectoryTransformer(n_residues=n_res, max_len=n_frames).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        training_losses: list[float] = []

        model.train()
        for epoch in range(150):
            optimizer.zero_grad()

            # Create mask (15 % of frames)
            mask = torch.rand(1, n_frames).to(device) < 0.15
            X_masked = X.clone()
            X_masked[mask] = 0

            hidden, frame_imp, recon = model(X_masked)

            # Reconstruction loss (only on masked positions)
            if mask.any():
                loss = F.mse_loss(recon[mask], X[mask])
            else:
                loss = F.mse_loss(recon, X)

            # Temporal smoothness regularisation
            if n_frames > 1:
                smooth_loss = 0.01 * torch.mean((hidden[:, 1:] - hidden[:, :-1]) ** 2)
                loss = loss + smooth_loss

            loss.backward()
            optimizer.step()

            loss_val = round(float(loss.item()), 6)
            training_losses.append(loss_val)

            if epoch % 50 == 0 or epoch == 149:
                logger.info(
                    "Transformer epoch %3d/150 | loss=%.6f", epoch, loss_val,
                )

        # ── Extract results ──────────────────────────────────────
        model.eval()
        with torch.no_grad():
            hidden, frame_importance, recon = model(X)

        recon_error: float = float(F.mse_loss(recon, X).item())

        frame_imp: np.ndarray = frame_importance.squeeze(0).cpu().numpy()
        hidden_states: np.ndarray = hidden.squeeze(0).cpu().numpy()

        # Per-frame importance (normalised to [0, 1])
        frame_imp_norm: np.ndarray = (
            (frame_imp - frame_imp.min()) / (frame_imp.max() - frame_imp.min() + 1e-8)
        )

        # Detect transition frames: abrupt hidden-state changes
        hidden_diffs: np.ndarray = np.sqrt(
            np.sum(np.diff(hidden_states, axis=0) ** 2, axis=1)
        )
        diff_threshold: float = float(np.mean(hidden_diffs) + 2 * np.std(hidden_diffs))
        transition_frame_indices: list[int] = np.where(
            hidden_diffs > diff_threshold
        )[0].tolist()

        # Per-residue temporal importance via gradient attribution
        X_grad = X.clone().detach().requires_grad_(True)
        _, frame_imp_grad, _ = model(X_grad)
        total_imp = frame_imp_grad.sum()
        total_imp.backward()
        residue_temporal: np.ndarray = (
            X_grad.grad.squeeze(0).abs().mean(dim=0).cpu().numpy()
        )
        residue_temporal_norm: list[float] = (
            residue_temporal / (residue_temporal.max() + 1e-8)
        ).tolist()

        # Build transition event list (map back to original frame indices)
        transitions: list[dict[str, Any]] = []
        for idx in transition_frame_indices:
            actual_frame = idx * actual_stride
            transitions.append({
                "frame": int(actual_frame),
                "magnitude": round(float(hidden_diffs[idx]), 4),
                "frame_importance": round(float(frame_imp_norm[idx]), 4),
            })
        transitions.sort(key=lambda t: -t["magnitude"])

        logger.info(
            "Transformer analysis complete: %d frames, %d transitions, "
            "recon_error=%.6f",
            n_frames, len(transition_frame_indices), recon_error,
        )

        return {
            "temporal_importance": frame_imp_norm.tolist(),
            "transition_frames": transitions[:20],
            "residue_temporal_scores": residue_temporal_norm,
            "resids": resids,
            "n_frames_analyzed": n_frames,
            "n_transitions_detected": len(transition_frame_indices),
            "model_type": "Transformer (self-supervised masked reconstruction)",
            "training_losses": training_losses,
            "reconstruction_error": round(recon_error, 6),
            "stride_used": actual_stride,
        }

    except Exception as e:
        logger.error("Transformer analysis failed: %s", e, exc_info=True)
        return {"error": str(e)}
