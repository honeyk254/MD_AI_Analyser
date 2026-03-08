"""
Transformer model for trajectory temporal dynamics analysis.
Uses self-attention to learn temporal patterns in residue motion.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def run_transformer_analysis(universe, max_frames=500, **kwargs):
    """
    Apply a Transformer to learn temporal patterns in residue dynamics.

    Treats the trajectory as a sequence of frames, with per-residue features
    at each timestep. Uses self-attention to identify:
      - Key transition frames
      - Temporal residue importance
      - Structural transition events

    Returns dict with:
        - transition_frames: frames with highest attention divergence
        - temporal_importance: per-frame importance from attention
        - residue_temporal_scores: per-residue dynamic importance over time
        - predicted_transitions: detected structural transition events
        - training_losses: per-epoch training loss curve (item 53)
        - reconstruction_error: final reconstruction MSE (item 55)
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        resids = ca.resids.tolist()

        # Collect per-frame features: Cα positions relative to mean
        positions = []
        for ts in universe.trajectory:
            positions.append(ca.positions.copy())
        positions = np.array(positions)

        n_frames_total = len(positions)

        # Subsample if too many frames (item 52: track stride for correct indexing)
        actual_stride = 1
        if n_frames_total > max_frames:
            actual_stride = n_frames_total // max_frames
            positions = positions[::actual_stride]

        n_frames = len(positions)
        mean_pos = positions.mean(axis=0)
        delta = (positions - mean_pos)  # (n_frames, n_res, 3)

        # Feature: displacement magnitude per residue per frame
        displacement = np.sqrt(np.sum(delta**2, axis=2))  # (n_frames, n_res)

        # Normalize
        displacement = (displacement - displacement.mean()) / (displacement.std() + 1e-8)

        # ── Transformer Model ─────────────────────────────────
        class TrajectoryTransformer(nn.Module):
            def __init__(self, n_residues, d_model=64, nhead=4, n_layers=3, dim_ff=128):
                super().__init__()
                self.input_proj = nn.Linear(n_residues, d_model)
                self.pos_encoding = PositionalEncoding(d_model, max_len=n_frames)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                    dropout=0.1, batch_first=True, activation='gelu'
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

                self.frame_head = nn.Linear(d_model, 1)  # per-frame importance
                self.reconstruct_head = nn.Linear(d_model, n_residues)  # reconstruction

            def forward(self, x):
                h = self.input_proj(x)  # (1, T, d_model)
                h = self.pos_encoding(h)
                h = self.transformer(h)  # (1, T, d_model)
                frame_importance = self.frame_head(h).squeeze(-1)  # (1, T)
                reconstruction = self.reconstruct_head(h)  # (1, T, n_res)
                return h, frame_importance, reconstruction

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len).unsqueeze(1).float()
                div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                    (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
                self.register_buffer('pe', pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        # ── Prepare data ──────────────────────────────────────
        X = torch.tensor(displacement, dtype=torch.float32).unsqueeze(0).to(device)

        # ── Train: self-supervised reconstruction ─────────────
        model = TrajectoryTransformer(n_residues=n_res).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Track training losses (item 53)
        training_losses = []

        # Masked reconstruction: mask 15% of frames and predict them
        model.train()
        for epoch in range(150):
            optimizer.zero_grad()

            # Create mask
            mask = torch.rand(1, n_frames).to(device) < 0.15
            X_masked = X.clone()
            X_masked[mask] = 0

            hidden, frame_imp, recon = model(X_masked)

            # Reconstruction loss (only on masked positions)
            if mask.any():
                loss = F.mse_loss(recon[mask], X[mask])
            else:
                loss = F.mse_loss(recon, X)

            # Temporal smoothness regularization
            if n_frames > 1:
                smooth_loss = 0.01 * torch.mean((hidden[:, 1:] - hidden[:, :-1])**2)
                loss = loss + smooth_loss

            loss.backward()
            optimizer.step()
            training_losses.append(round(float(loss.item()), 6))

        # ── Extract results ───────────────────────────────────
        model.eval()
        with torch.no_grad():
            hidden, frame_importance, recon = model(X)

        # Reconstruction quality (item 55)
        recon_error = float(F.mse_loss(recon, X).item())

        frame_imp = frame_importance.squeeze(0).cpu().numpy()
        hidden_states = hidden.squeeze(0).cpu().numpy()

        # Per-frame importance (normalized)
        frame_imp_norm = (frame_imp - frame_imp.min()) / (frame_imp.max() - frame_imp.min() + 1e-8)

        # Detect transition frames: frames where hidden state changes abruptly
        hidden_diffs = np.sqrt(np.sum(np.diff(hidden_states, axis=0)**2, axis=1))
        diff_threshold = np.mean(hidden_diffs) + 2 * np.std(hidden_diffs)
        transition_frame_indices = np.where(hidden_diffs > diff_threshold)[0].tolist()

        # Per-residue temporal importance
        # Use gradient of frame importance w.r.t. each residue's displacement
        X_grad = X.clone().detach().requires_grad_(True)
        _, frame_imp_grad, _ = model(X_grad)
        total_imp = frame_imp_grad.sum()
        total_imp.backward()
        residue_temporal = X_grad.grad.squeeze(0).abs().mean(dim=0).cpu().numpy()
        residue_temporal_norm = (residue_temporal / (residue_temporal.max() + 1e-8)).tolist()

        # Transition events (item 52: use actual_stride for correct frame mapping)
        transitions = []
        for idx in transition_frame_indices:
            actual_frame = idx * actual_stride
            transitions.append({
                "frame": int(actual_frame),
                "magnitude": round(float(hidden_diffs[idx]), 4),
                "frame_importance": round(float(frame_imp_norm[idx]), 4),
            })
        transitions.sort(key=lambda x: -x["magnitude"])

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
        return {"error": str(e)}
