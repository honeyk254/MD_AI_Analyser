from __future__ import annotations

"""Variational Autoencoder (VAE) for conformational latent-space learning.

Trains a small VAE on per-frame C-alpha pairwise-distance matrices to
learn a continuous latent space of conformations.  Supports configurable
latent dimensionality (2, 4, or 8) and reports per-frame reconstruction
error alongside latent-space density histograms.
"""

import logging
from typing import Any, Dict, List

import numpy as np

from ..utils.ml_feature_utils import (
    standardise_features,
    set_global_seed,
)

logger = logging.getLogger("md_ai_analyzer")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_vae_analysis(
    universe: Any,
    latent_dim: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a VAE on C-alpha distance features and project into latent space.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded trajectory universe.
    latent_dim : int
        Dimensionality of the latent space (e.g. 2, 4, or 8).
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for training.
    **kwargs
        Additional keyword arguments (unused; accepted for orchestrator
        compatibility).

    Returns
    -------
    dict
        Keys:
        - ``latent_coords`` : per-frame latent coordinates
        - ``reconstruction_loss`` : per-epoch reconstruction loss
        - ``kl_loss`` : per-epoch KL divergence loss
        - ``total_loss`` : per-epoch total loss
        - ``latent_density`` : 2-D histogram density of latent space
        - ``n_frames`` : number of frames
        - ``input_dim`` : feature dimensionality
        - ``latent_dim`` : latent dimensionality used
        - ``epochs_trained`` : number of epochs
        - ``reconstruction_error`` : final mean per-frame reconstruction MSE
        - ``per_frame_recon_error`` : per-frame reconstruction MSE list
        - ``latent_variance`` : variance per latent dimension
    """
    if not HAS_TORCH:
        logger.error("PyTorch not available; cannot run VAE analysis.")
        return {"error": "PyTorch not available for VAE analysis"}

    # Reproducibility
    set_global_seed(42)

    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            logger.error("No CA atoms found for VAE analysis.")
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        n_frames = len(universe.trajectory)

        if n_frames < 20:
            logger.warning("Only %d frames; need >= 20 for VAE.", n_frames)
            return {"error": "Too few frames for VAE training"}

        # ── Build feature matrix ─────────────────────────────────
        # For large proteins, sub-sample residues to keep features tractable.
        max_residues = 100
        if n_res > max_residues:
            stride = n_res // max_residues
            ca_subset = ca[::stride]
            n_feat_res = len(ca_subset)
        else:
            ca_subset = ca
            n_feat_res = n_res

        from MDAnalysis.lib.distances import distance_array

        n_pairs = n_feat_res * (n_feat_res - 1) // 2
        triu_idx = np.triu_indices(n_feat_res, k=1)
        features = np.zeros((n_frames, n_pairs), dtype=np.float64)

        for frame_idx, _ts in enumerate(universe.trajectory):
            dists = distance_array(ca_subset.positions, ca_subset.positions)
            features[frame_idx] = dists[triu_idx]

        # ── Normalise features via shared utility ────────────────
        features_norm, _feat_mean, _feat_std = standardise_features(features)

        # ── Build VAE ────────────────────────────────────────────
        input_dim: int = n_pairs
        hidden_dim: int = min(256, max(64, input_dim // 4))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _VAE(input_dim, hidden_dim, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        dataset = TensorDataset(torch.FloatTensor(features_norm))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # ── Training (with KL annealing warmup) ────────────────
        recon_losses: List[float] = []
        kl_losses: List[float] = []
        total_losses: List[float] = []

        # KL annealing: linearly ramp beta from 0 to 1 over the first
        # 40% of epochs to prevent posterior collapse (Bowman et al. 2016).
        kl_warmup_epochs: int = max(1, int(epochs * 0.4))

        for epoch in range(epochs):
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_samples = 0

            beta: float = min(1.0, epoch / kl_warmup_epochs)

            model.train()
            for (batch,) in loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)

                recon_loss = nn.functional.mse_loss(recon, batch, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                n_samples += len(batch)

            recon_losses.append(round(epoch_recon / max(n_samples, 1), 4))
            kl_losses.append(round(epoch_kl / max(n_samples, 1), 4))
            total_losses.append(
                round((epoch_recon + epoch_kl) / max(n_samples, 1), 4)
            )

        # ── Encode all frames (eval mode) ────────────────────────
        model.eval()
        with torch.no_grad():
            all_data = torch.FloatTensor(features_norm).to(device)
            mu_all, _logvar_all = model.encode(all_data)
            latent: np.ndarray = mu_all.cpu().numpy()

            # Reconstruction quality
            recon_all, _, _ = model(all_data)
            per_frame_mse: np.ndarray = (
                torch.mean((recon_all - all_data) ** 2, dim=1).cpu().numpy()
            )
            overall_recon_error = float(per_frame_mse.mean())

        # Per-dimension variance in latent space
        latent_variance: List[float] = np.var(latent, axis=0).tolist()

        # Latent-space density (2-D histogram on first two dims)
        density: Dict[str, Any]
        if latent.shape[1] >= 2:
            hist, xedges, yedges = np.histogram2d(
                latent[:, 0], latent[:, 1], bins=30
            )
            density = {
                "histogram": hist.tolist(),
                "x_edges": xedges.tolist(),
                "y_edges": yedges.tolist(),
            }
        else:
            density = {}

        logger.info(
            "VAE training complete: %d epochs, final recon error=%.6f.",
            epochs,
            overall_recon_error,
        )

        return {
            "latent_coords": latent.tolist(),
            "reconstruction_loss": recon_losses,
            "kl_loss": kl_losses,
            "total_loss": total_losses,
            "latent_density": density,
            "n_frames": n_frames,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "epochs_trained": epochs,
            "reconstruction_error": round(overall_recon_error, 6),
            "per_frame_recon_error": [round(float(x), 4) for x in per_frame_mse],
            "latent_variance": [round(float(v), 4) for v in latent_variance],
            "caveat": (
                "The latent space is an unsupervised low-dimensional representation for "
                "exploration, not a validated mechanistic or kinetic model."
            ),
        }

    except Exception as e:
        logger.exception("VAE analysis failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# VAE Model
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _VAE(nn.Module):
        """Simple Variational Autoencoder for conformational analysis.

        Parameters
        ----------
        input_dim : int
            Number of input features (pairwise distances).
        hidden_dim : int
            Width of the hidden layers.
        latent_dim : int
            Dimensionality of the latent space.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
        ) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Encode input into latent mean and log-variance.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape ``(batch, input_dim)``.

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                ``(mu, logvar)`` each of shape ``(batch, latent_dim)``.
            """
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(
            self, mu: torch.Tensor, logvar: torch.Tensor
        ) -> torch.Tensor:
            """Apply the reparameterisation trick.

            Parameters
            ----------
            mu : torch.Tensor
            logvar : torch.Tensor

            Returns
            -------
            torch.Tensor
                Sampled latent vector.
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Decode latent vector back to input space.

            Parameters
            ----------
            z : torch.Tensor

            Returns
            -------
            torch.Tensor
            """
            return self.decoder(z)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Full forward pass: encode, reparameterise, decode.

            Parameters
            ----------
            x : torch.Tensor

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ``(reconstruction, mu, logvar)``.
            """
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar
