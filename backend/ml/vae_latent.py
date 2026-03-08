"""
Variational Autoencoder (VAE) for conformational latent space learning.
Trains a small VAE on per-frame Cα pairwise distance matrices
to learn a continuous 2D latent space of conformations.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def run_vae_analysis(universe, latent_dim=2, epochs=50, batch_size=32, **kwargs):
    """
    Train a VAE on Cα distance features and project into latent space.

    Args:
        latent_dim: Dimensionality of latent space (2, 4, or 8) (item 54)

    Returns dict with:
        - latent_coords: per-frame latent coordinates
        - reconstruction_loss: per-epoch reconstruction loss
        - kl_loss: per-epoch KL divergence loss
        - total_loss: per-epoch total loss
        - latent_density: 2D histogram density of latent space
        - reconstruction_error: final per-frame reconstruction MSE (item 55)
        - latent_variance: variance explained per latent dimension
    """
    if not HAS_TORCH:
        return {"error": "PyTorch not available for VAE analysis"}

    try:
        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            return {"error": "No CA atoms found"}

        n_res = len(ca)
        n_frames = len(universe.trajectory)

        if n_frames < 20:
            return {"error": "Too few frames for VAE training"}

        # Build feature matrix: for each frame, compute upper triangle of pairwise distances
        # For large proteins, subsample residues
        max_residues = 100
        if n_res > max_residues:
            stride = n_res // max_residues
            ca_subset = ca[::stride]
            n_feat_res = len(ca_subset)
        else:
            ca_subset = ca
            n_feat_res = n_res

        # Collect pairwise distances
        from MDAnalysis.lib.distances import distance_array
        n_pairs = n_feat_res * (n_feat_res - 1) // 2
        features = np.zeros((n_frames, n_pairs))

        for frame_idx, ts in enumerate(universe.trajectory):
            dists = distance_array(ca_subset.positions, ca_subset.positions)
            # Upper triangle
            triu_idx = np.triu_indices(n_feat_res, k=1)
            features[frame_idx] = dists[triu_idx]

        # Normalize features
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0)
        feat_std[feat_std == 0] = 1.0
        features_norm = (features - feat_mean) / feat_std

        # Build VAE
        input_dim = n_pairs
        hidden_dim = min(256, max(64, input_dim // 4))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _VAE(input_dim, hidden_dim, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        dataset = TensorDataset(torch.FloatTensor(features_norm))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training
        recon_losses = []
        kl_losses = []
        total_losses = []

        for epoch in range(epochs):
            epoch_recon = 0
            epoch_kl = 0
            n_batches = 0

            model.train()
            for (batch,) in loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)

                recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                n_batches += len(batch)

            recon_losses.append(round(epoch_recon / n_batches, 4))
            kl_losses.append(round(epoch_kl / n_batches, 4))
            total_losses.append(round((epoch_recon + epoch_kl) / n_batches, 4))

        # Encode all frames
        model.eval()
        with torch.no_grad():
            all_data = torch.FloatTensor(features_norm).to(device)
            mu, logvar = model.encode(all_data)
            latent = mu.cpu().numpy()

            # Reconstruction quality (item 55)
            recon_all, _, _ = model(all_data)
            per_frame_mse = torch.mean((recon_all - all_data) ** 2, dim=1).cpu().numpy()
            overall_recon_error = float(per_frame_mse.mean())

        # Per-dimension variance in latent space
        latent_variance = np.var(latent, axis=0).tolist()

        # Latent space density (2D histogram)
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
        }

    except Exception as e:
        return {"error": str(e)}


if HAS_TORCH:
    class _VAE(nn.Module):
        """Simple Variational Autoencoder."""

        def __init__(self, input_dim, hidden_dim, latent_dim):
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

        def encode(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar
