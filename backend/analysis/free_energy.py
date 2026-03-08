"""
Free Energy Landscape computation.
Constructs 2D free energy surface from PCA projections.
"""
import numpy as np


def compute_free_energy_landscape(universe, n_bins=50, temperature=300, **kwargs):
    """
    Compute 2D free energy landscape from PCA projections.
    FEL = -kT * ln(P(PC1, PC2))

    Returns dict with:
        - fel: 2D free energy values (kJ/mol)
        - pc1_edges: bin edges for PC1
        - pc2_edges: bin edges for PC2
        - minima: list of free energy minima locations
        - temperature: temperature used
    """
    try:
        from sklearn.decomposition import PCA

        ca = universe.select_atoms("protein and name CA")
        if len(ca) == 0:
            ca = universe.select_atoms("all")

        coords = []
        for ts in universe.trajectory:
            coords.append(ca.positions.flatten().copy())
        coords = np.array(coords)

        n_comp = min(2, coords.shape[0] - 1, coords.shape[1])
        if n_comp < 2:
            return {"error": "Not enough data for FEL"}

        pca = PCA(n_components=2)
        projections = pca.fit_transform(coords)

        pc1 = projections[:, 0]
        pc2 = projections[:, 1]

        # 2D histogram
        H, xedges, yedges = np.histogram2d(pc1, pc2, bins=n_bins, density=True)

        # Free energy: F = -kT ln(P)
        kB = 8.314e-3  # kJ/(mol·K)
        kT = kB * temperature

        H[H == 0] = H[H > 0].min() * 0.01  # avoid log(0)
        F = -kT * np.log(H)
        F -= F.min()  # set minimum to 0

        # Find local minima
        minima = []
        for i in range(1, F.shape[0] - 1):
            for j in range(1, F.shape[1] - 1):
                val = F[i, j]
                neighbors = [F[i-1,j], F[i+1,j], F[i,j-1], F[i,j+1],
                            F[i-1,j-1], F[i-1,j+1], F[i+1,j-1], F[i+1,j+1]]
                if all(val <= n for n in neighbors) and val < np.mean(F) * 0.5:
                    x_center = (xedges[i] + xedges[i+1]) / 2
                    y_center = (yedges[j] + yedges[j+1]) / 2
                    minima.append({
                        "pc1": round(float(x_center), 2),
                        "pc2": round(float(y_center), 2),
                        "free_energy": round(float(val), 2),
                    })

        minima.sort(key=lambda x: x["free_energy"])

        return {
            "fel": F.tolist(),
            "pc1_edges": xedges.tolist(),
            "pc2_edges": yedges.tolist(),
            "minima": minima[:20],
            "temperature": temperature,
            "n_bins": n_bins,
            "kT": round(kT, 4),
            "pc1_range": [float(pc1.min()), float(pc1.max())],
            "pc2_range": [float(pc2.min()), float(pc2.max())],
        }

    except Exception as e:
        return {"error": str(e)}
