"""Phase 6 stretch: VAMPnet ablation against the TICA/MSM baseline.

A small lobet-style network (two tanh layers -> softmax state weights) trained
with the differentiable VAMP-2 score (Wu & Noe, 2017). The ablation reports
the leading implied timescale and state assignment of the learned nonlinear
embedding next to the linear TICA baseline — agreement as specific numbers.

torch is an optional dependency: ``pip install md-ai-platform[vampnets]``.
Without it the ablation reports ``available=False`` and the rest of the ML
layer is unaffected.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .schemas import VAMPNetAblation

try:
    import torch
except ModuleNotFoundError:  # optional dependency
    torch = None  # type: ignore[assignment]

EPOCHS = 300
SEED = 42


def run_vampnet_ablation(
    feature_matrix: np.ndarray,
    lag_frames: int,
    lag_ps: float,
    n_states: int,
    tica_labels: np.ndarray,
    tica_leading_timescale_ps: Optional[float],
    epochs: int = EPOCHS,
) -> VAMPNetAblation:
    """Train a tiny VAMPnet and ablate it against the TICA/MSM baseline."""
    if torch is None:
        return VAMPNetAblation(
            available=False,
            n_states=n_states,
            summary=(
                "VAMPnet ablation skipped: torch not installed "
                "(pip install md-ai-platform[vampnets])."
            ),
        )

    n_frames = feature_matrix.shape[0]
    if n_frames <= lag_frames + 2:
        return VAMPNetAblation(
            available=False,
            n_states=n_states,
            summary="VAMPnet ablation skipped: not enough frames for a lagged pair set.",
        )

    try:
        torch.manual_seed(SEED)
        net = torch.nn.Sequential(
            torch.nn.Linear(feature_matrix.shape[1], 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, n_states),
            torch.nn.Softmax(dim=1),
        )
        features = torch.tensor(feature_matrix, dtype=torch.float64)
        net = net.double()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        x0 = features[:-lag_frames]
        xt = features[lag_frames:]
        identity = torch.eye(n_states, dtype=torch.float64)
        for _ in range(epochs):
            optimizer.zero_grad()
            chi0 = net(x0)
            chit = net(xt)
            score = _vamp2_score(chi0, chit, identity)
            (-score).backward()
            optimizer.step()

        with torch.no_grad():
            probs = net(features).numpy()
            chi0 = net(x0)
            chit = net(xt)
            vamp2 = float(_vamp2_score(chi0, chit, identity))

        vamp_labels = np.argmax(probs, axis=1)

        # ponytail: reuse the Phase 4 comparison helpers instead of a second MSM path
        from .analysis import _normalized_mutual_information, _transition_timescales

        timescales = _transition_timescales(vamp_labels, lag_frames, lag_ps)
        leading = timescales[0] if timescales else None
        nmi = _normalized_mutual_information(vamp_labels, tica_labels)
        error = None
        if leading is not None and tica_leading_timescale_ps:
            error = abs(leading - tica_leading_timescale_ps) / max(
                leading, tica_leading_timescale_ps
            )

        summary = (
            f"VAMPnet ablation: leading implied timescale "
            f"{leading:.1f} ps vs TICA/MSM {tica_leading_timescale_ps:.1f} ps "
            f"(relative error {error:.2f}); state agreement {nmi:.2f} NMI; "
            f"VAMP-2 score {vamp2:.2f}."
            if leading is not None and tica_leading_timescale_ps and error is not None
            else f"VAMPnet ablation: state agreement {nmi:.2f} NMI; VAMP-2 score {vamp2:.2f}."
        )
        return VAMPNetAblation(
            available=True,
            vamp2_score=vamp2,
            leading_timescale_ps=leading,
            tica_leading_timescale_ps=tica_leading_timescale_ps,
            timescale_relative_error=error,
            state_agreement_nmi=float(nmi),
            n_states=n_states,
            epochs=epochs,
            summary=summary,
        )
    except Exception as exc:  # ablation must never take down the ML layer
        return VAMPNetAblation(
            available=False,
            n_states=n_states,
            summary=f"VAMPnet ablation failed: {exc}",
        )


def _vamp2_score(chi0: "torch.Tensor", chit: "torch.Tensor", identity: "torch.Tensor") -> "torch.Tensor":
    """Differentiable VAMP-2 score: ||C00^-1/2 C0t Ctt^-1/2||_F^2 via Cholesky."""
    n = max(len(chi0) - 1, 1)
    x = chi0 - chi0.mean(dim=0, keepdim=True)
    y = chit - chit.mean(dim=0, keepdim=True)
    eps = 1e-10 * n
    c00 = (x.T @ x) / n + eps * identity
    c0t = (x.T @ y) / n
    ctt = (y.T @ y) / n + eps * identity
    l0 = torch.linalg.cholesky(c00)
    lt = torch.linalg.cholesky(ctt)
    whitened = torch.linalg.solve_triangular(l0, c0t, upper=False) @ torch.linalg.solve_triangular(
        lt, torch.eye(ctt.shape[0], dtype=ctt.dtype), upper=False
    ).T
    return (whitened ** 2).sum()
