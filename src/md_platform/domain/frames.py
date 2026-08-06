"""Trajectory frame windowing.

Every classical module iterates frames through :func:`iter_frames` so that the
``start_frame`` / ``end_frame`` / ``stride`` parameters on an ``AnalysisRequest``
apply uniformly, and so that a module never silently analyses more frames than
the caller asked for.
"""

from dataclasses import dataclass
from typing import Optional

import MDAnalysis as mda


@dataclass(frozen=True)
class FrameWindow:
    """A half-open frame selection ``[start, stop)`` with a stride."""

    start: int = 0
    stop: Optional[int] = None
    step: int = 1

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if self.stop is not None and self.stop <= self.start:
            raise ValueError("stop must be greater than start")

    def resolve(self, n_frames: int) -> "FrameWindow":
        """Clamp this window to a trajectory of ``n_frames``."""
        stop = n_frames if self.stop is None else min(self.stop, n_frames)
        start = min(self.start, max(stop - 1, 0))
        return FrameWindow(start=start, stop=stop, step=self.step)


FULL_TRAJECTORY = FrameWindow()


def iter_frames(universe: mda.Universe, window: Optional[FrameWindow] = None):
    """Return a sliced, re-iterable frame iterator for ``universe``.

    MDAnalysis ``FrameIterator`` objects support ``len()`` and repeated
    iteration, so callers can use the result both to size arrays and to loop.
    """
    win = (window or FULL_TRAJECTORY).resolve(len(universe.trajectory))
    return universe.trajectory[win.start : win.stop : win.step]


def window_kwargs(universe: mda.Universe, window: Optional[FrameWindow] = None) -> dict:
    """Window as ``start``/``stop``/``step`` kwargs for MDAnalysis ``run()``."""
    win = (window or FULL_TRAJECTORY).resolve(len(universe.trajectory))
    return {"start": win.start, "stop": win.stop, "step": win.step}
