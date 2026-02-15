from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .track_odometry import TrackParams, track_step
from .types import TrackResult


@dataclass
class RadarTracker:
    params: TrackParams

    def step(self, png1: Path, png2: Path) -> TrackResult:
        dx, dy, dyaw, conf = track_step(png1, png2, self.params)

        # Minimal sanity checks (we’ll tune later)
        ok = True
        if not (conf == conf):  # NaN
            ok = False
        if abs(dx) > 20 or abs(dy) > 20:
            ok = False

        return TrackResult(dx=dx, dy=dy, dyaw=dyaw, conf=conf, ok=ok)
