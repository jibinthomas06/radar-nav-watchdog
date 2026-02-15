from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class TrackResult:
    dx: float
    dy: float
    dyaw: float
    conf: float
    ok: bool
