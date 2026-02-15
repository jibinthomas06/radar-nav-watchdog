from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from .recover import MapLocalizer, RecoverParams, RecoverHit
from .tracker import RadarTracker
from .types import Pose2D, TrackResult


class Mode(str, Enum):
    TRACK = "TRACK"
    RECOVER = "RECOVER"


@dataclass(frozen=True)
class WatchdogParams:
    # Switching logic
    low_conf_threshold: float = 0.10
    low_conf_patience: int = 3          # how many bad steps before switching to RECOVER
    recover_accept_score: float = 0.10  # accept if descriptor similarity above this
    recover_accept_resp: float = 0.05   # and translation response above this

    # Recover config
    recover: RecoverParams = field(default_factory=lambda: RecoverParams(keyframe_stride=5, top_k=5))


@dataclass
class WatchdogState:
    mode: Mode = Mode.TRACK
    low_conf_streak: int = 0
    pose: Pose2D = Pose2D(0.0, 0.0, 0.0)


def wrap_pi(a: float) -> float:
    while a > np.pi:
        a -= 2 * np.pi
    while a < -np.pi:
        a += 2 * np.pi
    return float(a)


def integrate_step(pose: Pose2D, dx: float, dy: float, dyaw: float) -> Pose2D:
    c = float(np.cos(pose.yaw))
    s = float(np.sin(pose.yaw))
    x = pose.x + c * dx - s * dy
    y = pose.y + s * dx + c * dy
    yaw = wrap_pi(pose.yaw + dyaw)
    return Pose2D(x, y, yaw)


class TrackRecoverWatchdog:
    def __init__(self, tracker: RadarTracker, localizer: MapLocalizer, wp: WatchdogParams):
        self.tracker = tracker
        self.localizer = localizer
        self.wp = wp
        self.state = WatchdogState()

    def step(self, png1: Path, png2: Path) -> dict:
        """
        One step over consecutive frames.
        Returns a log dict for CSV/debugging.
        """
        t1 = int(png1.stem)
        t2 = int(png2.stem)

        switched_events: list[str] = []

        # --- TRACK step always runs (gives us motion + confidence)
        tr: TrackResult = self.tracker.step(png1, png2)
        pose_pred = integrate_step(self.state.pose, tr.dx, tr.dy, tr.dyaw)

        # Update low confidence streak
        if tr.conf < self.wp.low_conf_threshold or (hasattr(tr, "ok") and not tr.ok):
            self.state.low_conf_streak += 1
        else:
            self.state.low_conf_streak = 0

        recover_hit: Optional[RecoverHit] = None

        # Switch to RECOVER if needed
        if self.state.mode == Mode.TRACK and self.state.low_conf_streak >= self.wp.low_conf_patience:
            self.state.mode = Mode.RECOVER
            switched_events.append("TRACK->RECOVER")

        # --- RECOVER mode: localize current frame (png2) against keyframe map
        if self.state.mode == Mode.RECOVER:
            hits = self.localizer.localize(png2)
            if hits:
                recover_hit = hits[0]

                accept = (recover_hit.score >= self.wp.recover_accept_score) and (
                    recover_hit.resp_trans >= self.wp.recover_accept_resp
                )

                if accept:
                    # Pragmatic correction (no global keyframe poses yet):
                    # apply the relative correction to current pose.
                    corr_pose = integrate_step(self.state.pose, recover_hit.dx, recover_hit.dy, recover_hit.dyaw)
                    self.state.pose = corr_pose

                    self.state.mode = Mode.TRACK
                    self.state.low_conf_streak = 0
                    switched_events.append("RECOVER->TRACK")
                else:
                    # stay in RECOVER; still advance using TRACK prediction so we don't freeze
                    self.state.pose = pose_pred
            else:
                # no hits: keep moving
                self.state.pose = pose_pred
        else:
            # normal TRACK mode: accept predicted pose
            self.state.pose = pose_pred

        return {
            "t1": t1,
            "t2": t2,
            "mode": self.state.mode.value,
            "switched": ";".join(switched_events),  # "" if none
            "track_dx": tr.dx,
            "track_dy": tr.dy,
            "track_dyaw": tr.dyaw,
            "track_conf": tr.conf,
            "pose_x": self.state.pose.x,
            "pose_y": self.state.pose.y,
            "pose_yaw": self.state.pose.yaw,
            "low_conf_streak": self.state.low_conf_streak,
            "recover_score": (recover_hit.score if recover_hit else np.nan),
            "recover_resp": (recover_hit.resp_trans if recover_hit else np.nan),
            "recover_conf_yaw": (recover_hit.conf_yaw if recover_hit else np.nan),
            "recover_dx": (recover_hit.dx if recover_hit else np.nan),
            "recover_dy": (recover_hit.dy if recover_hit else np.nan),
            "recover_dyaw": (recover_hit.dyaw if recover_hit else np.nan),
            "recover_key_ts": (recover_hit.key_timestamp if recover_hit else -1),
        }
