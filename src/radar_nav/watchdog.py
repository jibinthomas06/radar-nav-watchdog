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

    # Safety gates for accepting RECOVER
    max_recover_delta_m: float = 8.0        # gate on recover_hit (dx,dy) magnitude
    max_recover_snap_jump_m: float = 6.0    # gate on |pose_snap - pose_pred| in global frame

    # Recover config (used by MapLocalizer)
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


def pose_dist(a: Pose2D, b: Pose2D) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


class TrackRecoverWatchdog:
    """
    State machine:
      - TRACK: use radar odometry step-by-step (with confidence)
      - RECOVER: run map-based localization; if accepted, snap pose to map-consistent pose

    keyframe_poses:
      dict[key_timestamp] -> global pose of that keyframe in the map frame.
      If provided, RECOVER accept can set:
        pose_snap = key_pose ⊕ delta(key->current)
    """

    def __init__(
        self,
        tracker: RadarTracker,
        localizer: MapLocalizer,
        wp: WatchdogParams,
        keyframe_poses: Optional[dict[int, Pose2D]] = None,
    ):
        self.tracker = tracker
        self.localizer = localizer
        self.wp = wp
        self.keyframe_poses = keyframe_poses or {}
        self.state = WatchdogState()

    def step(self, png1: Path, png2: Path) -> dict:
        t1 = int(png1.stem)
        t2 = int(png2.stem)

        switched_events: list[str] = []

        # TRACK proposal (always computed so we can log it / keep moving)
        tr: TrackResult = self.tracker.step(png1, png2)
        pose_pred = integrate_step(self.state.pose, tr.dx, tr.dy, tr.dyaw)

        # low-confidence streak
        if tr.conf < self.wp.low_conf_threshold or (hasattr(tr, "ok") and not tr.ok):
            self.state.low_conf_streak += 1
        else:
            self.state.low_conf_streak = 0

        recover_hit: Optional[RecoverHit] = None
        recover_pose_snap: Optional[Pose2D] = None
        recover_delta_m = np.nan
        recover_snap_jump_m = np.nan
        recover_accepted = False

        # TRACK -> RECOVER transition
        if self.state.mode == Mode.TRACK and self.state.low_conf_streak >= self.wp.low_conf_patience:
            self.state.mode = Mode.RECOVER
            switched_events.append("TRACK->RECOVER")

        # RECOVER behavior
        if self.state.mode == Mode.RECOVER:
            hits = self.localizer.localize(png2)
            if hits:
                recover_hit = hits[0]
                recover_delta_m = float(np.hypot(recover_hit.dx, recover_hit.dy))

                # Build snapped pose if we have the keyframe global pose
                key_ts = int(recover_hit.key_timestamp)
                key_pose = self.keyframe_poses.get(key_ts)

                if key_pose is not None:
                    recover_pose_snap = integrate_step(key_pose, recover_hit.dx, recover_hit.dy, recover_hit.dyaw)
                    recover_snap_jump_m = pose_dist(recover_pose_snap, pose_pred)
                else:
                    # If we don't know the keyframe global pose, we cannot do a true snap.
                    # We treat this as "no snap": don't accept RECOVER->TRACK based on it.
                    recover_pose_snap = None
                    recover_snap_jump_m = np.nan

                # Acceptance checks
                accept_basic = (recover_hit.score >= self.wp.recover_accept_score) and (
                    recover_hit.resp_trans >= self.wp.recover_accept_resp
                )
                accept_delta = recover_delta_m <= self.wp.max_recover_delta_m

                # Only apply snap-jump gate if we can compute it (key_pose exists)
                accept_snap = True
                if recover_pose_snap is not None:
                    accept_snap = recover_snap_jump_m <= self.wp.max_recover_snap_jump_m
                else:
                    accept_snap = False

                accept = accept_basic and accept_delta and accept_snap

                if accept and recover_pose_snap is not None:
                    self.state.pose = recover_pose_snap
                    self.state.mode = Mode.TRACK
                    self.state.low_conf_streak = 0
                    switched_events.append("RECOVER->TRACK")
                    recover_accepted = True
                else:
                    # keep moving with TRACK while searching
                    self.state.pose = pose_pred
            else:
                # no candidates; keep moving
                self.state.pose = pose_pred

        # TRACK behavior
        else:
            self.state.pose = pose_pred

        # Log row
        return {
            "t1": t1,
            "t2": t2,
            "mode": self.state.mode.value,
            "switched": ";".join(switched_events) if switched_events else "",
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
            "recover_key_ts": (int(recover_hit.key_timestamp) if recover_hit else -1),
            "recover_delta_m": recover_delta_m,
            "recover_snap_jump_m": recover_snap_jump_m,
            "recover_accepted": int(recover_accepted),
        }
