from __future__ import annotations

import bisect
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from radar_nav.track_odometry import TrackParams, track_step


def load_cfg() -> dict:
    return yaml.safe_load(Path("configs/local.yaml").read_text(encoding="utf-8"))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def build_gt_pairs(trav_dir: Path):
    gt_path = trav_dir / "gt" / "radar_odometry.csv"
    if not gt_path.exists():
        return None

    df = pd.read_csv(gt_path)

    def pick(name_candidates):
        # exact match first
        for cand in name_candidates:
            for c in df.columns:
                if cand == str(c).lower():
                    return c
        # substring match fallback
        for cand in name_candidates:
            for c in df.columns:
                if cand in str(c).lower():
                    return c
        return None

    # Prefer radar-specific timestamps
    c_src = pick(["source_radar_timestamp"])
    c_dst = pick(["destination_radar_timestamp"])

    # Fallback to generic timestamps if radar ones are absent
    if c_src is None or c_dst is None:
        c_src = pick(["source_timestamp", "source"])
        c_dst = pick(["destination_timestamp", "destination"])

    c_x = pick(["x", "dx"])
    c_y = pick(["y", "dy"])
    c_yaw = pick(["yaw", "gamma", "dyaw"])

    if c_src is None or c_dst is None or c_x is None or c_y is None or c_yaw is None:
        return None

    pairs = []
    for _, r in df.iterrows():
        t1 = int(r[c_src])
        t2 = int(r[c_dst])
        dx = float(r[c_x])
        dy = float(r[c_y])
        dyaw = float(r[c_yaw])
        pairs.append((t1, t2, dx, dy, dyaw))

    pairs.sort(key=lambda x: x[0])
    return pairs


def match_gt_delta(t1: int, t2: int, gt_pairs, tol_us: int = 5000):
    """
    Match a consecutive radar frame pair (t1 -> t2).

    GT may be stored either as:
      forward:  (src=t1, dst=t2) gives (dx, dy, dyaw)
      reverse:  (src=t2, dst=t1) gives motion (t2 -> t1), so we invert it

    tol_us is tolerance in timestamp units (these look like microseconds in RobotCar).
    """
    if gt_pairs is None or len(gt_pairs) == 0:
        return None

    srcs = [p[0] for p in gt_pairs]
    i = bisect.bisect_left(srcs, t1)

    cand_idx = [i - 2, i - 1, i, i + 1, i + 2]

    best = None
    best_mode = None  # "forward" or "reverse"
    best_err = None

    def consider(s1, s2, dx, dy, dyaw, mode):
        nonlocal best, best_err, best_mode
        if mode == "forward":
            err = abs(s1 - t1) + abs(s2 - t2)
        else:
            err = abs(s1 - t2) + abs(s2 - t1)
        if best_err is None or err < best_err:
            best_err = err
            best = (s1, s2, dx, dy, dyaw)
            best_mode = mode

    for j in cand_idx:
        if j < 0 or j >= len(gt_pairs):
            continue
        s1, s2, dx, dy, dyaw = gt_pairs[j]
        consider(s1, s2, dx, dy, dyaw, "forward")
        consider(s1, s2, dx, dy, dyaw, "reverse")

    if best is None:
        return None

    s1, s2, dx, dy, dyaw = best

    if best_mode == "forward":
        if abs(s1 - t1) <= tol_us and abs(s2 - t2) <= tol_us:
            return (dx, dy, dyaw)
        return None

    # reverse mode: row describes (t2 -> t1). Invert to get (t1 -> t2).
    if abs(s1 - t2) <= tol_us and abs(s2 - t1) <= tol_us:
        return (-dx, -dy, -dyaw)

    return None


def integrate(deltas):
    x = 0.0
    y = 0.0
    yaw = 0.0
    xs = [x]
    ys = [y]

    for dx, dy, dyaw in deltas:
        c = math.cos(yaw)
        s = math.sin(yaw)
        x += c * dx - s * dy
        y += s * dx + c * dy
        yaw = wrap_pi(yaw + dyaw)
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def main() -> None:
    cfg = load_cfg()
    root = Path(cfg["dataset_root"])
    trav = root / cfg["traversal"]
    radar_dir = trav / "radar"

    pngs = sorted(radar_dir.glob("*.png"))
    if len(pngs) < 2:
        raise SystemExit("Need at least 2 radar frames.")

    gt_pairs = build_gt_pairs(trav)

    p = TrackParams(
        cart_res_m=0.20,
        max_range_m=80.0,
        highpass_blur=9,
    )

    est_deltas = []
    gt_deltas = []
    confs = []

    for a, b in zip(pngs[:-1], pngs[1:]):
        dx, dy, dyaw, conf = track_step(a, b, p)
        est_deltas.append((dx, dy, dyaw))
        confs.append(conf)

        t1 = int(a.stem)
        t2 = int(b.stem)
        gt = match_gt_delta(t1, t2, gt_pairs, tol_us=5000)
        gt_deltas.append(gt if gt is not None else (0.0, 0.0, 0.0))

    matched = sum(1 for d in gt_deltas if d != (0.0, 0.0, 0.0))
    print(f"GT matched pairs: {matched}/{len(gt_deltas)}")
    print("First 5 GT deltas:", gt_deltas[:5])
    print("First 5 EST deltas:", est_deltas[:5])

    ex, ey = integrate(est_deltas)
    gx, gy = integrate(gt_deltas)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    pos_err = np.sqrt((ex - gx) ** 2 + (ey - gy) ** 2)
    print(f"Position error: mean={pos_err.mean():.3f} m  max={pos_err.max():.3f} m")

    rows = []
    for k in range(len(est_deltas)):
        rows.append(
            {
                "step": k,
                "t1": int(pngs[k].stem),
                "t2": int(pngs[k + 1].stem),
                "est_dx": est_deltas[k][0],
                "est_dy": est_deltas[k][1],
                "est_dyaw": est_deltas[k][2],
                "gt_dx": gt_deltas[k][0],
                "gt_dy": gt_deltas[k][1],
                "gt_dyaw": gt_deltas[k][2],
                "confidence": confs[k],
                "pos_err_end": float(pos_err[k + 1]),
            }
        )

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "m2_track_log.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print(f"Frames: {len(pngs)}  Steps: {len(est_deltas)}")
    print(f"Conf: min={min(confs):.3f}  mean={sum(confs)/len(confs):.3f}  max={max(confs):.3f}")

    plt.figure()
    plt.plot(gx, gy, label="GT (matched pairs)")
    plt.plot(ex, ey, label="TRACK estimate")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Milestone 2: TRACK odometry vs GT")
    plt.legend()
    traj_path = out_dir / "m2_track_vs_gt.png"
    plt.savefig(traj_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {traj_path}")
    plt.show()

    plt.figure()
    plt.plot(confs)
    plt.xlabel("step")
    plt.ylabel("confidence")
    plt.title("TRACK confidence per step")
    conf_path = out_dir / "m2_track_confidence.png"
    plt.savefig(conf_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {conf_path}")
    plt.show()


if __name__ == "__main__":
    main()
