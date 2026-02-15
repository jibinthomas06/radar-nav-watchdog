from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from radar_nav.recover import MapLocalizer
from radar_nav.track_odometry import TrackParams
from radar_nav.tracker import RadarTracker
from radar_nav.watchdog import TrackRecoverWatchdog, WatchdogParams


def load_cfg() -> dict:
    return yaml.safe_load(Path("configs/local.yaml").read_text(encoding="utf-8"))


def main() -> None:
    cfg = load_cfg()
    trav = Path(cfg["dataset_root"]) / cfg["traversal"]
    radar_dir = trav / "radar"

    pngs = sorted(radar_dir.glob("*.png"))
    if len(pngs) < 2:
        raise SystemExit("Need at least 2 radar frames.")

    tracker = RadarTracker(
        TrackParams(cart_res_m=0.20, max_range_m=80.0, highpass_blur=9)
    )
    localizer = MapLocalizer(radar_dir, WatchdogParams().recover)



    wd = TrackRecoverWatchdog(tracker, localizer, WatchdogParams(
        low_conf_threshold=0.18,
        low_conf_patience=2,
        recover_accept_score=0.08,
        recover_accept_resp=0.01,
    ))


    logs = []
    for a, b in zip(pngs[:-1], pngs[1:]):
        row = wd.step(a, b)
        logs.append(row)

    df = pd.DataFrame(logs)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "m5_watchdog_log.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # show summary of switches
    switches = df[df["switched"] != ""]
    print(f"Switch events: {len(switches)}")
    if len(switches) > 0:
        print(switches[["t1", "t2", "switched", "track_conf", "recover_score", "recover_resp"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
