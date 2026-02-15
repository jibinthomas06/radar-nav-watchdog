from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    log_path = Path("outputs/m5_watchdog_log.csv")
    if not log_path.exists():
        raise SystemExit(f"Missing log: {log_path}. Run: python scripts/run_watchdog.py")

    # keep_default_na=False so empty switched cells stay as ""
    df = pd.read_csv(log_path, keep_default_na=False)

    conf = df["track_conf"].astype(float).to_numpy()
    steps = df.index.to_numpy()

    switched = df["switched"].astype(str)
    switch_idx = df.index[switched != ""].to_list()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Plot 1: confidence with switch markers
    plt.figure()
    plt.plot(steps, conf)
    for i in switch_idx:
        plt.axvline(i, linestyle="--", linewidth=1)

    plt.xlabel("step")
    plt.ylabel("TRACK confidence")
    plt.title("Milestone 5: Watchdog confidence + switch events")
    out_path = out_dir / "m5_watchdog_conf_switches.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()

    # Plot 2: mode timeline (TRACK=0, RECOVER=1)
    mode_map = {"TRACK": 0, "RECOVER": 1}
    mode_vals = df["mode"].map(mode_map).fillna(0).astype(int).to_numpy()

    plt.figure()
    plt.plot(steps, mode_vals)
    plt.yticks([0, 1], ["TRACK", "RECOVER"])
    plt.xlabel("step")
    plt.ylabel("mode")
    plt.title("Milestone 5: Watchdog mode over time")
    out_path2 = out_dir / "m5_watchdog_mode.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path2}")
    plt.show()


if __name__ == "__main__":
    main()
