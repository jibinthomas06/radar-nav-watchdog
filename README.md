# radar-nav-watchdog

A small, reproducible radar-navigation prototype built on the **Oxford Radar RobotCar** dataset.

The project implements a simple reliability idea used in real robotics stacks:

- **TRACK mode:** estimate motion step-by-step from consecutive radar frames (radar odometry).
- **RECOVER mode:** when tracking confidence drops, try to **re-localize** against a small “map” of keyframes using image descriptors, then snap back to a map-consistent pose.
- A **watchdog state machine** decides when to switch modes and logs everything for debugging.

This repo is intentionally lightweight: it focuses on the core logic, logging, and plots.



## What you can do with it

### Milestone 1: dataset sanity checks
- Detect traversal folders and radar frames
- Play radar frames
- Plot the provided GT trajectory from the dataset

### Milestone 2: TRACK odometry vs GT
- Convert polar radar to cartesian
- Estimate translation via phase correlation
- Estimate yaw via azimuth-shift correlation
- Integrate deltas to produce a trajectory
- Compare to GT and log per-step results

Outputs:
- `outputs/m2_track_vs_gt.png`
- `outputs/m2_track_confidence.png`
- `outputs/m2_track_log.csv`

### Milestone 5–6: Watchdog (TRACK <-> RECOVER)
- Build a descriptor “map” from radar frames
- For a query frame, find best matching keyframes (RECOVER candidates)
- Switch modes based on confidence + acceptance thresholds
- Gate RECOVER snaps to avoid unrealistic global jumps
- Plot mode timeline, switches, and trajectory vs GT

Outputs:
- `outputs/m5_watchdog_log.csv`
- `outputs/m5_watchdog_mode.png`
- `outputs/m5_watchdog_conf_switches.png`
- `outputs/m6_watchdog_log.csv`
- `outputs/m6_traj_vs_gt.png`
- `outputs/m6_switch_points.png`




