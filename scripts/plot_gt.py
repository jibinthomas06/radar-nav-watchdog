from __future__ import annotations

from pathlib import Path
import math
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def load_cfg() -> dict:
    return yaml.safe_load(Path('configs/local.yaml').read_text(encoding='utf-8'))

def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def main() -> None:
    cfg = load_cfg()
    root = Path(cfg['dataset_root'])
    trav = root / cfg['traversal']
    gt_path = trav / 'gt' / 'radar_odometry.csv'
    if not gt_path.exists():
        raise SystemExit(f'Missing: {gt_path}')

    # Read CSV (header may or may not exist)
    try:
        df = pd.read_csv(gt_path)
    except Exception:
        df = pd.read_csv(gt_path, header=None)

    cols = list(df.columns)

    def find_col(keys: list[str]):
        for c in cols:
            s = str(c).lower()
            if any(k in s for k in keys):
                return c
        return None

    cx = find_col(['x'])
    cy = find_col(['y'])
    cyaw = find_col(['gamma', 'yaw'])

    # Fallback to common indexing: [t_src, t_dst, x, y, z, alpha, beta, gamma, ...]
    if cx is None or cy is None or cyaw is None:
        if df.shape[1] < 8:
            raise SystemExit(f'Unexpected columns: {df.shape[1]}')
        cx, cy, cyaw = df.columns[2], df.columns[3], df.columns[7]

    x = 0.0
    y = 0.0
    yaw = 0.0
    xs = [x]
    ys = [y]

    for _, r in df.iterrows():
        dx = float(r[cx])
        dy = float(r[cy])
        dyaw = float(r[cyaw])

        c = math.cos(yaw)
        s = math.sin(yaw)
        x += c * dx - s * dy
        y += s * dx + c * dy
        yaw = wrap_pi(yaw + dyaw)

        xs.append(x)
        ys.append(y)

    plt.figure()
    plt.plot(xs, ys)
    plt.axis('equal')
    plt.title(f'GT radar odometry trajectory: {trav.name}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()

if __name__ == '__main__':
    main()
