from __future__ import annotations

from pathlib import Path
import yaml

def load_cfg() -> dict:
    cfg_path = Path('configs/local.yaml')
    if not cfg_path.exists():
        raise FileNotFoundError('Missing configs/local.yaml')
    return yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

def main() -> None:
    cfg = load_cfg()
    root = Path(cfg['dataset_root']).expanduser()

    if not root.exists():
        raise SystemExit(f'dataset_root does not exist: {root}')

    traversals = sorted([p for p in root.iterdir() if p.is_dir()])
    print(f'Found {len(traversals)} traversal folders under:\\n  {root}\\n')

    for p in traversals[:10]:
        radar_dir = p / 'radar'
        gt_csv = p / 'gt' / 'radar_odometry.csv'
        n_png = len(list(radar_dir.glob('*.png'))) if radar_dir.exists() else 0
        print(f'- {p.name}')
        print(f'  radar pngs: {n_png}')
        print(f'  has gt/radar_odometry.csv: {gt_csv.exists()}')

    if len(traversals) > 10:
        print('\\n(showing first 10 only)')

if __name__ == '__main__':
    main()
