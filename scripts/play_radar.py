from __future__ import annotations

from pathlib import Path
import time
import yaml
import cv2

def load_cfg() -> dict:
    return yaml.safe_load(Path('configs/local.yaml').read_text(encoding='utf-8'))

def main() -> None:
    cfg = load_cfg()
    root = Path(cfg['dataset_root'])
    trav = root / cfg['traversal']
    radar_dir = trav / 'radar'

    if not radar_dir.exists():
        raise SystemExit(f'Missing radar dir: {radar_dir}')

    pngs = sorted(radar_dir.glob('*.png'))
    if not pngs:
        raise SystemExit(f'No radar pngs found in: {radar_dir}')

    print(f'Traversal: {trav.name}')
    print(f'Radar frames: {len(pngs)}')
    print('Controls: q quit | space pause/resume')

    paused = False
    dt = 1.0 / 10.0  # 10 Hz-ish playback

    for fp in pngs:
        while True:
            if not paused:
                img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f'Failed to read: {fp.name}')
                    break

                # Normalize for display (polar scan)
                if len(img.shape) == 2:
                    vis = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
                else:
                    vis = img

                cv2.imshow('Oxford Radar (polar)', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return
                if key == ord(' '):
                    paused = True
                time.sleep(dt)
                break
            else:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q'):
                    return
                if key == ord(' '):
                    paused = False

if __name__ == '__main__':
    main()
