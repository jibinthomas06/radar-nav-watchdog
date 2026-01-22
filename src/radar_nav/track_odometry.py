from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class TrackParams:
    # Dataset properties (Oxford Radar RobotCar)
    range_res_m: float = 0.0432  # 4.32 cm per range bin (approx)  :contentReference[oaicite:1]{index=1}
    azimuth_bins: int = 400      # rows in polar image :contentReference[oaicite:2]{index=2}

    # For cartesian conversion + odometry
    cart_res_m: float = 0.20     # meters per pixel in cart grid (coarser = faster)
    max_range_m: float = 80.0    # limit range for stability/speed (meters)
    highpass_blur: int = 9       # odd kernel size for high-pass (0 disables)

    # Confidence thresholds (used later by watchdog)
    min_resp_trans: float = 0.10
    min_conf_yaw: float = 1.2


def read_radar_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read radar png: {path}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D polar radar image, got shape={img.shape} for {path}")
    return img


def _to_float(img_u8: np.ndarray) -> np.ndarray:
    # log(1 + x) helps compress bright returns
    f = img_u8.astype(np.float32)
    return np.log1p(f)


def polar_to_cart(polar_u8: np.ndarray, p: TrackParams) -> np.ndarray:
    """
    Convert polar image (azimuth rows, range cols) to a cartesian grid using cv2.remap.
    Output is float32.
    """
    H, W = polar_u8.shape
    az_bins = H
    max_bins = min(W, int(p.max_range_m / p.range_res_m))
    max_r_m = max_bins * p.range_res_m

    # Cartesian image size in pixels
    half = int(np.ceil(max_r_m / p.cart_res_m))
    size = 2 * half + 1  # square
    # Cartesian coordinates in meters
    xs = (np.arange(size, dtype=np.float32) - half) * p.cart_res_m
    ys = (np.arange(size, dtype=np.float32) - half) * p.cart_res_m
    X, Y = np.meshgrid(xs, ys)

    R = np.sqrt(X * X + Y * Y)  # meters
    Theta = np.arctan2(Y, X)    # [-pi, pi]
    Theta = np.where(Theta < 0, Theta + 2 * np.pi, Theta)  # [0, 2pi)

    # Map to polar indices
    map_x = (R / p.range_res_m).astype(np.float32)          # range -> x (cols)
    map_y = (Theta / (2 * np.pi) * az_bins).astype(np.float32)  # azimuth -> y (rows)

    # Anything beyond max range becomes invalid
    invalid = (map_x < 0) | (map_x >= max_bins)
    map_x = np.where(invalid, -1.0, map_x)
    map_y = np.where(invalid, -1.0, map_y)

    polar_crop = polar_u8[:, :max_bins]
    polar_f = _to_float(polar_crop)

    cart = cv2.remap(
        polar_f,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return cart


def preprocess_cart(cart_f: np.ndarray, p: TrackParams) -> np.ndarray:
    img = cart_f.copy()

    # High-pass to suppress static background / radial bias
    if p.highpass_blur and p.highpass_blur >= 3 and (p.highpass_blur % 2 == 1):
        blur = cv2.GaussianBlur(img, (p.highpass_blur, p.highpass_blur), 0)
        img = img - blur

    # Normalize
    std = float(img.std())
    if std > 1e-6:
        img = (img - float(img.mean())) / std
    else:
        img = img - float(img.mean())

    return img.astype(np.float32)


def estimate_translation_m(cart1: np.ndarray, cart2: np.ndarray, p: TrackParams) -> Tuple[float, float, float]:
    """
    Estimate translation using phase correlation.
    Returns (dx_m, dy_m, response).
    """
    # OpenCV expects float32
    a = cart1.astype(np.float32)
    b = cart2.astype(np.float32)

    # Hanning window improves phase correlation
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)

    (dx_px, dy_px), resp = cv2.phaseCorrelate(a, b, win)

    # phaseCorrelate gives shift to align a->b in pixel coords (x right, y down)
    dx_m = float(dx_px) * p.cart_res_m
    dy_m = float(dy_px) * p.cart_res_m
    return dx_m, dy_m, float(resp)


def estimate_yaw_rad(polar1_u8: np.ndarray, polar2_u8: np.ndarray, p: TrackParams) -> Tuple[float, float]:
    """
    Estimate yaw change via circular shift in azimuth dimension using 1D FFT correlation.
    Returns (dyaw_rad, confidence_ratio).
    """
    # Use azimuth profile: mean over range bins (and log-compress first)
    a = _to_float(polar1_u8).mean(axis=1)
    b = _to_float(polar2_u8).mean(axis=1)

    a = a - a.mean()
    b = b - b.mean()

    # Circular cross-correlation via FFT
    Fa = np.fft.rfft(a)
    Fb = np.fft.rfft(b)
    corr = np.fft.irfft(Fa * np.conj(Fb), n=a.shape[0])

    k = int(np.argmax(corr))
    peak = float(corr[k])
    # next-best peak for a crude confidence ratio
    corr2 = corr.copy()
    corr2[k] = -np.inf
    second = float(np.max(corr2))
    conf = (peak / (second + 1e-6)) if second > 0 else (peak / 1e-6)

    # Convert shift bins -> radians (rows correspond to azimuth angle)
    # Allow negative wrap (choose smallest magnitude shift)
    H = a.shape[0]
    shift = k if k <= H // 2 else k - H
    dyaw = float(shift) * (2 * np.pi / H)

    return dyaw, conf


def track_step(png1: Path, png2: Path, p: TrackParams) -> Tuple[float, float, float, float]:
    """
    Returns (dx_m, dy_m, dyaw_rad, confidence)
    confidence is a combined score (simple for now).
    """
    pol1 = read_radar_png(png1)
    pol2 = read_radar_png(png2)

    dyaw, conf_yaw = estimate_yaw_rad(pol1, pol2, p)

    cart1 = preprocess_cart(polar_to_cart(pol1, p), p)
    cart2 = preprocess_cart(polar_to_cart(pol2, p), p)

    dx_m, dy_m, resp = estimate_translation_m(cart1, cart2, p)

    # Simple combined confidence: translation response scaled by yaw confidence
    conf = float(resp) * float(np.clip(conf_yaw, 0.5, 5.0))
    return dx_m, dy_m, dyaw, conf
