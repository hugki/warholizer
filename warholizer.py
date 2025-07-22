#!/usr/bin/env python3
"""
warholizer.py — Generate Warhol-style pop-art grids from photographs (full version, 2025-05-28)

機能
────
* 任意サイズのパネルを N×M でタイル
* k-means 量子化（人物領域と背景を別クラスタ数に分離可）
* 高彩度パレットを HSV から自動生成
* 自動 Canny 閾値で線画抽出、線幅可変
* オプション: CMY ハーフトーン / 版ズレ
* オプション: 顔検出して自動クロップ、人物強調
* RNG シード指定で再現性
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ──────────────────────────────
#  ユーティリティ
# ──────────────────────────────
def to_square(img: np.ndarray, size: int = 512, blur_sigma: float | None = 1.0) -> np.ndarray:
    """Pad to square, optional Gaussian blur, then resize to (size×size)."""
    h, w = img.shape[:2]
    if h != w:
        pad = abs(h - w) // 2
        if h > w:
            img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_REFLECT)
        else:
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_REFLECT)
    if blur_sigma:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# ── 顔検出（OpenCV Haar cascade） ──
def detect_face_bbox(img: np.ndarray) -> Tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of largest detected face, or *None*."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])  # largest


def auto_crop(img: np.ndarray, expand: float = 1.3) -> np.ndarray:
    """Crop around detected face to zoom in; return original if no face."""
    bbox = detect_face_bbox(img)
    if bbox is None:
        return img
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    size = int(max(w, h) * expand)
    half = size // 2
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, img.shape[1]), min(cy + half, img.shape[0])
    crop = img[y0:y1, x0:x1]
    return crop if crop.size else img


def detect_face_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask: True for face pixels (whole image if no face)."""
    mask = np.ones(img.shape[:2], bool)
    bbox = detect_face_bbox(img)
    if bbox is None:
        return mask
    x, y, w, h = bbox
    mask[:] = False
    mask[y:y + h, x:x + w] = True
    return mask


# ── 線画抽出 ──
def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds derived from median pixel value."""
    med = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(gray, lower, upper)


def extract_lines(gray: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Return mask 1=keep colour, 0=line (black)."""
    edges = auto_canny(gray)
    if thickness > 1:
        kernel = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, 1)
    return (edges == 0).astype(np.uint8)


# ── 色量子化 & パレット ──
def kmeans_quantize(pix: np.ndarray, k: int = 6, attempts: int = 10) -> np.ndarray:
    Z = pix.reshape(-1, 3).astype(np.float32)
    _ret, label, center = cv2.kmeans(
        Z, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        attempts, cv2.KMEANS_PP_CENTERS,
    )
    return center.astype(np.uint8)[label.flatten()].reshape(pix.shape)


def random_palette(k: int, min_sat: int = 150, min_val: int = 150) -> np.ndarray:
    """Return k×3 BGR palette with vivid colours."""
    colours = []
    for _ in range(k):
        h = random.random() * 179
        s = random.uniform(min_sat, 255)
        v = random.uniform(min_val, 255)
        bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colours.append(bgr)
    return np.array(colours, np.uint8)


def apply_palette_by_rank(poster: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map unique posterised colours (sorted dark→light) onto palette."""
    flat = poster.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    order = np.argsort(uniq.sum(1))  # brightness rank
    return palette[order[inv] % len(palette)].reshape(poster.shape)


# ── ハーフトーン & 版ズレ ──
def _halftone_channel(ch: np.ndarray, dot_r: int = 3, angle: int = 45) -> np.ndarray:
    pil = Image.fromarray(ch)
    pil = pil.rotate(angle, expand=True, resample=Image.BICUBIC)
    arr = np.asarray(pil)
    h, w = arr.shape
    out = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(out)
    step = dot_r * 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            block = arr[y:y + step, x:x + step]
            r = int((255 - block.mean()) / 255 * dot_r)
            if r:
                draw.ellipse((x + step//2 - r, y + step//2 - r,
                              x + step//2 + r, y + step//2 + r), fill=0)
    out = out.rotate(-angle, expand=True, resample=Image.BICUBIC)
    cx = (out.width - ch.shape[1]) // 2
    cy = (out.height - ch.shape[0]) // 2
    return np.asarray(out)[cy:cy + ch.shape[0], cx:cx + ch.shape[1]]


def halftone(img: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img)
    return cv2.merge([
        _halftone_channel(b, 3, 15),
        _halftone_channel(g, 3, 75),
        _halftone_channel(r, 3, 0)
    ])


def misregister(img: np.ndarray, max_shift: int = 4) -> np.ndarray:
    channels = cv2.split(img)
    shifted = []
    for ch in channels:
        dx, dy = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted.append(cv2.warpAffine(ch, M, (ch.shape[1], ch.shape[0]), borderMode=cv2.BORDER_REFLECT))
    return cv2.merge(shifted)


# ──────────────────────────────
#  パネル生成
# ──────────────────────────────
def make_panel(
    img: np.ndarray,
    poster_k: int = 6,
    face_k: int = 0,
    bg_k: int = 0,
    size: int = 512,
    line_thickness: int = 2,
    halftone_on: bool = False,
    misregister_on: bool = False,
    auto_crop_on: bool = True,
) -> np.ndarray:
    base = auto_crop(img) if auto_crop_on else img
    base = to_square(base, size)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    line_mask = extract_lines(gray, line_thickness)

    # Colour quantisation
    if face_k and bg_k:                               # dual-depth mode
        fmask = detect_face_mask(base)
        face = kmeans_quantize(base[fmask], face_k)
        bg = kmeans_quantize(base[~fmask], bg_k)
        poster = np.zeros_like(base)
        poster[fmask], poster[~fmask] = face, bg
        poster[fmask] = apply_palette_by_rank(face, random_palette(face_k))
        poster[~fmask] = apply_palette_by_rank(bg, random_palette(bg_k))
    else:                                             # single-depth mode
        poster = kmeans_quantize(base, poster_k)
        poster = apply_palette_by_rank(poster, random_palette(poster_k))

    coloured = poster * line_mask[:, :, None]         # black lines where mask==0
    if misregister_on:
        coloured = misregister(coloured)
    if halftone_on:
        coloured = halftone(coloured)
    return coloured


def build_grid(img: np.ndarray, rows: int, cols: int, **kw) -> np.ndarray:
    panels = [make_panel(img, **kw) for _ in range(rows * cols)]
    rows_imgs = [np.hstack(panels[i*cols:(i+1)*cols]) for i in range(rows)]
    return np.vstack(rows_imgs)


# ──────────────────────────────
#  CLI
# ──────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Warhol-style pop-art grids")
    ap.add_argument("input", type=Path, help="Input image file")
    ap.add_argument("-o", "--output", type=Path, default=Path("warhol_out.png"))
    ap.add_argument("--rows", type=int, default=2, help="Grid rows")
    ap.add_argument("--cols", type=int, default=2, help="Grid columns")
    ap.add_argument("--size", type=int, default=512, help="Panel side before tiling")
    ap.add_argument("--poster_k", type=int, default=6, help="Colours without face split")
    ap.add_argument("--face_k", type=int, default=0, help="Colours for face region (enable split if >0)")
    ap.add_argument("--bg_k", type=int, default=0, help="Colours for background region")
    ap.add_argument("--halftone", action="store_true", help="Apply CMY halftone dots")
    ap.add_argument("--misregister", action="store_true", help="Randomly offset RGB layers")
    ap.add_argument("--line_thickness", type=int, default=2, help="Black line width (after dilation)")
    ap.add_argument("--no_auto_crop", action="store_true", help="Disable face-center crop")
    ap.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = ap.parse_args()

    # RNG determinism
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    img = cv2.imread(str(args.input))
    if img is None:
        raise FileNotFoundError(f"Cannot read {args.input!s}")

    grid = build_grid(
        img,
        rows=args.rows,
        cols=args.cols,
        poster_k=args.poster_k,
        face_k=args.face_k,
        bg_k=args.bg_k,
        size=args.size,
        line_thickness=args.line_thickness,
        halftone_on=args.halftone,
        misregister_on=args.misregister,
        auto_crop_on=not args.no_auto_crop,
    )

    cv2.imwrite(str(args.output), grid)
    print(f"Saved → {args.output.resolve()}")


if __name__ == "__main__":
    main()
