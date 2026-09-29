"""Отладка: медиальный контур по маске SAM и найденный выступ малого вертела."""
import sys, pickle
sys.path.insert(0, ".")
import numpy as np
from PIL import Image as P, ImageDraw, ImageFont
from scipy import ndimage
from dxa import roi, rotation
from dxa.data import LABEL_POSITIONING as L

ims = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "femur" and L in i.labels]
dm = np.load("out/sam_masks_sam_box.npz", allow_pickle=True)
masks = dict(zip(dm["keys"], dm["masks"]))
sel = [i for i in ims if i.labels[L]][:5] + [i for i in ims if not i.labels[L]][:5]
f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
T = []
for im in sel:
    lm = roi.landmarks(im.array, im.spacing, im.side)
    mask = masks[f"{im.study_uid}:{im.image_uid}"]
    lab, n = ndimage.label(mask); s = np.bincount(lab.ravel()); s[0] = 0; mask = lab == s.argmax()
    a = roi.normalize(im.array); hx, hy, r = lm["head"]
    if lm["medial"] == "left":
        a = a[:, ::-1]; mask = mask[:, ::-1]; hx = a.shape[1] - 1 - hx
    sy = im.spacing[1]; y1 = np.nonzero(mask.any(1))[0].max()
    rows_ = np.arange(max(y1 - int(40 / sy), int(hy + r)), y1 + 1)
    k, b = np.polyfit(rows_, [np.nonzero(mask[q])[0].mean() for q in rows_], 1)
    left, right = rotation._femur_runs(mask, lambda q: k * q + b)
    start, stop = int(hy + r * 0.6), int(rows_[0] + (y1 - rows_[0]) * 0.3)
    band = np.arange(start, stop); band = band[~np.isnan(right[band])]
    edge = ndimage.median_filter(right[band], 3)
    trend = ndimage.gaussian_filter1d(edge, sigma=10 / sy, mode="nearest")
    bump = (edge - trend) * im.spacing[0]; p = int(np.argmax(bump))
    rgb = (np.stack([a] * 3, -1) * 255).astype(np.uint8)
    rgb[mask] = (0.7 * rgb[mask] + 0.3 * np.array([0, 140, 255])).astype(np.uint8)
    img = P.fromarray(rgb).resize((a.shape[1] * 2, a.shape[0] * 2)); d = ImageDraw.Draw(img)
    d.line([(e * 2, q * 2) for e, q in zip(edge, band)], fill=(255, 60, 60), width=2)
    d.line([(t * 2, q * 2) for t, q in zip(trend, band)], fill=(255, 255, 0), width=1)
    d.ellipse([edge[p] * 2 - 6, band[p] * 2 - 6, edge[p] * 2 + 6, band[p] * 2 + 6], outline=(0, 255, 0), width=2)
    d.text((3, 3), f"{'BAD' if im.labels[L] else 'ok'} bump {bump.max():.0f}mm", font=f, fill=(255, 255, 0))
    T.append(img)
W = P.new("RGB", (580 * 5, 720 * 2))
for i, t in enumerate(T):
    W.paste(t, ((i % 5) * 580, (i // 5) * 720))
W.save("out/lt_debug.png")
