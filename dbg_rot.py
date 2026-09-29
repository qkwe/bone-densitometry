import sys, pickle
sys.path.insert(0, ".")
import numpy as np
from PIL import Image as P, ImageDraw, ImageFont
from scipy import ndimage
from dxa import roi, rotation
from dxa.data import LABEL_POSITIONING as L

ims = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "femur" and L in i.labels]
sel = [i for i in ims if i.labels[L]][:7] + [i for i in ims if not i.labels[L]][:7]
f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
T = []
for im in sel:
    lm = roi.landmarks(im.array, im.spacing, im.side)
    a = roi.normalize(im.array); mask = lm["mask"]; hx, hy, r = lm["head"]
    if lm["medial"] == "left":
        a = a[:, ::-1]; mask = mask[:, ::-1]; hx = a.shape[1] - 1 - hx
    ys = np.nonzero(mask.any(1))[0]; y1 = ys.max(); sy = im.spacing[1]
    rows = np.arange(max(y1 - int(40 / sy), int(hy + r)), y1 + 1)
    c = [np.nonzero(mask[q])[0].mean() for q in rows]; k, b = np.polyfit(rows, c, 1)
    left, right = rotation._femur_runs(mask, lambda q: k * q + b)
    rgb = (np.stack([a] * 3, -1) * 255).astype(np.uint8)
    rgb[mask] = (0.7 * rgb[mask] + 0.3 * np.array([0, 120, 255])).astype(np.uint8)
    img = P.fromarray(rgb).resize((a.shape[1] * 2, a.shape[0] * 2)); d = ImageDraw.Draw(img)
    for q in range(len(right)):
        if not np.isnan(right[q]):
            d.point([(right[q] * 2, q * 2), (left[q] * 2, q * 2)], fill=(255, 0, 0))
    d.line([(k * 0 + b) * 2, 0, (k * y1 + b) * 2, y1 * 2], fill=(0, 255, 0))
    d.ellipse([(hx - r) * 2, (hy - r) * 2, (hx + r) * 2, (hy + r) * 2], outline=(255, 255, 0), width=2)
    m = rotation.measure(im.array, im.spacing, im.side)
    d.text((3, 3), f"{'BAD' if im.labels[L] else 'ok'} bump{m['rot_lt_bump_mm']:.0f} gt{m['rot_gt_above_head_mm']:.0f} neck{m['rot_neck_width_mm']:.0f}", font=f, fill=(255, 255, 0))
    T.append(img)
W = P.new("RGB", (570 * 7, 720 * 2))
for i, t in enumerate(T):
    W.paste(t, ((i % 7) * 570, (i // 7) * 720))
W.save("out/rot_debug.png")
