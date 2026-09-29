"""Картинка: подсказки и маски SAM / MedSAM / эвристика."""
import sys
import numpy as np
from PIL import Image as P, ImageDraw, ImageFont

names = sys.argv[1:] or ["sam", "medsam"]
d = np.load("out/sam_input.npz", allow_pickle=True)
M = {n: np.load(f"out/sam_masks_{n}.npz", allow_pickle=True) for n in names}
n = min(len(M[k]["masks"]) for k in M)
idx = list(range(min(n, 7)))
f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
cols = {"sam": (0, 140, 255), "medsam": (255, 120, 0), "sam_box": (255, 120, 0)}
rows = []
for name in names:
    tiles = []
    for k in idx:
        a = d["pics"][k]
        rgb = np.stack([a] * 3, -1).astype(float)
        m = M[name]["masks"][k]
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(cols[name])
        img = P.fromarray(rgb.astype(np.uint8)).resize((a.shape[1] * 2, a.shape[0] * 2))
        dr = ImageDraw.Draw(img)
        for (x, y), l in zip(d["points"][k], d["labels"][k]):
            c = (0, 255, 0) if l else (255, 0, 0)
            dr.ellipse([x * 2 - 5, y * 2 - 5, x * 2 + 5, y * 2 + 5], fill=c)
        x0, y0, x1, y1 = [v * 2 for v in d["boxes"][k]]
        dr.rectangle([x0, y0, x1, y1], outline=(255, 255, 0))
        dr.text((4, 4), f"{name} iou {M[name]['scores'][k]:.2f}", font=f, fill=(255, 255, 0))
        tiles.append(img)
    rows.append(tiles)
W = P.new("RGB", (580 * len(idx), 720 * len(rows)))
for r, tiles in enumerate(rows):
    for c, t in enumerate(tiles):
        W.paste(t, (c * 580, r * 720))
W.save("out/sam_view.png")
