"""Подсказки для SAM/MedSAM из геометрических ориентиров (окружение проекта).

Положительные точки: центр головки, шейка, большой вертел, диафиз.
Отрицательные: крыша вертлужной впадины над головкой и седалищная кость медиально-ниже.
Результат: out/sam_input.npz (снимки uint8, точки, метки точек, рамки).
"""

import pickle
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from dxa import roi

images = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "femur"]
pics, points, labels, boxes, keys = [], [], [], [], []
for im in images:
    a = roi.normalize(im.array)
    lm = roi.landmarks(im.array, im.spacing, im.side)
    hx, hy, r = lm["head"]
    mask = lm["mask"]
    rows, cols = mask.shape
    sign = 1 if lm["medial"] == "right" else -1
    ys = np.nonzero(mask.any(axis=1))[0]
    y1 = int(ys.max())
    dist = ndimage.distance_transform_edt(mask)

    def snap(x, y, reach=8):
        """Притянуть точку в глубину кости: максимум карты расстояний в окрестности."""
        x, y = int(np.clip(x, 0, cols - 1)), int(np.clip(y, 0, rows - 1))
        y0, x0 = max(y - reach, 0), max(x - reach, 0)
        win = dist[y0 : y + reach + 1, x0 : x + reach + 1]
        dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
        return float(x0 + dx), float(y0 + dy)

    def shaft_at(y):
        xs = np.nonzero(mask[int(y)])[0]
        return (float(xs.mean()), float(len(xs))) if len(xs) else (hx - sign * 2.5 * r, 20.0)

    y_lo, y_mid = y1 - (y1 - hy) * 0.15, (y1 + hy) / 2 + r * 0.5
    (sx_lo, w_lo), (sx_mid, _) = shaft_at(y_lo), shaft_at(y_mid)
    pos = [snap(hx, hy), snap((hx + sx_mid) / 2, hy + 0.8 * r), snap(sx_mid, y_mid), snap(sx_lo, y_lo)]
    # мягкие ткани по обе стороны диафиза и таз над головкой — «не бедро»
    gap = w_lo / 2 + 12
    neg = [(hx, hy - 1.5 * r), (sx_lo - gap, y_lo), (sx_lo + gap, y_lo), (hx + sign * 1.3 * r, hy + 2.0 * r)]
    pts = [(float(np.clip(x, 0, cols - 1)), float(np.clip(y, 0, rows - 1))) for x, y in pos + neg]
    x0 = min(hx + sign * 1.2 * r, lm["gt_lateral"] - sign * 0.1 * r)
    x1 = max(hx + sign * 1.2 * r, lm["gt_lateral"] - sign * 0.1 * r)
    box = [float(np.clip(x0, 0, cols - 1)), float(np.clip(hy - 1.2 * r, 0, rows - 1)),
           float(np.clip(x1, 0, cols - 1)), float(y1)]
    pics.append((a * 255).astype(np.uint8))
    points.append(pts)
    labels.append([1] * len(pos) + [0] * len(neg))
    boxes.append(box)
    keys.append(f"{im.study_uid}:{im.image_uid}")

obj = np.empty(len(pics), dtype=object)
obj[:] = pics
np.savez("out/sam_input.npz", pics=obj, points=np.array(points), labels=np.array(labels),
         boxes=np.array(boxes), keys=np.array(keys))
print(len(pics), "снимков")
