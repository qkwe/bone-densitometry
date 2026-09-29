"""Оценка области интереса бедра: AUC отступов, правило ТЗ, картинки с разметкой."""

import pickle
import sys

import numpy as np
from PIL import Image as P, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from dxa import roi
from dxa.data import LABEL_ROI

images = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "femur" and LABEL_ROI in i.labels]
feats = [roi.measure(i.array, i.spacing, i.side) for i in images]
y = np.array([i.labels[LABEL_ROI] for i in images])
print(f"снимков {len(y)}, нарушений области интереса {y.sum()}")
for key in feats[0]:
    x = np.array([f[key] for f in feats], dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    auc = roc_auc_score(y, -x)  # меньше отступ — хуже
    print(f"{key:24s} AUC(меньше=хуже) {auc:.3f}   норма {np.median(x[y == 0]):5.1f}  нарушение {np.median(x[y == 1]):5.1f}   "
          f"наруш.: {np.round(np.sort(x[y == 1]), 1)}")
score = np.array([roi.deficit(f) for f in feats])
print(f"дефицит до нормы ТЗ     AUC {roc_auc_score(y, score):.3f}")
pred = np.array([roi.verdict(f)[0] for f in feats])
tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
print(f"правило ТЗ 3/3/2/2: TP {tp} FN {int(y.sum()) - tp} FP {fp} TN {int((y == 0).sum()) - fp}")
for side in ("top", "bottom", "medial", "lateral"):
    lim = roi.MARGIN_SIDE_CM if side in ("medial", "lateral") else roi.MARGIN_TOP_CM
    x = np.array([f[f"roi_margin_{side}_cm"] for f in feats])
    print(f"  {side:8s} < {lim}: у нарушений {int(((x < lim) & (y == 1)).sum())}, у норм {int(((x < lim) & (y == 0)).sum())}")


def draw(im, tag):
    lm = roi.landmarks(im.array, im.spacing, im.side)
    a = roi.normalize(im.array)
    img = P.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    over = np.array(img)
    over[lm["mask"]] = (0.6 * over[lm["mask"]] + 0.4 * np.array([0, 120, 255])).astype(np.uint8)
    edge = lm["field"] ^ ndimage_erode(lm["field"])
    over[edge] = (255, 0, 255)
    img = P.fromarray(over).resize((a.shape[1] * 2, a.shape[0] * 2))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = [v * 2 for v in lm["box"]]
    d.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=2)
    hx, hy, r = lm["head"]
    d.ellipse([(hx - r) * 2, (hy - r) * 2, (hx + r) * 2, (hy + r) * 2], outline=(0, 255, 0), width=2)
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    ft = roi.measure(im.array, im.spacing, im.side)
    d.text((4, 4), f"{tag} {'BAD' if im.labels[LABEL_ROI] else 'ok'}", font=f, fill=(255, 255, 0))
    d.text((4, 24), f"T{ft['roi_margin_top_cm']:.1f} B{ft['roi_margin_bottom_cm']:.1f} "
                    f"M{ft['roi_margin_medial_cm']:.1f} L{ft['roi_margin_lateral_cm']:.1f}", font=f, fill=(255, 255, 0))
    return img


def ndimage_erode(m):
    from scipy import ndimage
    return ndimage.binary_erosion(m, np.ones((3, 3)))


if __name__ == "__main__":
    pos = [k for k in range(len(images)) if y[k] == 1]
    worst_neg = [k for k in np.argsort(-score) if y[k] == 0][:7]
    tiles = [draw(images[k], "") for k in pos + worst_neg]
    w, h = 560, 700
    W = P.new("RGB", (w * 7, h * 2))
    for k, t in enumerate(tiles):
        W.paste(t, ((k % 7) * w, (k // 7) * h))
    W.save("out/roi_debug.png")
