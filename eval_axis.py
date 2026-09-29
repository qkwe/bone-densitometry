"""Оценка измерителя оси: AUC каждого измерения, правило ТЗ, картинки с линией."""

import pickle
import sys

import numpy as np
from PIL import Image as P, ImageDraw
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from dxa import axis
from dxa.data import LABEL_AXIS, LABEL_POSITIONING, LABEL_FOREIGN

images = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "spine"]
rows = []
for im in images:
    f = axis.measure(im.array, im.spacing)
    rows.append((im, f))

y = np.array([im.labels.get(LABEL_AXIS, 0) for im, _ in rows])
print(f"снимков {len(y)}, нарушений оси {y.sum()}")
for key in rows[0][1]:
    if key == "axis_signed_deg":
        continue
    x = np.array([f[key] for _, f in rows])
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    print(f"{key:24s} AUC {roc_auc_score(y, x):.3f}   норма {np.median(x[y == 0]):5.1f}  нарушение {np.median(x[y == 1]):5.1f}")

for label in (LABEL_POSITIONING, LABEL_FOREIGN):
    yy = np.array([im.labels.get(label, 0) for im, _ in rows])
    x = np.array([f["axis_deg"] for _, f in rows])
    print(f"axis_deg vs {label}: AUC {roc_auc_score(yy, np.nan_to_num(x)):.3f}")

pred = np.array([axis.verdict(f)[0] for _, f in rows])
tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
print(f"правило ТЗ >5°: TP {tp} FP {fp} FN {fn} TN {tn}; чувств. {tp/max(tp+fn,1):.2f} спец. {tn/max(tn+fp,1):.2f}")


def draw(im, label):
    a = axis.normalize(im.array)
    img = P.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    ys, xs = axis.centerline(im.array, im.spacing)
    d = ImageDraw.Draw(img)
    d.point(list(zip(xs, ys)), fill=(255, 60, 60))
    s, b = axis._huber_line(ys, xs)
    d.line([(s * ys[0] + b, ys[0]), (s * ys[-1] + b, ys[-1])], fill=(60, 255, 60), width=1)
    d.text((3, 3), label, fill=(255, 255, 0))
    return img.resize((img.width * 2 // 3, img.height * 2 // 3))


order = np.argsort([-f["axis_deg"] for _, f in rows])
pos = [k for k in range(len(rows)) if y[k] == 1]
pick = pos + [k for k in order if y[k] == 0][:10]
tiles = [draw(rows[k][0], f"{'BAD' if y[k] else 'ok'} {rows[k][1]['axis_deg']:.1f}") for k in pick]
W = P.new("RGB", (tiles[0].width * 10, max(t.height for t in tiles) * 2))
for k, t in enumerate(tiles):
    W.paste(t, ((k % 10) * tiles[0].width, (k // 10) * max(t.height for t in tiles)))
W.save("out/axis_lines.png")
pickle.dump([(im.study_uid, im.image_uid, f) for im, f in rows], open("out/axis_features.pkl", "wb"))
