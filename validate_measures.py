"""Поверка измерителя (ТЗ п. 2.7/3.1: погрешность ≤ 3° для углов, ≤ 5% для линейных величин).

Снимок меняется на ИЗВЕСТНУЮ величину, измеренное изменение сравнивается с заданным:
  * ось позвоночника — поворот на ±3, 6, 10°;
  * отступы области интереса бедра — обрезка кадра снизу/сверху на 0.5, 1, 2 см.
Результат: out/validation.json.
"""

import json
import pickle
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from dxa import axis, roi

images = pickle.load(open("out/cache.pkl", "rb"))
res = {}

errs = []
for im in (i for i in images if i.region == "spine"):
    base = axis.measure(im.array, im.spacing)["axis_signed_deg"]
    for d in (-10, -6, -3, 3, 6, 10):
        r = ndimage.rotate(im.array.astype(float), d, reshape=False, order=1, mode="constant", cval=0)
        errs.append(abs((axis.measure(r, im.spacing)["axis_signed_deg"] - base) - d))
e = np.array(errs)
res["ось позвоночника, °"] = dict(n=len(e), mean=float(e.mean()), p95=float(np.percentile(e, 95)),
                                 within_3deg=float((e <= 3).mean()))

for side in ("bottom", "top"):
    rel = []
    for im in (i for i in images if i.region == "femur"):
        base = roi.measure(im.array, im.spacing, im.side)[f"roi_margin_{side}_cm"]
        if not np.isfinite(base):
            continue
        sy = im.spacing[1]
        for cm in (0.5, 1.0, 2.0):
            n = int(round(cm * 10 / sy))
            true = n * sy / 10
            if base - true <= 0.5:
                continue
            arr = im.array[:-n] if side == "bottom" else im.array[n:]
            got = base - roi.measure(arr, im.spacing, im.side)[f"roi_margin_{side}_cm"]
            rel.append(abs(got - true) / base)
    r = np.array(rel)
    res[f"отступ {'снизу' if side == 'bottom' else 'сверху'} (бедро), отн. ошибка"] = dict(
        n=len(r), mean=float(r.mean()), median=float(np.median(r)), within_5pct=float((r <= 0.05).mean()))

for k, v in res.items():
    print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()})
json.dump(res, open("out/validation.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
