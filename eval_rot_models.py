"""Ротация/укладка бедра: геометрия против признаков предобученных моделей."""

import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from dxa import cv
from dxa.data import LABEL_POSITIONING

images = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "femur" and LABEL_POSITIONING in i.labels]
keys = [f"{i.study_uid}:{i.image_uid}" for i in images]
y = np.array([i.labels[LABEL_POSITIONING] for i in images])
groups = np.array([i.study_uid for i in images])

rot = {f"{s}:{u}": f for s, u, f in pickle.load(open("out/rot_features.pkl", "rb"))}
# по смыслу: наружная ротация укорачивает шейку в проекции и приближает головку к оси диафиза
geo_keys = ["rot_offset_mm", "rot_neck_len_mm", "rot_neck_angle_deg"]
geo = np.array([[rot[k][g] for g in geo_keys] for k in keys])
copies = np.array([[float(i.copies > 1)] for i in images])


def emb(tag):
    d = np.load(f"out/emb_{tag}.npz", allow_pickle=True)
    lookup = dict(zip(d["keys"], d["emb"]))
    return np.array([lookup[k] for k in keys])


def run(name, X, pca=None):
    p = cv.oof(X, y, groups, make=lambda: cv.linear(pca))
    auc, lo, hi = cv.auc_ci(y, p, groups)
    print(f"{name:34s} AUC {auc:.3f} [{lo:.2f}; {hi:.2f}]", flush=True)
    return p


if __name__ == "__main__":
    print(f"снимков {len(y)}, нарушений {y.sum()}")
    run("геометрия (офсет, шейка)", geo)
    preds = {}
    for crop in ("full", "center", "lesser"):
        E = emb(f"siglip_{crop}")
        preds[crop] = run(f"siglip {crop} PCA16", E, 16)
    both = np.hstack([emb("siglip_full"), emb("siglip_lesser")])
    run("siglip full+lesser PCA16", both, 16)
    avg = np.mean([preds[c] for c in preds], axis=0)
    print(f"{'среднее по кропам':34s} AUC {cv.auc_ci(y, avg, groups)[0]:.3f}")
    run("siglip full PCA16 + копии", np.hstack([emb("siglip_full"), copies * 10]), 16)
    np.save("out/oof_rot_siglip.npy", preds["full"])
