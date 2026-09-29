"""Ротация бедра: AUC каждого измерения (метка «Некорректная укладка» бедра)."""

import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from dxa import rotation
from dxa.data import LABEL_POSITIONING

images = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "femur" and LABEL_POSITIONING in i.labels]
import os
MASKS = os.environ.get("MASKS")  # например out/sam_masks_sam_box.npz
if MASKS:
    dm = np.load(MASKS, allow_pickle=True)
    lookup = dict(zip(dm["keys"], dm["masks"]))
    feats = [rotation.measure(i.array, i.spacing, i.side, lookup[f"{i.study_uid}:{i.image_uid}"]) for i in images]
else:
    feats = [rotation.measure(i.array, i.spacing, i.side) for i in images]
y = np.array([i.labels[LABEL_POSITIONING] for i in images])

if __name__ == "__main__":
    print(f"снимков {len(y)}, нарушений укладки {y.sum()}")
    for key in feats[0]:
        x = np.array([f.get(key, np.nan) for f in feats], dtype=float)
        bad = np.isnan(x)
        x = np.where(bad, np.nanmedian(x), x)
        auc = roc_auc_score(y, x)
        print(f"{key:22s} AUC {auc:.3f} (|0.5-AUC| {abs(auc - .5):.2f})  норма {np.median(x[y == 0]):7.1f}  "
              f"нарушение {np.median(x[y == 1]):7.1f}  пропусков {bad.sum()}")
    pickle.dump([(i.study_uid, i.image_uid, f) for i, f in zip(images, feats)], open("out/rot_features_sam.pkl" if MASKS else "out/rot_features.pkl", "wb"))
