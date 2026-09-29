"""Признаки предобученных моделей (SigLIP, CLIP) для снимков бедра.

Снимок приводится к единой ориентации (медиально справа), дополняется до квадрата.
Два шага: снимки готовит окружение проекта (pydicom), признаки считает Python с torch:
    .venv python embed.py dump
    python310 embed.py
Результат: out/emb_<model>.npz с ключами keys, emb.
"""

import pickle
import sys

import numpy as np
from PIL import Image as P


MODELS = {
    "siglip": "google/siglip-so400m-patch14-384",
    "clip": "openai/clip-vit-base-patch32",
}


def canonical(im) -> P.Image:
    a = im.array.astype(np.float32)
    lo, hi = np.percentile(a, (0.5, 99.5))
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if im.side == "l":  # у левого бедра на снимке медиальная сторона слева
        a = a[:, ::-1]
    h, w = a.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.float32)
    sq[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = a
    return P.fromarray((sq * 255).astype(np.uint8)).convert("RGB")


# кропы в единой ориентации (медиально справа), доли высоты/ширины исходного кадра
CROPS = {
    "full": None,
    "center": (0.1, 0.9, 0.15, 0.95),      # проксимальный отдел без краёв кадра
    "lesser": (0.3, 0.85, 0.35, 1.0),      # медиальный контур ниже шейки — зона малого вертела
}


def crop(im, box) -> P.Image:
    if box is None:
        return canonical(im)
    a = im.array.astype(np.float32)
    lo, hi = np.percentile(a, (0.5, 99.5))
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if im.side == "l":
        a = a[:, ::-1]
    h, w = a.shape
    y0, y1, x0, x1 = box
    a = a[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]
    return P.fromarray((a * 255).astype(np.uint8)).convert("RGB")


def dump_pics():
    sys.path.insert(0, ".")
    spine = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "spine"]
    pics = np.empty(len(spine), dtype=object)
    for k, i in enumerate(spine):
        pics[k] = np.array(canonical(i).convert("L"))  # у позвоночника side=None — без отражения
    np.savez("out/spine_pics_full.npz", keys=np.array([f"{i.study_uid}:{i.image_uid}" for i in spine]), pics=pics)
    images = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "femur"]
    keys = np.array([f"{i.study_uid}:{i.image_uid}" for i in images])
    for name, box in CROPS.items():
        pics = np.empty(len(images), dtype=object)
        for k, i in enumerate(images):
            pics[k] = np.array(crop(i, box).convert("L"))
        np.savez(f"out/femur_pics_{name}.npz", keys=keys, pics=pics)


def main(names):
    # снимки готовит dump_pics() в окружении проекта (там pydicom), здесь только torch
    import torch
    from transformers import AutoImageProcessor, AutoModel

    for name in names:
        proc = AutoImageProcessor.from_pretrained(MODELS[name])
        model = AutoModel.from_pretrained(MODELS[name]).eval()
        if "--spine" in sys.argv:
            data = np.load("out/spine_pics_full.npz", allow_pickle=True)
            embed_set(model, proc, [P.fromarray(p).convert("RGB") for p in data["pics"]], data["keys"], f"{name}_spine")
            continue
        for crop_name in CROPS:
            data = np.load(f"out/femur_pics_{crop_name}.npz", allow_pickle=True)
            keys = data["keys"]
            pics = [P.fromarray(p).convert("RGB") for p in data["pics"]]
            embed_set(model, proc, pics, keys, f"{name}_{crop_name}")


def embed_set(model, proc, pics, keys, tag):
    import torch

    if True:
        out = []
        with torch.no_grad():
            for k in range(0, len(pics), 8):
                batch = proc(images=pics[k : k + 8], return_tensors="pt")
                feats = model.get_image_features(**batch)
                feats = getattr(feats, "pooler_output", feats)
                out.append(feats.float().numpy())
        emb = np.concatenate(out)
        np.savez(f"out/emb_{tag}.npz", keys=keys, emb=emb)
        print(tag, emb.shape)


if __name__ == "__main__":
    if sys.argv[1:] == ["dump"]:
        dump_pics()
    else:
        main([a for a in sys.argv[1:] if not a.startswith("--")] or list(MODELS))
