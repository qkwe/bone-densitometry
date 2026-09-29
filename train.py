"""Обучение итоговой модели на наборе организатора (НД_для_обучения).

    python train.py --data C:\\dxa --out models/quality_model.joblib

Шаги: чтение DICOM и меток → измерения по критериям ТЗ → признаки SigLIP (замороженный) →
классификатор области (позвоночник/бедро) и модель качества (пороги по обучающим данным).
Для честных метрик см. evaluate_final.py (тот же код модели, вложенная CV по исследованиям).
"""

import argparse
import datetime as dt
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from dxa import features
from dxa.data import load_dataset
from dxa.model import QualityModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="распакованный НД_для_обучения (с xlsx разметкой)")
    ap.add_argument("--out", default="models/quality_model.joblib")
    ap.add_argument("--siglip", default=features.SIGLIP)
    ap.add_argument("--cache", default="out/train_cache.pkl", help="кэш признаков (ускоряет повторное обучение)")
    ap.add_argument("--refresh-geo", action="store_true", help="пересчитать измерения, SigLIP взять из кэша")
    args = ap.parse_args()

    images = load_dataset(args.data)
    print(f"снимков {len(images)}", flush=True)
    cache = {}
    if Path(args.cache).exists():
        cache = pickle.load(open(args.cache, "rb"))
    emb_model = None
    geo, plain, femur_emb = [], [], []
    for n, im in enumerate(images):
        key = f"{im.study_uid}:{im.image_uid}"
        if key not in cache:
            if emb_model is None:
                emb_model = features.Embedder(args.siglip)
            e_plain = emb_model([features.canonical_femur(im.array, None)])[0]
            e_fem = None
            if im.region == "femur":
                e_fem = e_plain if im.side == "r" else emb_model([features.canonical_femur(im.array, im.side)])[0]
            cache[key] = dict(geo=features.geometry(im.array, im.spacing, im.region, im.side), plain=e_plain, femur=e_fem)
            if n % 20 == 0:
                print(f"  {n}/{len(images)}", flush=True)
                Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(cache, open(args.cache, "wb"))
        c = cache[key]
        if args.refresh_geo:
            c["geo"] = features.geometry(im.array, im.spacing, im.region, im.side)
        geo.append(c["geo"])
        plain.append(c["plain"])
        femur_emb.append(c["femur"] if c["femur"] is not None else np.zeros_like(c["plain"]))
    pickle.dump(cache, open(args.cache, "wb"))

    plain, femur_emb = np.array(plain), np.array(femur_emb)
    regions = [im.region for im in images]
    region_clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    region_clf.fit(plain, np.array([r == "spine" for r in regions], int))

    lab = [i for i, im in enumerate(images) if im.quality is not None]
    model = QualityModel().fit([geo[i] for i in lab], femur_emb[lab], [regions[i] for i in lab],
                               [images[i].labels for i in lab], np.array([images[i].study_uid for i in lab]))
    meta = dict(trained=dt.datetime.now().isoformat(timespec="seconds"), n_images=len(images), n_labeled=len(lab),
                siglip=args.siglip, fo_threshold=model.fo_threshold, femur_threshold=model.femur_threshold,
                area_norm=model.area_norm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(quality=model, region=region_clf, meta=meta), args.out)
    json.dump(meta, open(Path(args.out).with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("сохранено", args.out, meta)


if __name__ == "__main__":
    main()
