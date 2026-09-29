"""Честная оценка итоговой системы тем же кодом, что в программе (dxa/model.py).

Внешняя CV — StratifiedGroupKFold по исследованию, 5 фолдов × 5 повторов; модель (пороги,
логрегрессия, эталоны) обучается только на обучающей части фолда. Метрики — по каждому
критерию, по области («качественное / есть нарушение») и в целом, с 95% ДИ бутстрэпом
по исследованиям. Результат: out/metrics_final.json и таблица в консоли.
"""

import json
import pickle
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
from dxa import features
from dxa.data import LABELS
from dxa.model import CRITERIA, QualityModel

SEEDS, FOLDS = 5, 5
images = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.quality is not None]
keys = [f"{i.study_uid}:{i.image_uid}" for i in images]


def load_features():
    try:
        cached = pickle.load(open("out/features_all.pkl", "rb"))
        if set(cached) >= set(keys):
            return cached
    except FileNotFoundError:
        pass
    out = {k: features.geometry(i.array, i.spacing, i.region, i.side) for k, i in zip(keys, images)}
    pickle.dump(out, open("out/features_all.pkl", "wb"))
    return out


F = load_features()
feats = [F[k] for k in keys]
d = np.load("out/emb_siglip_full.npz", allow_pickle=True)
E_lookup = dict(zip(d["keys"], d["emb"]))
dim = len(next(iter(E_lookup.values())))
emb = np.array([E_lookup.get(k, np.zeros(dim)) for k in keys])
regions = [i.region for i in images]
labels = [i.labels for i in images]
groups = np.array([i.study_uid for i in images])
y_any = np.array([int(i.quality) for i in images])

flags = {L: np.zeros(len(images)) for L in LABELS}
scores = {L: np.zeros(len(images)) for L in LABELS}
for seed in range(SEEDS):
    for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed).split(emb, y_any, groups):
        m = QualityModel().fit([feats[i] for i in tr], emb[tr], [regions[i] for i in tr], [labels[i] for i in tr], groups[tr])
        for i in te:
            for L, (flag, score, _) in m.predict(feats[i], emb[i] if regions[i] == "femur" else None, regions[i]).items():
                flags[L][i] += flag / SEEDS
                scores[L][i] += score / SEEDS


def boot(y, pred, score, g, n=1000):
    rng = np.random.default_rng(0)
    u = np.unique(g)
    ix = {s: np.nonzero(g == s)[0] for s in u}
    vals = {"auc": [], "f1": [], "sens": [], "spec": [], "bacc": []}
    for _ in range(n):
        i = np.concatenate([ix[s] for s in rng.choice(u, len(u))])
        if 0 < y[i].sum() < len(i):
            for k, v in metrics(y[i], pred[i], score[i]).items():
                vals[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def metrics(y, pred, score):
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(auc=float(roc_auc_score(y, score)), f1=float(f1_score(y, pred, zero_division=0)),
                sens=sens, spec=spec, bacc=(sens + spec) / 2)


def rank01(v):
    return (np.argsort(np.argsort(v)) + 0.5) / len(v)


report = {}
rows = []
for region, crits in CRITERIA.items():
    idx = np.array([i for i, r in enumerate(regions) if r == region])
    any_pred = np.zeros(len(idx))
    any_score = np.zeros(len(idx))
    for L in crits:
        sub = np.array([i for i in idx if L in labels[i]])
        y = np.array([labels[i][L] for i in sub])
        pred = (flags[L][sub] >= 0.5).astype(int)
        m = metrics(y, pred, scores[L][sub])
        m["ci"] = boot(y, pred, scores[L][sub], groups[sub])
        m.update(n=len(y), positives=int(y.sum()))
        report[f"{region}: {L}"] = m
        pos = np.searchsorted(idx, sub)
        any_pred[pos] = np.maximum(any_pred[pos], pred)
        any_score[pos] = np.maximum(any_score[pos], rank01(scores[L][sub]))
    y = y_any[idx]
    m = metrics(y, any_pred.astype(int), any_score)
    m["ci"] = boot(y, any_pred.astype(int), any_score, groups[idx])
    m.update(n=len(y), positives=int(y.sum()))
    report[f"{region}: качество (есть нарушение)"] = m

# в целом по всем снимкам и macro-F1 по типам нарушений
all_pred = np.zeros(len(images), int)
all_score = np.zeros(len(images))
for region, crits in CRITERIA.items():
    for L in crits:
        sub = np.array([i for i, r in enumerate(regions) if r == region and L in labels[i]])
        all_pred[sub] = np.maximum(all_pred[sub], (flags[L][sub] >= 0.5).astype(int))
        all_score[sub] = np.maximum(all_score[sub], rank01(scores[L][sub]))
m = metrics(y_any, all_pred, all_score)
m["ci"] = boot(y_any, all_pred, all_score, groups)
m.update(n=len(y_any), positives=int(y_any.sum()))
report["все снимки: качество (есть нарушение)"] = m
report["macro-F1 по типам нарушений"] = float(np.mean([v["f1"] for k, v in report.items()
                                                      if isinstance(v, dict) and "качество" not in k]))

print(f"{'критерий':62s} {'n':>4s} {'поз':>4s}  AUC [95% ДИ]         F1 [95% ДИ]         чувств  специф  сбал.т")
for k, v in report.items():
    if not isinstance(v, dict):
        print(f"{k}: {v:.3f}")
        continue
    c = v["ci"]
    print(f"{k:62s} {v['n']:4d} {v['positives']:4d}  {v['auc']:.3f} [{c['auc'][0]:.2f}; {c['auc'][1]:.2f}]  "
          f"{v['f1']:.3f} [{c['f1'][0]:.2f}; {c['f1'][1]:.2f}]  {v['sens']:.2f}    {v['spec']:.2f}    {v['bacc']:.2f}")
json.dump(report, open("out/metrics_final.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pickle.dump(dict(flags=flags, scores=scores, keys=keys), open("out/oof_final.pkl", "wb"))
