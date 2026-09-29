"""Честная оценка моделей на малых данных.

* разбиение StratifiedGroupKFold по исследованию (снимки одного пациента не делятся);
* повторы с разными сидами, предсказания out-of-fold усредняются;
* регуляризация подбирается внутренней кросс-валидацией только на обучающей части;
* доверительный интервал — бутстрэп по исследованиям.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def linear(pca: int | None = None):
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=pca, random_state=0))
    steps.append(LogisticRegressionCV(Cs=np.logspace(-4, 1, 11), cv=4, scoring="roc_auc",
                                      class_weight="balanced", max_iter=5000))
    return make_pipeline(*steps)


def oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, make=linear, folds: int = 5, seeds: int = 5) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
    pred = np.zeros(len(y))
    for seed in range(seeds):
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups):
            model = make()
            model.fit(X[tr], y[tr])
            p = model.predict_proba(X[te])[:, 1]
            # ранги внутри фолда, чтобы фолды с разной калибровкой не смешивались
            pred[te] += p
    return pred / seeds


def auc_ci(y: np.ndarray, score: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.nonzero(groups == g)[0] for g in uniq}
    values = []
    for _ in range(n):
        idx = np.concatenate([index[g] for g in rng.choice(uniq, len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            values.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(values, (2.5, 97.5))
    return float(roc_auc_score(y, score)), float(lo), float(hi)
