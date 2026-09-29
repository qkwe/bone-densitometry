"""Перестановочный тест для линейной пробы SigLIP (укладка бедра).

Метки перемешиваются N раз, пайплайн (CV по исследованию, PCA16 + логрегрессия) прогоняется
заново. Доля перестановок с AUC >= настоящего — p-значение; распределение показывает шум.
"""

import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
import eval_rot_models as E
from dxa import cv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
X = E.emb("siglip_full")
real = roc_auc_score(E.y, cv.oof(X, E.y, E.groups, make=lambda: cv.linear(16), seeds=1))
rng = np.random.default_rng(0)
null = []
for k in range(N):
    yp = rng.permutation(E.y)
    null.append(roc_auc_score(yp, cv.oof(X, yp, E.groups, make=lambda: cv.linear(16), seeds=1)))
    if (k + 1) % 20 == 0:
        print(f"{k + 1} перестановок: среднее {np.mean(null):.3f}, 95-й перцентиль {np.percentile(null, 95):.3f}", flush=True)
null = np.array(null)
print(f"настоящие метки: AUC {real:.3f}")
print(f"перемешанные: среднее {null.mean():.3f}, std {null.std():.3f}, 95% {np.percentile(null, 95):.3f}, max {null.max():.3f}")
print(f"p-значение: {(1 + (null >= real).sum()) / (1 + N):.3f}")
np.save("out/perm_null.npy", null)
