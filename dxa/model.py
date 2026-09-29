"""Модель контроля качества: решение по каждому критерию ТЗ.

| Область | Критерий (метка организатора) | Как решается |
|---|---|---|
| позвоночник | Не выравнена ось позвоночника | правило ТЗ: наклон > 5° (линия через крайние позвонки) |
| позвоночник | Некорректная укладка | правило ТЗ: не видны гребни подвздошных костей |
| позвоночник | Присутствуют посторонние предметы | сила тонких ярких линий вне столба, порог по обучению |
| бедро | Некорректная область интереса | правило ТЗ: < 3 см ниже малого вертела до края кадра |
| бедро | Некорректная укладка (ротация) | среднее трёх оценок: малый вертел (U-образно), офсет головки, SigLIP |

Обучаемые параметры (fit): медиана выступа малого вертела у нормы, эталонные распределения
для перевода оценок в процентили, логрегрессия на SigLIP, пороги для двух оценок.
Пороги подбираются по F1 на out-of-fold предсказаниях ОБУЧАЮЩЕЙ части.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from . import cv
from .data import LABEL_AXIS, LABEL_FOREIGN, LABEL_POSITIONING, LABEL_ROI

AXIS_LIMIT_DEG = 5.0          # ТЗ
ILIAC_MIN_FRAC = 0.005        # «гребни не видны»: кости по бокам у нижнего края практически нет
ROI_BOTTOM_CM = 3.0           # ТЗ

CRITERIA = {
    "spine": [LABEL_POSITIONING, LABEL_AXIS, LABEL_FOREIGN],
    "femur": [LABEL_POSITIONING, LABEL_ROI],
}


def _nan(x, fill):
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, fill)


def _cdf(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    ref = np.sort(ref)
    return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


def _best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Порог с максимальным F1; кандидаты — квантили оценки (без подгонки под отдельные точки)."""
    cands = np.unique(np.quantile(score, np.linspace(0.5, 0.98, 49)))
    f1 = [f1_score(y, score >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1))])


class QualityModel:
    def fit(self, feats: list[dict], emb: np.ndarray, regions: list[str], labels: list[dict], groups: np.ndarray):
        regions = np.array(regions)
        # --- позвоночник: посторонние предметы — один порог по обучению
        sp = np.nonzero(regions == "spine")[0]
        y_fo = np.array([labels[i].get(LABEL_FOREIGN, 0) for i in sp])
        ridge = _nan([feats[i].get("fo_ridge_p99", np.nan) for i in sp], 0.0)
        self.fo_threshold = _best_threshold(y_fo, ridge)

        # --- бедро: укладка/ротация — три оценки
        fe = np.nonzero(regions == "femur")[0]
        y = np.array([labels[i].get(LABEL_POSITIONING, 0) for i in fe])
        area = np.array([feats[i].get("rot_lt_area_mm2", np.nan) for i in fe], float)
        off = np.array([feats[i].get("rot_offset_mm", np.nan) for i in fe], float)
        self.area_fill, self.off_fill = float(np.nanmedian(area)), float(np.nanmedian(off))
        area, off = _nan(area, self.area_fill), _nan(off, self.off_fill)
        self.area_norm = float(np.median(area[y == 0]))
        E = emb[fe]
        # эталон для SigLIP — out-of-fold вероятности на обучающей части (как будут выглядеть на новых данных)
        g = groups[fe]
        sig_oof = np.zeros(len(fe))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(E, y, g):
            m = cv.linear(16).fit(E[tr], y[tr])
            sig_oof[te] = m.predict_proba(E[te])[:, 1]
        self.sig_model = cv.linear(16).fit(E, y)
        self.ref_u = np.abs(area - self.area_norm)
        self.ref_off = -off
        self.ref_sig = sig_oof
        score = self._femur_score(area, off, sig_oof)
        self.femur_threshold = _best_threshold(y, score)

        # --- калибровка: оценка каждого критерия -> вероятность нарушения (для quality_prob).
        # Для ротации бедра берётся оценка с out-of-fold SigLIP — как она будет выглядеть на новых данных.
        self.calib = {}
        spine_out = [self.predict(feats[i], None, "spine") for i in sp]
        for L in (LABEL_AXIS, LABEL_POSITIONING, LABEL_FOREIGN):
            idx = [k for k, i in enumerate(sp) if L in labels[i]]
            self._fit_calib(("spine", L), [spine_out[k][L][1] for k in idx], [labels[sp[k]][L] for k in idx])
        roi_idx = [i for i in fe if LABEL_ROI in labels[i]]
        self._fit_calib(("femur", LABEL_ROI), [self.predict(feats[i], None, "femur")[LABEL_ROI][1] for i in roi_idx],
                        [labels[i][LABEL_ROI] for i in roi_idx])
        self._fit_calib(("femur", LABEL_POSITIONING), score, y)
        return self

    def _fit_calib(self, key, score, y):
        s = np.asarray(score, float)
        y = np.asarray(y, int)
        mu, sd = float(s.mean()), float(s.std() + 1e-9)
        lr = LogisticRegression(C=1.0).fit(((s - mu) / sd)[:, None], y) if 0 < y.sum() < len(y) else None
        self.calib[key] = (mu, sd, lr, float(y.mean()))

    def probabilities(self, out: dict, region: str) -> tuple[dict, float]:
        """Вероятность нарушения по каждому критерию и итоговая quality_prob = P(хотя бы одно нарушение)."""
        probs = {}
        for L, (_, score, _) in out.items():
            mu, sd, lr, base = self.calib.get((region, L), (0.0, 1.0, None, 0.0))
            probs[L] = float(lr.predict_proba([[(score - mu) / sd]])[0, 1]) if lr is not None else base
        return probs, float(1 - np.prod([1 - p for p in probs.values()]))

    def _femur_score(self, area, off, sig):
        u = np.abs(area - self.area_norm)
        return (_cdf(self.ref_u, u) + _cdf(self.ref_off, -off) + _cdf(self.ref_sig, sig)) / 3

    def predict(self, feat: dict, emb: np.ndarray | None, region: str) -> dict:
        """{метка: (флаг 0/1, оценка, объяснение)} по критериям области."""
        out = {}
        if region == "spine":
            ang = feat.get("axis_chord15_deg", np.nan)
            out[LABEL_AXIS] = (int(np.isfinite(ang) and ang > AXIS_LIMIT_DEG), float(_nan([ang], 0)[0]),
                               f"наклон оси {ang:.1f}° (допуск ТЗ {AXIS_LIMIT_DEG:g}°)")
            fr = feat.get("sp_iliac_frac", np.nan)
            out[LABEL_POSITIONING] = (int(np.isfinite(fr) and fr < ILIAC_MIN_FRAC), float(-_nan([fr], 0)[0]),
                                      "гребни подвздошных костей не видны у нижнего края" if np.isfinite(fr) and fr < ILIAC_MIN_FRAC
                                      else "гребни подвздошных костей видны")
            r = float(_nan([feat.get("fo_ridge_p99", np.nan)], 0)[0])
            out[LABEL_FOREIGN] = (int(r >= self.fo_threshold), r,
                                  f"тонкие яркие линии вне позвоночника: {r:.3f} (порог {self.fo_threshold:.3f})")
        else:
            b = feat.get("roi_margin_bottom_cm", np.nan)
            out[LABEL_ROI] = (int(np.isfinite(b) and b < ROI_BOTTOM_CM), float(-_nan([b], 0)[0]),
                              f"ниже малого вертела {b:.1f} см (норма ТЗ ≥ {ROI_BOTTOM_CM:g}); сверху "
                              f"{feat.get('roi_margin_top_cm', np.nan):.1f}, медиально {feat.get('roi_margin_medial_cm', np.nan):.1f} см")
            area = float(_nan([feat.get("rot_lt_area_mm2", np.nan)], self.area_fill)[0])
            off = float(_nan([feat.get("rot_offset_mm", np.nan)], self.off_fill)[0])
            sig = float(self.sig_model.predict_proba(emb[None])[0, 1]) if emb is not None else float(np.median(self.ref_sig))
            s = float(self._femur_score(np.array([area]), np.array([off]), np.array([sig]))[0])
            kind = "переротация (контур гладкий)" if area < self.area_norm else "недоротация (малый вертел крупный)"
            out[LABEL_POSITIONING] = (int(s >= self.femur_threshold), s,
                                      f"малый вертел {area:.0f} мм² (норма ~{self.area_norm:.0f}), офсет головки {off:.0f} мм; "
                                      f"оценка {s:.2f} (порог {self.femur_threshold:.2f})"
                                      + (f"; вероятно {kind}" if s >= self.femur_threshold else ""))
        return out
