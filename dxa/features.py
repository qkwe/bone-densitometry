"""Все признаки одного снимка: измерения по критериям ТЗ (без обучения)."""

from __future__ import annotations

import numpy as np

from . import axis, roi, rotation, spine


def geometry(array: np.ndarray, spacing, region: str, side: str | None) -> dict:
    """Измерения без обучения; ошибки отдельных измерений не роняют обработку."""
    if region == "spine":
        feats = axis.measure(array, spacing)
        feats.update(spine.measure(array, spacing))
        return feats
    feats = roi.measure(array, spacing, side)
    feats.update(rotation.measure(array, spacing, side))
    return feats
