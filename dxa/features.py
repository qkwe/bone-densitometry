"""Все признаки одного снимка: измерения по критериям ТЗ + признаки SigLIP для бедра."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as P

from . import axis, roi, rotation, spine

_LOCAL = Path(__file__).resolve().parents[1] / "weights" / "siglip-vision"
# локальная визуальная часть SigLIP (контейнер работает без сети), иначе — кэш Hugging Face
SIGLIP = str(_LOCAL) if _LOCAL.exists() else "google/siglip-so400m-patch14-384"


def canonical_femur(array: np.ndarray, side: str | None) -> P.Image:
    """Бедро в единой ориентации (медиально справа), дополнено до квадрата."""
    a = array.astype(np.float32)
    lo, hi = np.percentile(a, (0.5, 99.5))
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if side == "l":  # у левого бедра медиальная сторона на снимке слева
        a = a[:, ::-1]
    h, w = a.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.float32)
    sq[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = a
    return P.fromarray((sq * 255).astype(np.uint8)).convert("RGB")


class Embedder:
    """SigLIP (замороженный) — вектор признаков снимка. Грузится один раз, работает на GPU при наличии."""

    def __init__(self, model_dir: str = SIGLIP):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model_dir)
        # только визуальная часть: признаки совпадают с SiglipModel.get_image_features
        self.model = SiglipVisionModel.from_pretrained(model_dir).to(self.device).eval()

    def __call__(self, pics: list[P.Image]) -> np.ndarray:
        with self.torch.no_grad():
            batch = self.proc(images=pics, return_tensors="pt").to(self.device)
            out = self.model(**batch).pooler_output
        return out.float().cpu().numpy()


def geometry(array: np.ndarray, spacing, region: str, side: str | None) -> dict:
    """Измерения без обучения; ошибки отдельных измерений не роняют обработку."""
    if region == "spine":
        feats = axis.measure(array, spacing)
        feats.update(spine.measure(array, spacing))
        return feats
    feats = roi.measure(array, spacing, side)
    feats.update(rotation.measure(array, spacing, side))
    return feats
