"""Выгрузить только визуальную часть SigLIP в weights/siglip-vision (для контейнера, без интернета).

    python scripts/export_siglip.py [--check]

Источник — кэш Hugging Face или загрузка google/siglip-so400m-patch14-384 (один раз, при сборке).
--check сверяет признаки с полной моделью (должны совпасть).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

SRC = "google/siglip-so400m-patch14-384"
DST = Path(__file__).resolve().parents[1] / "weights" / "siglip-vision"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    vision = SiglipVisionModel.from_pretrained(SRC)
    proc = AutoImageProcessor.from_pretrained(SRC)
    DST.mkdir(parents=True, exist_ok=True)
    vision.save_pretrained(DST)
    proc.save_pretrained(DST)
    print("сохранено в", DST)
    if args.check:
        img = Image.fromarray((np.random.default_rng(0).random((300, 300)) * 255).astype(np.uint8)).convert("RGB")
        x = proc(images=[img], return_tensors="pt")
        with torch.no_grad():
            full = AutoModel.from_pretrained(SRC).get_image_features(**x)
            full = getattr(full, "pooler_output", full)
            part = SiglipVisionModel.from_pretrained(DST)(**x).pooler_output
        print("макс. расхождение признаков:", float((full - part).abs().max()))


if __name__ == "__main__":
    main()
