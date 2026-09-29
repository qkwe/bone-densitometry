"""Маски бедренной кости SAM/MedSAM по подсказкам (Python с torch).

    python310 sam_run.py sam      -> out/sam_masks_sam.npz
    python310 sam_run.py medsam   -> out/sam_masks_medsam.npz
"""

import os
import sys

import numpy as np
import torch
from PIL import Image as P
from transformers import SamModel, SamProcessor

MODELS = {"sam": "facebook/sam-vit-base", "medsam": "flaviagiammarino/medsam-vit-base"}
name = sys.argv[1] if len(sys.argv) > 1 else "sam"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

d = np.load("out/sam_input.npz", allow_pickle=True)
proc = SamProcessor.from_pretrained(MODELS[name])
model = SamModel.from_pretrained(MODELS[name]).eval()

masks, scores = [], []
n = len(d["pics"]) if limit is None else limit
for k in range(n):
    img = P.fromarray(d["pics"][k]).convert("RGB")
    box = [[list(map(float, d["boxes"][k]))]]
    if name == "medsam":  # MedSAM обучена на рамках, точки не использует
        inputs = proc(img, input_boxes=box, return_tensors="pt")
    else:
        pts = [[[list(map(float, p)) for p in d["points"][k]]]]
        lab = [[list(map(int, d["labels"][k]))]]
        # рамка захватывает таз, поэтому по умолчанию только точки; SAM_BOX=1 — с рамкой
        if os.environ.get("SAM_BOX") == "1":
            inputs = proc(img, input_points=pts, input_labels=lab, input_boxes=box, return_tensors="pt")
        else:
            inputs = proc(img, input_points=pts, input_labels=lab, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    m = proc.image_processor.post_process_masks(out.pred_masks, inputs["original_sizes"],
                                                inputs["reshaped_input_sizes"])[0][0]
    iou = out.iou_scores[0, 0].numpy()
    best = int(np.argmax(iou))
    masks.append(m[best].numpy().astype(bool))
    scores.append(float(iou[best]))
    if k % 25 == 0:
        print(name, k, f"iou {iou.round(2)}", flush=True)

obj = np.empty(len(masks), dtype=object)
obj[:] = masks
tag = name + ("_box" if os.environ.get("SAM_BOX") == "1" else "")
np.savez(f"out/sam_masks_{tag}.npz", masks=obj, scores=np.array(scores), keys=d["keys"][:n])
print("готово", n)
