"""Эксперимент: локальная VLM (Qwen3-VL-2B) как детектор посторонних предметов на позвоночнике.

Один снимок + промпт под область; оценка — вероятность ответа «Yes» на первом токене.
Сравнивается с детектором тонких линий по AUC. Результат: out/vlm_foreign.npz.
"""

import pickle
import sys

import numpy as np
import torch
from PIL import Image as P
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
PROMPT = ("This is a DXA (bone densitometry) scan of the lumbar spine. Are there any foreign objects on the "
          "image, such as metal bra underwires, clasps, clips, jewelry or other artificial items outside the "
          "bones? Answer with one word: Yes or No.")

sys.path.insert(0, ".")
data = np.load("out/spine_pics_full.npz", allow_pickle=True)
proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
tok = proc.tokenizer
yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(data["pics"])
scores = []
for k in range(limit):
    img = P.fromarray(data["pics"][k]).convert("RGB").resize((448, 448))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    p = torch.softmax(logits[yes + no], 0)
    scores.append(float(p[: len(yes)].sum()))
    if k % 10 == 0:
        print(k, round(scores[-1], 3), flush=True)
np.savez("out/vlm_foreign.npz", keys=data["keys"][:limit], score=np.array(scores))
print("готово", limit)
