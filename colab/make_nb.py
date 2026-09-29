"""Собирает ноутбук Colab для дообучения SigLIP (укладка/ротация бедра)."""

import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("""# Дообучение SigLIP: укладка/ротация бедра (DXA)

Защита от переобучения:
* внешняя CV — StratifiedGroupKFold по исследованию (как локально), 2 повтора;
* **число эпох фиксировано заранее** (EPOCHS), лучшая эпоха по тестовому фолду НЕ выбирается;
* дообучаются только последние блоки + голова, маленький lr, weight decay, аугментации;
* базовая линия (замороженный SigLIP + логрегрессия) — на тех же фолдах;
* **контроль с перемешанными метками** — обязан дать AUC ≈ 0.5;
* разрыв AUC train/test по фолдам — индикатор переобучения.""")

code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
import transformers, torch; print(transformers.__version__, torch.__version__, torch.cuda.is_available())""")

code("""from google.colab import files
up = files.upload()   # femur.npz
print(list(up))""")

code(r'''import numpy as np, torch, torch.nn as nn, time, copy
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torchvision.transforms.v2 as T
from transformers import SiglipVisionModel

d = np.load("femur.npz", allow_pickle=True)
pics, Y, groups, offset = d["pics"], d["y"].astype(int), d["groups"], d["offset"]
print(pics.shape, Y.sum(), len(np.unique(groups)))

MODEL = "google/siglip-so400m-patch14-384"
DEV = "cuda"
EPOCHS, UNFREEZE, LR_BB, LR_HEAD, WD, BS = 10, 2, 1e-5, 1e-3, 0.05, 16
SEEDS, FOLDS = [0, 1], 5

base = SiglipVisionModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
MEAN, STD = 0.5, 0.5  # нормализация SigLIP


def to_tensor(idx):
    x = torch.from_numpy(pics[idx]).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - MEAN) / STD


aug = T.Compose([T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                 T.ColorJitter(brightness=0.2, contrast=0.2)])


@torch.no_grad()
def frozen_features():
    m = base.to(DEV).half()
    out = []
    for k in range(0, len(pics), 16):
        idx = np.arange(k, min(k + 16, len(pics)))
        out.append(m(pixel_values=to_tensor(idx).to(DEV).half()).pooler_output.float().cpu())
    base.float().cpu()
    return torch.cat(out).numpy()


FEAT = frozen_features()
print("frozen", FEAT.shape)''')

code(r'''class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = copy.deepcopy(base)
        for p in self.bb.parameters():
            p.requires_grad = False
        vm = self.bb.vision_model
        for mod in list(vm.encoder.layers[-UNFREEZE:]) + [vm.post_layernorm, vm.head]:
            for p in mod.parameters():
                p.requires_grad = True
        self.cls = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.bb.config.hidden_size, 1))

    def forward(self, x):
        return self.cls(self.bb(pixel_values=x).pooler_output).squeeze(1)


def predict(net, idx):
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for k in range(0, len(idx), 32):
            out.append(torch.sigmoid(net(to_tensor(idx[k:k + 32]).to(DEV)).float()).cpu())
    return torch.cat(out).numpy()


def finetune(tr, te, labels, seed):
    """labels — метки для обучения и для диагностики (в контроле — перемешанные)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net().to(DEV)
    bb_params = [p for p in net.bb.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": LR_BB},
                             {"params": net.cls.parameters(), "lr": LR_HEAD}], weight_decay=WD)
    steps = EPOCHS * int(np.ceil(len(tr) / BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_BB, LR_HEAD], total_steps=steps, pct_start=0.2)
    pos_w = torch.tensor((labels[tr] == 0).sum() / max((labels[tr] == 1).sum(), 1), device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = torch.amp.GradScaler("cuda")
    curve = []
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(tr)
        for k in range(0, len(perm), BS):
            b = perm[k:k + BS]
            x = torch.stack([aug(t) for t in to_tensor(b)]).to(DEV)
            t = torch.tensor(labels[b], dtype=torch.float32, device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(net(x), t)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        # только диагностика: по этой кривой эпоху НЕ выбираем
        curve.append((roc_auc_score(labels[tr], predict(net, tr)), roc_auc_score(labels[te], predict(net, te))))
    p = predict(net, te)
    del net
    torch.cuda.empty_cache()
    return p, curve


def linear_probe(tr, te, labels):
    m = make_pipeline(StandardScaler(), PCA(16, random_state=0),
                      LogisticRegressionCV(Cs=np.logspace(-4, 1, 11), cv=4, scoring="roc_auc",
                                           class_weight="balanced", max_iter=5000))
    m.fit(FEAT[tr], labels[tr])
    return m.predict_proba(FEAT[te])[:, 1]


def auc_ci(labels, score, n=1000):
    rng = np.random.default_rng(0)
    u = np.unique(groups)
    ix = {g: np.nonzero(groups == g)[0] for g in u}
    v = []
    for _ in range(n):
        i = np.concatenate([ix[g] for g in rng.choice(u, len(u))])
        if 0 < labels[i].sum() < len(i):
            v.append(roc_auc_score(labels[i], score[i]))
    return (roc_auc_score(labels, score), *np.percentile(v, [2.5, 97.5]))


def run(labels, seeds, with_ft=True, tag=""):
    oof_ft, oof_lp, curves = np.zeros(len(labels)), np.zeros(len(labels)), []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(cv.split(pics, labels, groups)):
            t0 = time.time()
            oof_lp[te] += linear_probe(tr, te, labels) / len(seeds)
            if with_ft:
                p, c = finetune(tr, te, labels, seed)
                oof_ft[te] += p / len(seeds)
                curves.append(c)
                print(f"{tag}seed {seed} fold {f}: train AUC {c[-1][0]:.2f}  test AUC {c[-1][1]:.2f}  ({time.time() - t0:.0f} с)", flush=True)
    return oof_lp, oof_ft, np.array(curves)''')

code(r'''oof_lp, oof_ft, curves = run(Y, SEEDS)
print("линейная проба (замороженный SigLIP): AUC %.3f [%.2f; %.2f]" % auc_ci(Y, oof_lp))
print("дообучение SigLIP (фикс. %d эпох):    AUC %.3f [%.2f; %.2f]" % ((EPOCHS,) + auc_ci(Y, oof_ft)))
print("средняя кривая по эпохам (train / test) — только диагностика:")
for e, (a, b) in enumerate(curves.mean(0)):
    print(f"  эпоха {e + 1:2d}: {a:.3f} / {b:.3f}")
np.savez("oof_siglip_ft.npz", keys=d["keys"], oof_ft=oof_ft, oof_lp=oof_lp, curves=curves)''')

code(r'''# Контроль: метки перемешаны (связь снимок–метка разрушена). Test AUC обязан быть ≈ 0.5.
Y_SHUF = np.random.default_rng(123).permutation(Y)
sh_lp, sh_ft, sh_curves = run(Y_SHUF, [0], tag="[перемешано] ")
print("перемешанные метки: линейная проба AUC %.3f, дообучение AUC %.3f (оба должны быть ≈0.5)"
      % (auc_ci(Y_SHUF, sh_lp)[0], auc_ci(Y_SHUF, sh_ft)[0]))
print("train AUC на перемешанных метках: %.2f — насколько модель способна просто запомнить шум" % sh_curves[:, -1, 0].mean())''')

code(r'''# Объединение с геометрией (офсет головки): среднее рангов, вес задан заранее (0.5)
from scipy.stats import rankdata
off_score = -offset  # меньше офсет — хуже
for name, s in [("офсет", off_score), ("линейная проба", oof_lp), ("дообучение", oof_ft),
                ("дообучение + офсет", rankdata(oof_ft) + rankdata(off_score)),
                ("линейная проба + офсет", rankdata(oof_lp) + rankdata(off_score))]:
    print(f"{name:25s} AUC %.3f [%.2f; %.2f]" % auc_ci(Y, s))
files.download("oof_siglip_ft.npz")''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("siglip_finetune.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")
