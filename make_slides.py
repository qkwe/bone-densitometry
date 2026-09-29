"""Картинки для презентации: ось позвоночника, согласие и расхождения с разметкой."""

import pickle
import sys

import numpy as np
from PIL import Image as P, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, ".")
from dxa import axis
from dxa.data import LABEL_AXIS

SCALE = 3
FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"
GREEN, RED, YELLOW, CYAN, WHITE, GREY = (70, 200, 90), (230, 70, 60), (255, 220, 0), (0, 200, 255), (255, 255, 255), (170, 170, 170)

images = [i for i in pickle.load(open("out/cache.pkl", "rb")) if i.region == "spine"]


def chord(array, spacing):
    ys, xs = axis.centerline(array, spacing)
    n = max(int(len(ys) * 0.15), 3)
    top = (float(np.median(xs[:n])), float(np.median(ys[:n])))
    bot = (float(np.median(xs[-n:])), float(np.median(ys[-n:])))
    return top, bot, axis.measure(array, spacing)["axis_chord15_deg"]


def panel(array, spacing, title, lines, color):
    top, bot, angle = chord(array, spacing)
    a = axis.normalize(array)
    img = P.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    img = img.resize((img.width * SCALE, img.height * SCALE), P.BICUBIC)
    d = ImageDraw.Draw(img)
    t = (top[0] * SCALE, top[1] * SCALE)
    b = (bot[0] * SCALE, bot[1] * SCALE)
    # продлеваем линию на всю высоту столба, как на рис. 2 ТЗ
    k = (t[0] - b[0]) / (t[1] - b[1])
    y0 = 10
    d.line([b, (b[0] + k * (y0 - b[1]), y0)], fill=YELLOW, width=4)
    d.line([b, (b[0], y0)], fill=CYAN, width=3)
    d.ellipse([b[0] - 7, b[1] - 7, b[0] + 7, b[1] + 7], fill=RED)
    d.ellipse([t[0] - 6, t[1] - 6, t[0] + 6, t[1] + 6], fill=RED)
    f = ImageFont.truetype(BOLD, 44)
    d.rectangle([b[0] + 14, b[1] - 60, b[0] + 190, b[1] - 6], fill=(0, 0, 0))
    d.text((b[0] + 22, b[1] - 58), f"{angle:.1f}°", font=f, fill=YELLOW)

    head = 64 + 38 * len(lines) + 16
    card = P.new("RGB", (img.width, img.height + head), (25, 25, 28))
    card.paste(img, (0, head))
    dc = ImageDraw.Draw(card)
    dc.rectangle([0, 0, card.width, 8], fill=color)
    dc.text((16, 18), title, font=ImageFont.truetype(BOLD, 34), fill=WHITE)
    for i, (text, c) in enumerate(lines):
        dc.text((16, 64 + i * 38), text, font=ImageFont.truetype(FONT, 30), fill=c)
    return card, angle


def row(cards, caption, path):
    gap, head = 24, 90
    h = max(c.height for c in cards)
    W = P.new("RGB", (sum(c.width for c in cards) + gap * (len(cards) + 1), h + head + gap), (15, 15, 18))
    d = ImageDraw.Draw(W)
    d.text((gap, 24), caption, font=ImageFont.truetype(BOLD, 44), fill=WHITE)
    x = gap
    for c in cards:
        W.paste(c, (x, head))
        x += c.width + gap
    W.save(path)
    print(path, W.size)


def doctor_card(k, verdict_color):
    im = images[k]
    bad = im.labels.get(LABEL_AXIS) == 1
    angle = axis.measure(im.array, im.spacing)["axis_chord15_deg"]
    ours = angle > axis.AXIS_LIMIT_DEG
    agree = ours == bad
    lines = [
        (f"Врач: {'ось НЕ выровнена' if bad else 'ось в норме'}", RED if bad else GREEN),
        (f"Правило ТЗ (>5°): {'нарушение' if ours else 'норма'}", RED if ours else GREEN),
        (f"{'совпадает' if agree else 'РАСХОДИТСЯ'}  ·  {im.study_uid[-8:]}", GREY if agree else YELLOW),
    ]
    card, _ = panel(im.array, im.spacing, f"Наклон {angle:.1f}°", lines, verdict_color)
    return card


y = np.array([i.labels.get(LABEL_AXIS, 0) for i in images])
ang = np.array([axis.measure(i.array, i.spacing)["axis_chord15_deg"] for i in images])
tp = [k for k in np.argsort(-ang) if y[k] == 1 and ang[k] > 5]
tn = [k for k in np.argsort(ang) if y[k] == 0]
fn = [k for k in np.argsort(ang) if y[k] == 1 and ang[k] <= 5]
fp = [k for k in np.argsort(-ang) if y[k] == 0 and ang[k] > 5]

# 1. согласие: два нарушения и норма
row([doctor_card(tp[0], GREEN), doctor_card(tp[1], GREEN), doctor_card(tn[0], GREEN)],
    "Где врач и правило ТЗ согласны", "out/slides/1_agree.png")
# 2. врач ставит брак прямому позвоночнику
row([doctor_card(k, YELLOW) for k in fn[:3]],
    "Врач: «ось не выровнена» — но наклон меньше 5°", "out/slides/2_doctor_bad_straight.png")
# 3. врач пропускает явный наклон (первый в списке — с неровной линией центров, берём следующие)
row([doctor_card(k, YELLOW) for k in fp[1:4]],
    "Врач: «норма» — но наклон больше 5°", "out/slides/3_doctor_ok_tilted.png")

# 4. поверка: поворачиваем нормальный снимок на известный угол
k = tn[0]
im = images[k]
base = axis.measure(im.array, im.spacing)["axis_chord15_deg"]
cards = []
for deg in (0, 4, 7, 10):
    arr = ndimage.rotate(im.array.astype(float), deg, reshape=False, order=1, mode="constant", cval=0) if deg else im.array
    measured = axis.measure(arr, im.spacing)["axis_chord15_deg"]
    card, _ = panel(arr, im.spacing, f"Повернули на {deg}°",
                    [(f"Сервис намерил: {measured:.1f}°", WHITE),
                     (f"Правило ТЗ: {'нарушение' if measured > axis.AXIS_LIMIT_DEG else 'норма'}", GREY)],
                    CYAN)
    cards.append(card)
row(cards, "Поверка: 594 поворота, средняя ошибка 0.34° (допуск ТЗ — 3°)", "out/slides/4_validation.png")

# 0. как меряет врач в ТЗ (рис. 2) и как меряем мы
tz = "out/slides/tz_fig2_%s.jpg"
figs = []
for name, text, color in (("a", "ТЗ, рис. 2а: 1.6° — корректно", GREEN), ("b", "ТЗ, рис. 2б: 6.9° — некорректно", RED)):
    f = P.open(tz % name).convert("RGB")
    f = f.resize((int(f.width * 1100 / f.height), 1100))
    c = P.new("RGB", (f.width, f.height + 80), (25, 25, 28))
    c.paste(f, (0, 80))
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, c.width, 8], fill=color)
    d.text((16, 22), text, font=ImageFont.truetype(BOLD, 36), fill=WHITE)
    figs.append(c)
row(figs, "Метод ТЗ: наклон линии между позвонками", "out/slides/0_tz_method.png")

# сводка для текста
print("TP", len(tp), "FN", len(fn), "FP", len(fp), "TN", int(((y == 0) & (ang <= 5)).sum()))
for name, L in (("TP", tp[:2]), ("TN", tn[:1]), ("FN", fn[:3]), ("FP", fp[1:4])):
    for k in L:
        print(name, images[k].study_uid, f"{ang[k]:.1f}", "quality", images[k].quality, images[k].labels)
