"""Позвоночник: укладка и посторонние предметы — измерения без обучения.

ТЗ: «корректная укладка (на нижнем уровне сканирования визуализированы верхние края подвздошных
костей, верхний уровень — половина тела позвонка Th12)»; «отсутствие посторонних предметов,
выраженных артефактов или наложений (металлические артефакты от одежды)».

Укладка: гребни подвздошных костей — яркие «крылья» по бокам от столба у нижнего края кадра;
сверху у Th12 отходят 12-е рёбра. Если кадр обрезан, крыльев нет.
Посторонние предметы: косточки белья и металл — тонкие очень яркие линии/пятна; рёбра тоже
дугообразные, но шире и бледнее, поэтому отбор по ширине и яркости гребня (ridge).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import axis

Spacing = tuple[float, float]


def _column_center(array: np.ndarray, spacing: Spacing) -> tuple[np.ndarray, np.ndarray, float]:
    ys, xs = axis.centerline(array, spacing)
    width = axis.VERTEBRA_WIDTH_MM / spacing[0]
    return ys, xs, width


def _ridges(values: np.ndarray, sigma: float) -> np.ndarray:
    """Яркие тонкие линии: наибольшее по модулю отрицательное собственное число гессиана."""
    g = ndimage.gaussian_filter(values, sigma)
    gyy = ndimage.gaussian_filter(values, sigma, order=(2, 0))
    gxx = ndimage.gaussian_filter(values, sigma, order=(0, 2))
    gxy = ndimage.gaussian_filter(values, sigma, order=(1, 1))
    tmp = np.sqrt((gxx - gyy) ** 2 + 4 * gxy**2)
    lam = (gxx + gyy - tmp) / 2  # наименьшее собственное число: сильно отрицательно на ярком гребне
    del g
    return np.clip(-lam, 0, None) * sigma**2


def measure(array: np.ndarray, spacing: Spacing) -> dict:
    values = axis.normalize(array)
    rows, cols = values.shape
    sx, sy = spacing
    valid = axis.valid_rows(values)
    vr = np.nonzero(valid)[0]
    if len(vr) < 20:
        return {}
    top, bottom = int(vr[0]), int(vr[-1])
    ys, xs, width = _column_center(array, spacing)
    center = float(np.median(xs)) if len(xs) else cols / 2

    # кость по порогу между мягкими тканями и плотной костью
    soft, dense = np.percentile(values[valid], (50, 98))
    bone = values > soft + (dense - soft) * 0.25
    xx = np.arange(cols)[None, :]
    lateral = np.abs(xx - center) > width * 0.9  # вне позвоночного столба

    def band(y0, y1):
        m = np.zeros_like(bone)
        m[max(y0, 0) : max(y1, 0)] = True
        return m

    h = bottom - top
    low = band(bottom - int(h * 0.25), bottom + 1) & lateral
    high = band(top, top + int(h * 0.2)) & lateral
    # гребни: доля кости по бокам внизу и насколько высоко они поднимаются от нижнего края
    iliac_frac = float(bone[low].mean()) if low.any() else 0.0
    lat_rows = np.nonzero((bone & lateral)[top : bottom + 1].mean(axis=1) > 0.08)[0] + top
    low_rows = lat_rows[lat_rows > top + h * 0.5]
    iliac_height_mm = float((bottom - low_rows.min()) * sy) if len(low_rows) else 0.0
    ribs_frac = float(bone[high].mean()) if high.any() else 0.0

    # посторонние предметы: тонкие очень яркие гребни вне столба.
    # Поле дополнения (нули по краям и прямоугольные вырезы) даёт резкий перепад, который
    # детектор принимает за линию, — заполняем его ближайшими значениями и не смотрим у границы.
    pad = array == 0
    pad = ndimage.binary_opening(pad, np.ones((3, 3)))
    filled = values
    if pad.any():
        idx = ndimage.distance_transform_edt(pad, return_distances=False, return_indices=True)
        filled = values[tuple(idx)]
    inside = ~ndimage.binary_dilation(pad, iterations=5)
    inside[:4], inside[-4:], inside[:, :4], inside[:, -4:] = False, False, False, False
    r1 = _ridges(filled, 1.0) * inside
    r3 = _ridges(filled, 3.0)
    thin = (r1 > 0.12) & (r1 > 1.5 * r3) & (values > np.percentile(values[valid], 97)) & lateral & valid[:, None]
    thin = ndimage.binary_opening(thin, np.ones((1, 1)))
    lab, n = ndimage.label(thin, np.ones((3, 3)))
    sizes = ndimage.sum(thin, lab, range(1, n + 1)) if n else np.array([])
    long_objs = int((sizes >= 12).sum())
    # компактные сверхъяркие пятна (скобы, металл): ярче 99.8 перцентиля кости
    very = (values >= 0.995) & lateral & valid[:, None]
    lab2, n2 = ndimage.label(very)
    spots = int((ndimage.sum(very, lab2, range(1, n2 + 1)) >= 4).sum()) if n2 else 0

    # зона поиска предметов: вне столба, внутри снятого поля (без нулей дополнения — раньше они
    # попадали в перцентиль и размывали его по-разному на каждом снимке) и без таза:
    # края гребней подвздошных костей давали большинство ложных тревог
    area = lateral & valid[:, None] & inside
    # таз ищем только вне столба: иначе позвоночник склеивает с тазом рёбра и всё, что к ним
    # прилегает (в т.ч. косточки белья вверху), и они исключаются вместе с тазом
    pelvis_mask = ndimage.binary_closing(bone, np.ones((5, 5))) & lateral
    lab3, _ = ndimage.label(pelvis_mask)
    ids = [k for k in np.unique(lab3[max(bottom - 3, 0) : bottom + 1]) if k]
    big = [k for k in ids if (lab3 == k).sum() > bone.size * 0.005]  # крупные кости у нижнего края — таз
    if big:
        area &= ~ndimage.binary_dilation(np.isin(lab3, big), iterations=4)
    fo_score = float(np.percentile(r1[area], 99.5)) if area.any() else 0.0

    return dict(
        sp_iliac_frac=iliac_frac,
        sp_iliac_height_mm=iliac_height_mm,
        sp_ribs_frac=ribs_frac,
        sp_field_height_mm=float(h * sy),
        sp_center_offset_mm=float((center - cols / 2) * sx),
        fo_thin_pixels=float(thin.sum()),
        fo_thin_objects=float(long_objs),
        fo_spots=float(spots),
        fo_ridge_p99=fo_score,
    )
