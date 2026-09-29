"""Дополнительные серии (ТЗ п. 2.6): визуализация нарушения и текстовое описание в DICOM SR.

Для каждого снимка:
  * PNG с разметкой (ось позвоночника, зона гребней, область интереса бедра, головка);
  * DICOM Secondary Capture (RGB) той же разметкой — в том же исследовании, новая серия;
  * DICOM Basic Text SR с перечнем критериев и объяснением решения.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from PIL import Image as P, ImageDraw, ImageFont
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from . import axis, roi

SC_CLASS = "1.2.840.10008.5.1.4.1.1.7"        # Secondary Capture Image Storage
SR_CLASS = "1.2.840.10008.5.1.4.1.1.88.11"    # Basic Text SR Storage
RED, GREEN, YELLOW, CYAN = (230, 50, 40), (60, 200, 80), (255, 215, 0), (0, 190, 255)


def _font(size):
    for f in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(it) -> P.Image:
    a = axis.normalize(it.array)
    scale = 3
    img = P.fromarray((a * 255).astype(np.uint8)).convert("RGB").resize((a.shape[1] * scale, a.shape[0] * scale))
    d = ImageDraw.Draw(img)
    crit = it.result.get("criteria", {})
    spacing = it.result.get("spacing", (0.6, 0.6))
    bad = any(flag for flag, _, _ in crit.values())
    if it.region == "spine":
        ys, xs = axis.centerline(it.array, spacing)
        if len(ys):
            n = max(int(len(ys) * 0.15), 3)
            top = (np.median(xs[:n]) * scale, np.median(ys[:n]) * scale)
            bot = (np.median(xs[-n:]) * scale, np.median(ys[-n:]) * scale)
            ok = not crit.get("Не выравнена ось позвоночника", (0,))[0]
            d.line([bot, top], fill=GREEN if ok else RED, width=4)
            d.line([bot, (bot[0], top[1])], fill=CYAN, width=2)
        h = a.shape[0] * scale
        ok = not crit.get("Некорректная укладка", (0,))[0]
        d.rectangle([2, h * 0.75, a.shape[1] * scale - 3, h - 3], outline=GREEN if ok else RED, width=3)
    else:
        lm = roi.landmarks(it.array, spacing, it.side)
        if lm:
            x0, y0, x1, y1 = [v * scale for v in lm["box"]]
            ok = not crit.get("Некорректная область интереса", (0,))[0]
            d.rectangle([x0, y0, x1, y1], outline=GREEN if ok else RED, width=3)
            hx, hy, r = lm["head"]
            ok = not crit.get("Некорректная укладка", (0,))[0]
            d.ellipse([(hx - r) * scale, (hy - r) * scale, (hx + r) * scale, (hy + r) * scale],
                      outline=GREEN if ok else RED, width=3)
            yb = lm["lt_bottom"] * scale
            d.line([(x0, yb), (x1, yb)], fill=YELLOW, width=2)
    # подпись
    lines = [("ЕСТЬ НАРУШЕНИЕ" if bad else "КАЧЕСТВЕННОЕ", RED if bad else GREEN)]
    lines += [(f"{'[НАРУШЕНИЕ]' if flag else '[норма]'} {name}", RED if flag else GREEN) for name, (flag, _, _) in crit.items()]
    f = _font(max(14, img.width // 45))
    y = 6
    for text, color in lines:
        d.rectangle([4, y - 2, 12 + d.textlength(text, font=f), y + f.size + 4], fill=(0, 0, 0))
        d.text((8, y), text, font=f, fill=color)
        y += f.size + 8
    d.text((8, img.height - f.size - 8), "ИИ-контроль качества DXA · не для диагностики", font=_font(max(11, img.width // 70)),
           fill=(200, 200, 200))
    return img


def _base(src, sop_class, modality, series_desc):
    ds = Dataset()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    for tag in ("PatientID", "PatientName", "PatientBirthDate", "PatientSex", "StudyInstanceUID",
                "StudyID", "StudyDate", "StudyTime", "AccessionNumber", "ReferringPhysicianName"):
        setattr(ds, tag, src.get(tag, "") or "")
    ds.Modality = modality
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 9001 if modality == "OT" else 9002
    ds.InstanceNumber = 1
    ds.SeriesDescription = series_desc
    now = dt.datetime.now()
    ds.ContentDate, ds.ContentTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
    ds.Manufacturer = "DXA-QC (хакатон)"
    ds.SpecificCharacterSet = "ISO_IR 192"
    return ds


def secondary_capture(it, img: P.Image) -> Dataset:
    ds = _base(it.ds, SC_CLASS, "OT", "ИИ: контроль качества DXA — разметка")
    arr = np.asarray(img.convert("RGB"))
    ds.ConversionType = "WSD"
    ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
    ds.Rows, ds.Columns = arr.shape[:2]
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = arr.tobytes()
    return ds


def text_sr(it) -> Dataset:
    ds = _base(it.ds, SR_CLASS, "SR", "ИИ: контроль качества DXA — заключение")
    crit = it.result.get("criteria", {})
    bad = [n for n, (f, _, _) in crit.items() if f]

    def code(value, meaning, scheme="99DXAQC"):
        c = Dataset()
        c.CodeValue, c.CodingSchemeDesignator, c.CodeMeaning = value, scheme, meaning
        return c

    def text_item(meaning, text, rel="CONTAINS"):
        item = Dataset()
        item.RelationshipType, item.ValueType = rel, "TEXT"
        item.ConceptNameCodeSequence = Sequence([code("TXT", meaning)])
        item.TextValue = text[:1024]
        return item

    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = Sequence([code("QC", "Контроль качества денситометрии")])
    ds.CompletionFlag, ds.VerificationFlag = "COMPLETE", "UNVERIFIED"
    items = [text_item("Изображение", f"SOPInstanceUID {it.image_uid}"),
             text_item("Область", f"{it.region}{' ' + it.side if it.side else ''}"),
             text_item("Итог", ("Есть нарушение: " + "; ".join(bad)) if bad else "Исследование качественное")]
    items += [text_item(name, ("НАРУШЕНИЕ. " if flag else "норма. ") + text) for name, (flag, _, text) in crit.items()]
    items.append(text_item("Примечание", "Результат ИИ-сервиса, требует подтверждения специалистом."))
    ds.ContentSequence = Sequence(items)
    ref = Dataset()
    ref.ReferencedSOPClassUID = it.ds.get("SOPClassUID", "")
    ref.ReferencedSOPInstanceUID = it.image_uid
    ds.CurrentRequestedProcedureEvidenceSequence = Sequence([])
    return ds


def write_series(it, out_dir: Path):
    out_dir = Path(out_dir) / _safe(it.study_uid)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = overlay(it)
    stem = _safe(it.image_uid)[-40:]
    img.save(out_dir / f"{stem}.png")
    secondary_capture(it, img).save_as(out_dir / f"{stem}_SC.dcm", enforce_file_format=True)
    text_sr(it).save_as(out_dir / f"{stem}_SR.dcm", enforce_file_format=True)
    return out_dir / f"{stem}.png"


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))
