"""Пакетная обработка из командной строки.

    python -m dxa.predict --input /data/input --output /data/output/results.xlsx [--series /data/output/series]

--input  — каталог (любой вложенности) или zip с DICOM-файлами;
--output — .xlsx или .csv (второй формат пишется рядом автоматически);
--series — каталог для доп. серий (PNG, DICOM SC, DICOM SR); рядом создаётся series.zip.
Код возврата 0, даже если отдельные файлы не обработались: ошибки — в колонке error.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from .pipeline import Service

COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "projection", "quality_class", "violation_type",
           "processing_status", "time_of_processing", "details", "duplicate_of", "error"]


def save_table(rows: list[dict], output: Path):
    df = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    df = df[COLUMNS + [c for c in df.columns if c not in COLUMNS]]
    output.parent.mkdir(parents=True, exist_ok=True)
    xlsx, csv = output.with_suffix(".xlsx"), output.with_suffix(".csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="results")
        ws = w.sheets["results"]
        for col, width in zip("ABCDEFGHIJKL", (40, 30, 30, 34, 8, 8, 44, 10, 10, 90, 20, 30)):
            ws.column_dimensions[col].width = width
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="DXA-QC: контроль качества денситометрии")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--series", default=None)
    ap.add_argument("--model", default="models/quality_model.joblib")
    ap.add_argument("--siglip", default=None, help="локальный каталог весов SigLIP (по умолчанию из кэша HF)")
    args = ap.parse_args(argv)

    t0 = time.time()
    kwargs = {"model_path": args.model}
    if args.siglip:
        kwargs["siglip_dir"] = args.siglip
    service = Service(**kwargs)
    print(f"модели загружены за {time.time() - t0:.1f} с", flush=True)
    rows = service.process(args.input, args.series,
                           progress=lambda i, n: print(f"  исследование {i}/{n}", flush=True))
    df = save_table(rows, Path(args.output))
    if args.series and Path(args.series).exists():
        shutil.make_archive(str(Path(args.series)), "zip", args.series)
    ok = (df["processing_status"] == "Success").sum()
    print(f"готово: {len(df)} файлов, успешно {ok}, с нарушением {int((df['quality_class'] == 1).sum())}, "
          f"время {time.time() - t0:.1f} с → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
