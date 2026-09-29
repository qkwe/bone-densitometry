"""Дымовые тесты: чтение устойчиво к мусору; полный прогон на фрагменте организатора.

    python -m pytest tests -q
Полный прогон пропускается, если нет модели или каталога с тестовым фрагментом (DXA_TEST_DIR).
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dxa.pipeline import read_items  # noqa: E402

TEST_DIR = Path(os.environ.get("DXA_TEST_DIR", "C:/dxa_test"))


def test_reader_reports_garbage(tmp_path):
    (tmp_path / "garbage.dcm").write_text("not a dicom")
    (tmp_path / "empty.dcm").write_bytes(b"")
    items = read_items(tmp_path)
    assert len(items) == 2
    assert all(it.error for it in items), "мусорные файлы должны попадать в отчёт как ошибка"


@pytest.mark.skipif(not (ROOT / "models" / "quality_model.joblib").exists() or not TEST_DIR.exists(),
                    reason="нет модели или тестового фрагмента")
def test_full_run(tmp_path):
    os.chdir(ROOT)
    from dxa.predict import main
    import pandas as pd

    assert main(["--input", str(TEST_DIR), "--output", str(tmp_path / "r.xlsx"), "--series", str(tmp_path / "s")]) == 0
    df = pd.read_csv(tmp_path / "r.csv")
    assert (df["processing_status"] == "Success").all()
    assert set(df["quality_class"]) <= {0, 1}
    for col in ("path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class",
                "violation_type", "processing_status", "time_of_processing"):
        assert col in df.columns
    assert (df["time_of_processing"] < 180).all()
    assert any((tmp_path / "s").rglob("*_SR.dcm"))
