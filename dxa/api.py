"""HTTP API и веб-интерфейс (ТЗ п. 3.2 «API для пакетной обработки», п. 2.6 «веб-интерфейс»).

    uvicorn dxa.api:app --host 0.0.0.0 --port 8000

GET  /                         — веб-интерфейс
GET  /health                   — проверка готовности
POST /api/jobs                 — загрузка zip / DICOM-файлов, запуск обработки (form-data: files)
POST /api/jobs/path            — обработка каталога на сервере: {"path": "/data/input"}
GET  /api/jobs/{id}            — статус и строки результата
GET  /api/jobs/{id}/results.xlsx | results.csv | series.zip
GET  /api/jobs/{id}/overlay/{path} — PNG с разметкой
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .pipeline import Service
from .predict import save_table

JOBS = Path(os.environ.get("DXA_JOBS", "out/jobs"))
MODEL = os.environ.get("DXA_MODEL", "models/quality_model.joblib")
SIGLIP = os.environ.get("DXA_SIGLIP")
app = FastAPI(title="DXA-QC", version="1.0")
_service: Service | None = None
_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def service() -> Service:
    global _service
    with _lock:
        if _service is None:
            _service = Service(MODEL, SIGLIP) if SIGLIP else Service(MODEL)
    return _service


def _run(job_id: str, src: Path):
    job = _jobs[job_id]
    out = JOBS / job_id
    try:
        t0 = time.time()
        rows = service().process(src, out / "series", progress=lambda i, n: job.update(progress=[i, n]))
        save_table(rows, out / "results.xlsx")
        if (out / "series").exists():
            shutil.make_archive(str(out / "series"), "zip", out / "series")
        job.update(status="done", rows=rows, seconds=round(time.time() - t0, 1))
    except Exception as e:
        job.update(status="error", error=f"{e}", trace=traceback.format_exc(limit=5))


def _start(src: Path, job_id: str) -> dict:
    _jobs[job_id] = dict(id=job_id, status="running", progress=[0, 0], created=time.time())
    threading.Thread(target=_run, args=(job_id, src), daemon=True).start()
    return {"id": job_id, "status": "running"}


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "model_loaded": _service is not None}


@app.post("/api/jobs")
async def create_job(files: list[UploadFile] = File(...)):
    job_id = uuid.uuid4().hex[:12]
    src = JOBS / job_id / "input"
    src.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(f.filename or "file.dcm").name
        (src / name).write_bytes(await f.read())
    return _start(src, job_id)


@app.post("/api/jobs/path")
def create_job_path(body: dict):
    path = Path(body.get("path", ""))
    if not path.exists():
        raise HTTPException(404, f"нет такого пути: {path}")
    return _start(path, uuid.uuid4().hex[:12])


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "нет такого задания")
    return JSONResponse({k: v for k, v in job.items() if k != "trace"})


@app.get("/api/jobs/{job_id}/{name}")
def job_file(job_id: str, name: str):
    if name not in ("results.xlsx", "results.csv", "series.zip"):
        raise HTTPException(404)
    path = JOBS / job_id / name
    if not path.exists():
        raise HTTPException(404, "файл ещё не готов")
    return FileResponse(path, filename=name)


@app.get("/api/jobs/{job_id}/overlay/{rel:path}")
def overlay(job_id: str, rel: str):
    base = (JOBS / job_id / "series").resolve()
    path = (base / rel).resolve()
    if base not in path.parents or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)
