# DXA-QC — контроль качества денситометрии. Работает полностью локально: веса внутри образа.
#
#   CPU:  docker build -t dxa-qc .
#   GPU:  docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t dxa-qc:gpu .
#
# Перед сборкой в каталоге должны быть models/quality_model.joblib и weights/siglip-vision
# (см. README: python train.py ...; python scripts/export_siglip.py).
FROM python:3.10.16-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.12.0
ARG TORCHVISION_VERSION=0.27.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DXA_JOBS=/data/jobs

# шрифт с кириллицей для подписей на доп. сериях; curl — для healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --upgrade pip==25.0.1 && pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY weights ./weights
COPY models ./models
COPY dxa ./dxa
COPY train.py evaluate_final.py ./

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "dxa.api:app", "--host", "0.0.0.0", "--port", "8000"]
