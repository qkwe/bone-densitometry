#!/usr/bin/env bash
# Сборка и запуск DXA-QC в контейнере (Linux / UNIX-подобные системы).
#
#   ./run.sh build                        # собрать образ (CPU)
#   ./run.sh build-gpu                    # собрать образ с CUDA
#   ./run.sh predict <вход> <выход>       # пакетная обработка: каталог или zip → <выход>/results.xlsx|csv, series.zip
#   ./run.sh api                          # веб-интерфейс и API на http://localhost:8000
#   ./run.sh api-gpu | predict-gpu ...    # то же на GPU (нужен nvidia-container-toolkit)
set -euo pipefail

IMAGE="${IMAGE:-dxa-qc}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_assets() {
    [[ -f "${DIR}/models/quality_model.joblib" ]] || { echo "нет models/quality_model.joblib — см. README (train.py)"; exit 1; }
    [[ -f "${DIR}/weights/siglip-vision/model.safetensors" ]] || { echo "нет weights/siglip-vision — python scripts/export_siglip.py"; exit 1; }
}

predict() {
    local gpu="$1" input="$2" output="$3"
    [[ -e "${input}" ]] || { echo "нет входа: ${input}"; exit 1; }
    mkdir -p "${output}"
    input="$(cd "$(dirname "${input}")" && pwd)/$(basename "${input}")"
    output="$(cd "${output}" && pwd)"
    docker run --rm ${gpu} \
        -v "${input}:/data/input:ro" -v "${output}:/data/output" \
        "${IMAGE}${gpu:+:gpu}" \
        python -m dxa.predict --input /data/input --output /data/output/results.xlsx --series /data/output/series
}

case "${1:-}" in
    build)       check_assets; docker build -t "${IMAGE}" "${DIR}" ;;
    build-gpu)   check_assets; docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t "${IMAGE}:gpu" "${DIR}" ;;
    predict)     predict "" "${2:?вход}" "${3:?выход}" ;;
    predict-gpu) predict "--gpus all" "${2:?вход}" "${3:?выход}" ;;
    api)         docker run --rm -p 8000:8000 "${IMAGE}" ;;
    api-gpu)     docker run --rm --gpus all -p 8000:8000 "${IMAGE}:gpu" ;;
    *) echo "Использование: $0 {build|build-gpu|predict <вход> <выход>|predict-gpu ...|api|api-gpu}"; exit 1 ;;
esac
