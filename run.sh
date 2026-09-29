#!/usr/bin/env bash
# Сборка и запуск DXA-QC в контейнере (Linux / UNIX-подобные системы).
#
#   ./run.sh build                        # собрать образ
#   ./run.sh predict <вход> <выход>       # пакетная обработка: каталог или zip → <выход>/results.xlsx|csv, series.zip
#   ./run.sh api                          # веб-интерфейс и API на http://localhost:8000
set -euo pipefail

IMAGE="${IMAGE:-dxa-qc}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_assets() {
    [[ -f "${DIR}/models/quality_model.joblib" ]] || { echo "нет models/quality_model.joblib — см. README (train.py)"; exit 1; }
}

predict() {
    local input="$1" output="$2"
    [[ -e "${input}" ]] || { echo "нет входа: ${input}"; exit 1; }
    mkdir -p "${output}"
    input="$(cd "$(dirname "${input}")" && pwd)/$(basename "${input}")"
    output="$(cd "${output}" && pwd)"
    docker run --rm \
        -v "${input}:/data/input:ro" -v "${output}:/data/output" \
        "${IMAGE}" \
        python -m dxa.predict --input /data/input --output /data/output/results.xlsx --series /data/output/series
}

case "${1:-}" in
    build)   check_assets; docker build -t "${IMAGE}" "${DIR}" ;;
    predict) predict "${2:?вход}" "${3:?выход}" ;;
    api)     docker run --rm -p 8000:8000 "${IMAGE}" ;;
    *) echo "Использование: $0 {build|predict <вход> <выход>|api}"; exit 1 ;;
esac
