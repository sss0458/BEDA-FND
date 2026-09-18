#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/ablations}"
RUN_SEED="${RUN_SEED:-${REPORT_SEED:-}}"
: "${RUN_SEED:?Set RUN_SEED to your chosen integer random seed.}"
DATASETS="${DATASETS:-weibo weibo21}"
VARIANTS="${VARIANTS:-full without_drcm without_bem without_refine without_dcea without_all}"

data_root() {
    if [[ "$1" == "weibo" ]]; then
        printf '%s' "${WEIBO_ROOT:-${PROJECT_ROOT}/data/weibo}"
    else
        printf '%s' "${WEIBO21_ROOT:-${PROJECT_ROOT}/data/weibo21}"
    fi
}

initial_checkpoint() {
    if [[ "$1" == "weibo" ]]; then
        printf '%s' "${WEIBO_INIT_CHECKPOINT:-}"
    else
        printf '%s' "${WEIBO21_INIT_CHECKPOINT:-}"
    fi
}

for dataset in ${DATASETS}; do
    for variant in ${VARIANTS}; do
        output_dir="${OUTPUT_ROOT}/${dataset}/${variant}/seed_${RUN_SEED}"
        command=(
            "${PYTHON_BIN}" "${PROJECT_ROOT}/src/main.py"
            --dataset "${dataset}"
            --variant "${variant}"
            --data-root "$(data_root "${dataset}")"
            --seed "${RUN_SEED}"
            --epochs "${EPOCHS:-15}"
            --early-stop "${EARLY_STOP:-5}"
            --learning-rate 0.0001
            --selection-metric accuracy
            --output-dir "${output_dir}"
            --skip-test
        )
        checkpoint="$(initial_checkpoint "${dataset}")"
        if [[ -n "${checkpoint}" ]]; then
            command+=(--init-checkpoint "${checkpoint}")
        fi
        mkdir -p "${output_dir}"
        "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    done
done
