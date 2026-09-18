#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/outputs/ablations}"
FEATURE_ROOT="${FEATURE_ROOT:-${PROJECT_ROOT}/outputs/features}"
RUN_SEED="${RUN_SEED:-${REPORT_SEED:-}}"
: "${RUN_SEED:?Set RUN_SEED to your chosen integer random seed.}"
DATASETS="${DATASETS:-weibo weibo21}"
VARIANTS="${VARIANTS:-full without_drcm without_bem without_refine without_dcea without_all}"

mkdir -p "${FEATURE_ROOT}"
for dataset in ${DATASETS}; do
    data_root="${PROJECT_ROOT}/data/${dataset}"
    if [[ "${dataset}" == "weibo" ]]; then
        data_root="${WEIBO_ROOT:-${data_root}}"
    else
        data_root="${WEIBO21_ROOT:-${data_root}}"
    fi
    for variant in ${VARIANTS}; do
        checkpoint="${TRAIN_ROOT}/${dataset}/${variant}/seed_${RUN_SEED}/checkpoints/parameter_beda_fnd_best_accuracy.pkl"
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/export_features.py" \
            --dataset "${dataset}" \
            --variant "${variant}" \
            --data-root "${data_root}" \
            --seed "${RUN_SEED}" \
            --checkpoint "${checkpoint}" \
            --output "${FEATURE_ROOT}/${dataset}_${variant}.npz"
    done
done
