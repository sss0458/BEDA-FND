#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/state_ablations}"
RUN_SEED="${RUN_SEED:-${REPORT_SEED:-}}"
: "${RUN_SEED:?Set RUN_SEED to your chosen integer random seed.}"
DATASETS="${DATASETS:-weibo weibo21}"
STATE_MODES="${STATE_MODES:-dual feature_only decision_only without_both_channels legacy}"
NEW_MODULE_LR_MULTIPLIER="${NEW_MODULE_LR_MULTIPLIER:-5}"

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

for state_mode in ${STATE_MODES}; do
    for dataset in ${DATASETS}; do
        run_dir="${OUTPUT_ROOT}/${dataset}/${state_mode}/seed_${RUN_SEED}"
        checkpoint="$(initial_checkpoint "${dataset}")"
        if [[ -z "${checkpoint}" ]]; then
            echo "missing initial checkpoint for ${dataset}" >&2
            exit 2
        fi
        epochs=15
        early_stop=5
        if [[ "${state_mode}" == "legacy" || "${state_mode}" == "without_both_channels" ]]; then
            # These modes have no active state-calibrator path. Loading and
            # evaluating the common warm start is the clean mechanism
            # ablation; optimizer steps would have no state-gradient target.
            epochs=0
            early_stop=1
        fi
        mkdir -p "${run_dir}"
        "${PYTHON_BIN}" "${PROJECT_ROOT}/src/main.py" \
            --dataset "${dataset}" \
            --variant full \
            --state-mode "${state_mode}" \
            --data-root "$(data_root "${dataset}")" \
            --init-checkpoint "${checkpoint}" \
            --output-dir "${run_dir}" \
            --seed "${RUN_SEED}" \
            --epochs "${epochs}" \
            --early-stop "${early_stop}" \
            --learning-rate 0.0001 \
            --new-module-lr-multiplier "${NEW_MODULE_LR_MULTIPLIER}" \
            --selection-metric accuracy \
            --freeze-backbone \
            --train-scope state \
            --skip-test \
            > "${run_dir}/train.log" 2>&1

        selected="${run_dir}/checkpoints/parameter_beda_fnd_best_accuracy.pkl"
        eval_dir="${run_dir}/fixed_test"
        mkdir -p "${eval_dir}"
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate.py" \
            --project-root "${PROJECT_ROOT}" \
            --dataset "${dataset}" \
            --variant full \
            --state-mode "${state_mode}" \
            --data-root "$(data_root "${dataset}")" \
            --checkpoint "${selected}" \
            --seed "${RUN_SEED}" \
            --output-dir "${eval_dir}" \
            > "${eval_dir}/evaluate.log" 2>&1
    done
done

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${RUN_SEED}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
seed = sys.argv[2]
rows = []
for dataset in ("weibo", "weibo21"):
    for mode in (
        "dual", "feature_only", "decision_only",
        "without_both_channels", "legacy",
    ):
        path = root / dataset / mode / f"seed_{seed}" / "fixed_test" / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "dataset": dataset,
            "state_mode": mode,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "auc": metrics["auc"],
            "real_f1": metrics["per_class"]["real"]["f1"],
            "fake_f1": metrics["per_class"]["fake"]["f1"],
        })
with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2, ensure_ascii=False))
PY
