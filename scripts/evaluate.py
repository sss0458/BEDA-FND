#!/usr/bin/env python3
"""Evaluate a fixed BEDA-FND checkpoint and save reusable predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--dataset", choices=("weibo", "weibo21"), required=True)
    parser.add_argument(
        "--variant",
        choices=("full", "without_drcm", "without_bem", "without_refine", "without_dcea", "without_all", "without_both"),
        required=True,
    )
    parser.add_argument(
        "--state-mode",
        choices=(
            "dual", "feature_only", "decision_only",
            "without_both_channels", "legacy",
        ),
        default=None,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--bert", default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--seed", type=int, required=True, help="integer random seed chosen for this run"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    prediction = (probability >= 0.5).astype(np.int64)
    confidence = np.maximum(probability, 1.0 - probability)
    correct = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.5, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        upper_closed = index == bins - 1
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if upper_closed
            else confidence < edges[index + 1]
        )
        if selected.any():
            value += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(value)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    source_dir = project_root / "src"
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)

    import torch

    from config import STRUCTURAL_VARIANTS
    from main import seed_everything
    from model.beda_fnd import BEDAFNDModel
    from run import Run
    from utils.utils import clipdata2gpu

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current visual encoders")

    seed_everything(args.seed)
    variant = STRUCTURAL_VARIANTS[args.variant]
    loader_config = {
        "batchsize": args.batch_size,
        "max_len": 197,
        "num_workers": args.num_workers,
        "vocab_file": "./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt",
        "data_root": str(args.data_root) if args.data_root else "",
        "seed": args.seed,
        "dataset": args.dataset,
    }
    train_loader, validation_loader, test_loader = Run(loader_config).get_dataloader(
        args.dataset
    )
    del train_loader, validation_loader

    model = BEDAFNDModel(
        emb_dim=768,
        mlp_dims=[384],
        bert=args.bert,
        out_channels=320,
        dropout=0.2,
        ablation="beda_full",
        signal_mode="off",
        acceptance_mode="off",
        bem_mode=variant["bem_mode"],
        drcm_mode=variant["drcm_mode"],
        dcea_mode=variant["dcea_mode"],
        state_mode=args.state_mode or variant["state_mode"],
        output_mode=variant["output_mode"],
    ).cuda()
    state = torch.load(checkpoint, map_location="cuda")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    labels, probabilities, categories = [], [], []
    branch_probabilities, routing_weights = [], []
    with torch.inference_mode():
        for batch in test_loader:
            data = clipdata2gpu(batch)
            result = model(**data)
            labels.append(data["label"].reshape(-1).cpu().numpy())
            categories.append(data["category"].reshape(-1).cpu().numpy())
            probabilities.append(result[0].reshape(-1).cpu().numpy())
            has_branches = model.branch_diagnostics_available
            branches = (
                torch.stack(
                    (result[7].reshape(-1), result[8].reshape(-1), result[9].reshape(-1)),
                    dim=1,
                )
                if has_branches else result[0].new_empty((result[0].numel(), 0))
            )
            branch_probabilities.append(branches.cpu().numpy())
            if not has_branches:
                weights = branches.clone()
            elif variant["output_mode"] in ("no_arbitration", "fixed_mean") or len(result) <= 10:
                weights = branches.new_full(branches.shape, 1.0 / 3.0)
            else:
                weights = result[10]
            routing_weights.append(weights.cpu().numpy())

    labels_array = np.concatenate(labels).astype(np.int64)
    probability_array = np.concatenate(probabilities).astype(np.float64)
    category_array = np.concatenate(categories).astype(np.int64)
    branch_array = np.concatenate(branch_probabilities).astype(np.float64)
    weight_array = np.concatenate(routing_weights).astype(np.float64)
    prediction_array = (probability_array >= 0.5).astype(np.int64)

    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels_array, prediction_array, labels=[0, 1], zero_division=0
    )
    metrics = {
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "branch_diagnostics_available": model.branch_diagnostics_available,
        "checkpoint_sha256": sha256_file(checkpoint),
        "samples": int(labels_array.size),
        "accuracy": float(accuracy_score(labels_array, prediction_array)),
        "macro_f1": float(f1_score(labels_array, prediction_array, average="macro")),
        "auc": float(roc_auc_score(labels_array, probability_array)),
        "brier": float(brier_score_loss(labels_array, probability_array)),
        "nll": float(log_loss(labels_array, np.clip(probability_array, 1e-7, 1 - 1e-7))),
        "ece_15": calibration_error(labels_array, probability_array),
        "confusion_matrix": confusion_matrix(labels_array, prediction_array).tolist(),
        "per_class": {
            "real": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(class_f1[0]),
                "support": int(support[0]),
            },
            "fake": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(class_f1[1]),
                "support": int(support[1]),
            },
        },
    }
    np.savez_compressed(
        output_dir / "predictions.npz",
        labels=labels_array,
        probability=probability_array,
        prediction=prediction_array,
        categories=category_array,
        branch_probability=branch_array,
        routing_weights=weight_array,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
