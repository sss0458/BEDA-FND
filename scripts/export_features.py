#!/usr/bin/env python3
"""Export test-set evidence representations for paper visualizations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--bert", default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch")
    parser.add_argument(
        "--seed", type=int, required=True, help="integer random seed chosen for this run"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    source_dir = project_root / "src"
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)

    import torch
    from config import STRUCTURAL_VARIANTS
    from main import seed_everything
    from model.beda_fnd import BEDAFNDModel
    from run import Run
    from utils.utils import clipdata2gpu

    seed_everything(args.seed)
    variant = STRUCTURAL_VARIANTS[args.variant]
    config = {
        "batchsize": 64,
        "max_len": 197,
        "num_workers": 4,
        "vocab_file": "./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt",
        "data_root": str(args.data_root) if args.data_root else "",
        "seed": args.seed,
        "dataset": args.dataset,
    }
    train_loader, validation_loader, test_loader = Run(config).get_dataloader(
        args.dataset
    )
    del train_loader, validation_loader

    model = BEDAFNDModel(
        768,
        [384],
        args.bert,
        320,
        0.2,
        ablation="beda_full",
        signal_mode="off",
        acceptance_mode="off",
        bem_mode=variant["bem_mode"],
        drcm_mode=variant["drcm_mode"],
        dcea_mode=variant["dcea_mode"],
        state_mode=variant["state_mode"],
        output_mode=variant["output_mode"],
    ).cuda()
    state = torch.load(checkpoint, map_location="cuda")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def capture_input(name):
        def hook(_module, values):
            captured[name] = values[0].detach()

        return hook

    def capture_output(name):
        def hook(_module, _values, result):
            captured[name] = result.detach()

        return hook

    handles = [
        model.domain_aware_text_classifier.register_forward_pre_hook(
            capture_input("text")
        ),
        model.domain_aware_image_classifier.register_forward_pre_hook(
            capture_input("image")
        ),
        model.domain_aware_fusion_classifier.register_forward_pre_hook(
            capture_input("fusion")
        ),
    ]
    if hasattr(model, "plain_fusion_head"):
        handles.append(
            model.plain_fusion_head[2].register_forward_hook(
                capture_output("plain_ffn")
            )
        )

    collected: dict[str, list[np.ndarray]] = {
        "labels": [],
        "categories": [],
        "probability": [],
        "branch_probability": [],
        "weights": [],
        "text_embedding": [],
        "image_embedding": [],
        "fusion_embedding": [],
        "decision_embedding": [],
    }
    with torch.inference_mode():
        for batch in test_loader:
            captured.clear()
            data = clipdata2gpu(batch)
            result = model(**data, analysis_mode="normal")
            batch_size = result[0].numel()
            has_branches = model.branch_diagnostics_available
            branch_probability = (
                torch.stack(
                    [result[7].reshape(-1), result[8].reshape(-1), result[9].reshape(-1)],
                    dim=1,
                )
                if has_branches else result[0].new_empty((batch_size, 0))
            )
            if not has_branches:
                weights = branch_probability.clone()
            elif variant["output_mode"] in ("no_arbitration", "fixed_mean"):
                weights = branch_probability.new_full((batch_size, 3), 1.0 / 3.0)
            elif len(result) > 10:
                weights = result[10]
            else:
                weights = branch_probability.new_full((batch_size, 3), 1.0 / 3.0)

            if has_branches:
                text_embedding = captured["text"]
                image_embedding = captured["image"]
                fusion_embedding = captured["fusion"]
            else:
                text_embedding = result[0].new_empty((batch_size, 0))
                image_embedding = text_embedding.clone()
                fusion_embedding = text_embedding.clone()
            if variant["output_mode"] in ("plain_ffn", "plain_ffn_early"):
                decision_embedding = captured["plain_ffn"]
            else:
                stacked = torch.stack(
                    [text_embedding, image_embedding, fusion_embedding], dim=1
                )
                decision_embedding = torch.sum(weights.unsqueeze(-1) * stacked, dim=1)

            values = {
                "labels": data["label"].reshape(-1),
                "categories": data["category"].reshape(-1),
                "probability": result[0].reshape(-1),
                "branch_probability": branch_probability,
                "weights": weights,
                "text_embedding": text_embedding,
                "image_embedding": image_embedding,
                "fusion_embedding": fusion_embedding,
                "decision_embedding": decision_embedding,
            }
            for key, value in values.items():
                collected[key].append(value.detach().cpu().numpy())

    for handle in handles:
        handle.remove()
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    arrays["prediction"] = (arrays["probability"] >= 0.5).astype(np.int64)
    arrays["correct"] = (arrays["prediction"] == arrays["labels"]).astype(np.int64)
    arrays["branch_conflict"] = (
        arrays["branch_probability"].max(axis=1)
        - arrays["branch_probability"].min(axis=1)
        if model.branch_diagnostics_available
        else np.full(len(arrays["labels"]), np.nan)
    )
    np.savez_compressed(output, **arrays)
    metadata = {
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "branch_diagnostics_available": model.branch_diagnostics_available,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "output_mode": variant["output_mode"],
        "bem_mode": variant["bem_mode"],
        "drcm_mode": variant["drcm_mode"],
        "dcea_mode": variant["dcea_mode"],
        "rows": int(len(arrays["labels"])),
        "accuracy": float(np.mean(arrays["correct"])),
        "embedding_shapes": {
            key: list(arrays[key].shape)
            for key in (
                "text_embedding",
                "image_embedding",
                "fusion_embedding",
                "decision_embedding",
            )
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
