#!/usr/bin/env python3
"""Check that both state-calibration channels receive classification gradients."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("weibo", "weibo21"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True,
                        help="integer random seed chosen for this run")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.project_root.resolve() / "src"
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)

    from main import seed_everything
    from model.beda_fnd import BEDAFNDModel
    from run import Run
    from utils.utils import clipdata2gpu

    seed_everything(args.seed)
    loader_config = {
        "batchsize": 16,
        "max_len": 197,
        "num_workers": 0,
        "vocab_file": "./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt",
        "data_root": str(args.data_root.resolve()),
        "seed": args.seed,
        "dataset": args.dataset,
    }
    train_loader, _, _ = Run(loader_config).get_dataloader(args.dataset)
    model = BEDAFNDModel(
        emb_dim=768,
        mlp_dims=[384],
        bert="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch",
        out_channels=320,
        dropout=0.2,
        ablation="beda_full",
        state_mode="dual",
    ).cuda()
    state = torch.load(args.checkpoint.resolve(), map_location="cuda")
    model.load_state_dict(state, strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("interaction_feature_state_calibrator")
            or name.startswith("interaction_decision_state_calibrator")
        )
    model.eval()
    batch = clipdata2gpu(next(iter(train_loader)))
    prediction = model(**batch)[0].reshape(-1)
    loss = torch.nn.functional.binary_cross_entropy(
        prediction, batch["label"].float().reshape(-1)
    )
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            norm = 0.0 if parameter.grad is None else float(parameter.grad.norm())
            print(f"{name}\t{norm:.12g}")


if __name__ == "__main__":
    main()
