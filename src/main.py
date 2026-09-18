"""Train BEDA-FND and its structural ablations."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from config import STRUCTURAL_VARIANTS


SOURCE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("weibo", "weibo21"), required=True)
    parser.add_argument(
        "--variant", choices=tuple(STRUCTURAL_VARIANTS), default="full"
    )
    parser.add_argument("--data-root", default="")
    parser.add_argument(
        "--bert",
        default=str(SOURCE_DIR / "pretrained_model/chinese_roberta_wwm_base_ext_pytorch"),
    )
    parser.add_argument(
        "--bert-vocab-file",
        default=str(
            SOURCE_DIR / "pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt"
        ),
    )
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="integer random seed chosen for this run",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=197)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--new-module-lr-multiplier", type=float, default=5.0)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--train-scope", choices=("all", "state"), default="all"
    )
    parser.add_argument(
        "--state-mode",
        choices=(
            "dual", "feature_only", "decision_only",
            "without_both_channels", "legacy",
        ),
        default=None,
        help="four-state estimator used inside domain-conditioned arbitration",
    )
    parser.add_argument(
        "--selection-metric", choices=("accuracy", "auc"), default="accuracy"
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_config(args: argparse.Namespace) -> dict:
    variant = STRUCTURAL_VARIANTS[args.variant]
    return {
        "use_cuda": True,
        "batchsize": args.batch_size,
        "max_len": args.max_length,
        "early_stop": args.early_stop,
        "num_workers": args.num_workers,
        "vocab_file": args.bert_vocab_file,
        "bert": args.bert,
        "data_root": args.data_root,
        "weight_decay": args.weight_decay,
        "model": {"mlp": {"dims": [384], "dropout": 0.2}},
        "emb_dim": 768,
        "lr": args.learning_rate,
        "epoch": args.epochs,
        "seed": args.seed,
        "save_param_dir": args.output_dir,
        "dataset": args.dataset,
        "variant": args.variant,
        "arbitration_loss_weight": variant["arbitration_loss_weight"],
        "match_loss_weight": variant["match_loss_weight"],
        "domain_loss_weight": variant["domain_loss_weight"],
        "view_loss_weight": variant["view_loss_weight"],
        "init_checkpoint": args.init_checkpoint,
        "new_module_lr_multiplier": args.new_module_lr_multiplier,
        "skip_final_test": args.skip_test,
        "validation_only_checkpoint": True,
        "freeze_backbone": args.freeze_backbone,
        "train_scope": args.train_scope,
        "bem_mode": variant["bem_mode"],
        "drcm_mode": variant["drcm_mode"],
        "dcea_mode": variant["dcea_mode"],
        "state_mode": args.state_mode or variant["state_mode"],
        "output_mode": variant["output_mode"],
        "selection_metric": args.selection_metric,
    }


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(args.seed)

    launch_dir = Path.cwd()
    for attribute in ("data_root", "bert", "bert_vocab_file", "init_checkpoint", "output_dir"):
        value = getattr(args, attribute)
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = launch_dir / path
        setattr(args, attribute, str(path.resolve()))
    os.chdir(SOURCE_DIR)

    from run import Run

    Run(config=build_config(args)).main()


if __name__ == "__main__":
    main()
