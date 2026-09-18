#!/usr/bin/env python3
"""Profile inference complexity for BEDA-FND structural variants.

The script intentionally uses the same synthetic input shapes for every path.
FLOPs do not depend on the sample values, and this avoids counting data-loader
work.  PyTorch's profiler estimates FLOPs for supported matrix-multiplication
and convolution operators, so the report calls the result ``profiled_flops``
rather than presenting it as an exact symbolic count.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


# Compatibility with data dependencies that still reference it.
if not hasattr(np, "float"):
    np.float = float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing src/model and src/pretrained_model",
    )
    parser.add_argument(
        "--bert",
        default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-length", type=int, default=197)
    parser.add_argument("--clip-text-length", type=int, default=52)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--latency-runs", type=int, default=20)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("full", "without_drcm", "without_bem", "without_refine", "without_dcea", "without_all", "without_both"),
        default=("full", "without_drcm", "without_bem", "without_refine", "without_dcea", "without_all"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_inputs(args: argparse.Namespace, device: torch.device) -> dict:
    batch = args.batch_size
    return {
        "content": torch.ones(
            batch, args.text_length, dtype=torch.long, device=device
        ),
        "content_masks": torch.ones(
            batch, args.text_length, dtype=torch.long, device=device
        ),
        "image": torch.zeros(
            batch,
            3,
            args.image_size,
            args.image_size,
            dtype=torch.float32,
            device=device,
        ),
        "clip_image": torch.zeros(
            batch,
            3,
            args.image_size,
            args.image_size,
            dtype=torch.float32,
            device=device,
        ),
        "clip_text": torch.ones(
            batch, args.clip_text_length, dtype=torch.long, device=device
        ),
        "category": torch.zeros(batch, dtype=torch.long, device=device),
    }


def model_kwargs(name: str) -> dict:
    from config import STRUCTURAL_VARIANTS

    common = {
        "emb_dim": 768,
        "mlp_dims": [384],
        "out_channels": 320,
        "dropout": 0.2,
    }
    variant = STRUCTURAL_VARIANTS[name]
    return {
        **common,
        "ablation": "beda_full",
        "bem_mode": variant["bem_mode"],
        "drcm_mode": variant["drcm_mode"],
        "dcea_mode": variant["dcea_mode"],
        "state_mode": variant["state_mode"],
        "output_mode": variant["output_mode"],
        "signal_mode": "off",
        "acceptance_mode": "off",
    }


def active_parameter_counter(model: torch.nn.Module):
    active_ids: set[int] = set()
    parameter_sizes = {id(p): p.numel() for p in model.parameters()}
    handles = []

    def record(module, _inputs, _output):
        for parameter in module.parameters(recurse=False):
            active_ids.add(id(parameter))

    for module in model.modules():
        handles.append(module.register_forward_hook(record))

    def finish() -> int:
        for handle in handles:
            handle.remove()
        return sum(parameter_sizes[item] for item in active_ids)

    return finish


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_one(name: str, args: argparse.Namespace, device: torch.device) -> dict:
    from model.beda_fnd import BEDAFNDModel

    kwargs = model_kwargs(name)
    kwargs["bert"] = args.bert
    model = BEDAFNDModel(**kwargs).to(device).eval()
    inputs = make_inputs(args, device)

    finish_active_count = active_parameter_counter(model)
    with torch.inference_mode():
        model(**inputs)
    active_parameters = finish_active_count()

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(**inputs)
    synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_flops=True) as prof:
        with torch.inference_mode():
            model(**inputs)
        synchronize(device)
    profiled_flops = int(sum(event.flops or 0 for event in prof.key_averages()))

    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.latency_runs):
            model(**inputs)
    synchronize(device)
    latency_ms = (
        (time.perf_counter() - start) * 1000.0 / args.latency_runs
    )

    result = {
        "model": name,
        "batch_size": args.batch_size,
        "profiled_flops_per_batch": profiled_flops,
        "profiled_flops_per_sample": profiled_flops / args.batch_size,
        "profiled_gflops_per_sample": profiled_flops / args.batch_size / 1e9,
        "all_registered_parameters": sum(p.numel() for p in model.parameters()),
        "active_forward_parameters": active_parameters,
        "latency_ms_per_batch": latency_ms,
        "latency_ms_per_sample": latency_ms / args.batch_size,
    }

    del inputs, model, prof
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def add_comparisons(results: list[dict]) -> dict:
    indexed = {item["model"]: item for item in results}
    comparisons = {}
    if "full" in indexed and "without_dcea" in indexed:
        full = indexed["full"]["profiled_flops_per_sample"]
        no_dcea = indexed["without_dcea"]["profiled_flops_per_sample"]
        comparisons["dcea_marginal_flops"] = full - no_dcea
        comparisons["dcea_marginal_flops_percent_of_full"] = (
            (full - no_dcea) / full * 100.0
        )
    return comparisons


def main() -> None:
    args = parse_args()
    # Model assets are resolved relative to ``src``.
    source_dir = args.project_root.expanduser().resolve() / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_dir}")
    if args.output:
        args.output = args.output.expanduser().resolve()
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Chinese-CLIP loader")
    device = torch.device("cuda")
    results = []
    for name in args.variants:
        result = profile_one(name, args, device)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "input": {
            "batch_size": args.batch_size,
            "text_length": args.text_length,
            "clip_text_length": args.clip_text_length,
            "image_size": args.image_size,
        },
        "measurement": (
            "PyTorch profiler with_flops=True; supported matmul/conv ops only"
        ),
        "results": results,
        "comparisons": add_comparisons(results),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
