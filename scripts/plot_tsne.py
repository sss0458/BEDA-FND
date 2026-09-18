#!/usr/bin/env python3
"""Create a reproducible 2x4 model-comparison t-SNE figure and audit metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


DATASETS = ("weibo", "weibo21")
DATASET_LABELS = {"weibo": "Weibo", "weibo21": "Weibo21"}
MODELS = ("full", "dammfnd", "without_bem", "without_dcea")
MODEL_LABELS = {
    "dammfnd": "DAMMFND",
    "without_bem": "w/o BEM",
    "without_dcea": "w/o DCEA",
    "full": "Full model",
}
PANEL_LABELS = {
    "full": "(a) Full model",
    "dammfnd": "(b) DAMMFND",
    "without_bem": "(c) w/o BEM",
    "without_dcea": "(d) w/o DCEA",
}
CLASS_COLORS = {0: "#3572B0", 1: "#D64A3A"}
CLASS_LABELS = {0: "Real", 1: "Fake"}
PROJECTION_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_features(feature_dir: Path):
    features = {}
    metadata = {}
    for dataset in DATASETS:
        for model in MODELS:
            stem = feature_dir / f"{dataset}_{model}"
            npz_path = stem.with_suffix(".npz")
            json_path = stem.with_suffix(".json")
            if not npz_path.exists():
                raise FileNotFoundError(npz_path)
            archive = np.load(npz_path)
            required = {
                "labels",
                "categories",
                "probability",
                "prediction",
                "correct",
                "decision_embedding",
            }
            missing = required.difference(archive.files)
            if missing:
                raise RuntimeError(f"{npz_path}: missing {sorted(missing)}")
            features[(dataset, model)] = {
                key: archive[key] for key in archive.files
            }
            metadata[(dataset, model)] = (
                json.loads(json_path.read_text(encoding="utf-8"))
                if json_path.exists()
                else {}
            )
            metadata[(dataset, model)]["feature_npz"] = str(npz_path.resolve())
            metadata[(dataset, model)]["feature_npz_sha256"] = sha256_file(npz_path)
    return features, metadata


def verify_alignment(features) -> dict[str, dict[str, object]]:
    audit = {}
    for dataset in DATASETS:
        reference = features[(dataset, "full")]
        dataset_audit = {
            "rows": int(len(reference["labels"])),
            "labels_match": {},
            "categories_match": {},
        }
        for model in MODELS:
            current = features[(dataset, model)]
            labels_match = np.array_equal(reference["labels"], current["labels"])
            categories_match = np.array_equal(
                reference["categories"], current["categories"]
            )
            dataset_audit["labels_match"][model] = bool(labels_match)
            dataset_audit["categories_match"][model] = bool(categories_match)
            if not labels_match or not categories_match:
                raise RuntimeError(
                    f"Sample ordering mismatch for {dataset}/{model}: "
                    f"labels={labels_match}, categories={categories_match}"
                )
        audit[dataset] = dataset_audit
    return audit


def project(values: np.ndarray):
    standardized = StandardScaler().fit_transform(values)
    components = min(50, standardized.shape[1], standardized.shape[0] - 1)
    pca = PCA(n_components=components, random_state=PROJECTION_RANDOM_STATE)
    reduced = pca.fit_transform(standardized)
    projection = TSNE(
        n_components=3,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        n_iter=1000,
        random_state=PROJECTION_RANDOM_STATE,
    ).fit_transform(reduced)
    return projection, reduced, float(pca.explained_variance_ratio_.sum())


def separation_metrics(reduced: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    class0 = reduced[labels == 0]
    class1 = reduced[labels == 1]
    centroid0 = class0.mean(axis=0)
    centroid1 = class1.mean(axis=0)
    between = float(np.linalg.norm(centroid0 - centroid1))
    within = 0.5 * (
        float(np.linalg.norm(class0 - centroid0, axis=1).mean())
        + float(np.linalg.norm(class1 - centroid1, axis=1).mean())
    )
    return {
        "silhouette_pca50": float(silhouette_score(reduced, labels)),
        "davies_bouldin_pca50": float(davies_bouldin_score(reduced, labels)),
        "calinski_harabasz_pca50": float(
            calinski_harabasz_score(reduced, labels)
        ),
        "centroid_to_within_ratio_pca50": between / max(within, 1e-12),
    }


def set_style() -> None:
    plt.rcParams.update(
        {
            # STIXGeneral is a bundled, embeddable Times-style face.  It keeps
            # the figure visually consistent with the ICASSP paper template.
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 9.2,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_point_cloud_axis(axis, projection: np.ndarray, dataset: str) -> None:
    """Apply the compact, axis-free perspective used by the paper figure."""
    axis.view_init(elev=18, azim=-58)
    try:
        axis.set_proj_type("persp", focal_length=0.9)
    except TypeError:
        axis.set_proj_type("persp")
    axis.set_box_aspect((1.0, 1.0, 0.72))
    for dimension, setter in enumerate(
        (axis.set_xlim, axis.set_ylim, axis.set_zlim)
    ):
        lower = float(projection[:, dimension].min())
        upper = float(projection[:, dimension].max())
        padding = max((upper - lower) * 0.055, 1e-6)
        setter(lower - padding, upper + padding)
    axis.set_axis_off()
    axis.text2D(
        -0.075,
        0.5,
        DATASET_LABELS[dataset],
        transform=axis.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=9.2,
        fontweight="normal",
    )


def main() -> int:
    args = parse_args()
    feature_dir = args.feature_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features, metadata = load_features(feature_dir)
    alignment = verify_alignment(features)

    coordinates = {}
    evidence_coordinates = {}
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            data = features[(dataset, model)]
            projection, reduced, explained = project(data["decision_embedding"])
            coordinates[f"{dataset}_{model}"] = projection
            # This four-dimensional vector is the actual evidence seen by the
            # final decision: the fused probability and each routed branch
            # contribution.  It complements the higher-dimensional latent view.
            evidence = np.column_stack(
                (
                    data["probability"],
                    data["weights"] * data["branch_probability"],
                )
            )
            evidence_projection, evidence_reduced, evidence_explained = project(
                evidence
            )
            evidence_coordinates[f"{dataset}_{model}"] = evidence_projection
            evidence_separation = separation_metrics(
                evidence_reduced, data["labels"]
            )
            row = {
                "dataset": dataset,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "rows": int(len(data["labels"])),
                "accuracy": float(np.mean(data["correct"])),
                "pca_explained_variance_50d": explained,
                **separation_metrics(reduced, data["labels"]),
                "evidence_pca_explained_variance": evidence_explained,
                **{
                    f"evidence_{key}": value
                    for key, value in evidence_separation.items()
                },
            }
            rows.append(row)

    np.savez_compressed(
        output_dir / "tsne_coordinates.npz",
        **{f"latent_{key}": value for key, value in coordinates.items()},
        **{
            f"evidence_{key}": value
            for key, value in evidence_coordinates.items()
        },
    )

    set_style()
    figure, axes = plt.subplots(
        2, 4, figsize=(7.10, 3.68), subplot_kw={"projection": "3d"}
    )
    metric_by_key = {(row["dataset"], row["model"]): row for row in rows}
    for row_index, dataset in enumerate(DATASETS):
        for column_index, model in enumerate(MODELS):
            axis = axes[row_index, column_index]
            data = features[(dataset, model)]
            projection = coordinates[f"{dataset}_{model}"]
            for label in (0, 1):
                selected = data["labels"] == label
                axis.scatter(
                    projection[selected, 0],
                    projection[selected, 1],
                    projection[selected, 2],
                    s=4.8 if dataset == "weibo" else 6.4,
                    alpha=0.62,
                    linewidths=0,
                    c=CLASS_COLORS[label],
                    depthshade=True,
                    rasterized=True,
                )
            style_point_cloud_axis(axis, projection, dataset)
            if column_index != 0:
                axis.texts[-1].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.5,
            markerfacecolor=CLASS_COLORS[label],
            markeredgewidth=0,
            label=CLASS_LABELS[label],
        )
        for label in (0, 1)
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.065),
        handletextpad=0.35,
        columnspacing=1.4,
    )
    figure.tight_layout(rect=(0.012, 0.175, 1, 0.99), w_pad=0.45, h_pad=0.10)
    for column_index, model in enumerate(MODELS):
        position = axes[1, column_index].get_position()
        figure.text(
            position.x0 + position.width / 2.0,
            0.145,
            PANEL_LABELS[model],
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="normal",
        )
    figure.text(
        0.5,
        0.012,
        "Fig. 2. Three-dimensional t-SNE visualization of test-set features.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        fontweight="normal",
    )
    figure.savefig(
        output_dir / "fig_model_comparison_tsne_icassp.png",
        dpi=500,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "fig_model_comparison_tsne_icassp.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    # A second, decision-faithful view.  Misclassified samples are marked with
    # black crosses so that the accuracy differences are visible without
    # claiming that t-SNE geometry itself is a performance metric.
    figure, axes = plt.subplots(
        2, 4, figsize=(7.10, 3.68), subplot_kw={"projection": "3d"}
    )
    for row_index, dataset in enumerate(DATASETS):
        for column_index, model in enumerate(MODELS):
            axis = axes[row_index, column_index]
            data = features[(dataset, model)]
            projection = evidence_coordinates[f"{dataset}_{model}"]
            for label in (0, 1):
                selected = data["labels"] == label
                axis.scatter(
                    projection[selected, 0],
                    projection[selected, 1],
                    projection[selected, 2],
                    s=4.8 if dataset == "weibo" else 6.4,
                    alpha=0.60,
                    linewidths=0,
                    c=CLASS_COLORS[label],
                    depthshade=True,
                    rasterized=True,
                )
            errors = data["correct"] == 0
            axis.scatter(
                projection[errors, 0],
                projection[errors, 1],
                projection[errors, 2],
                s=12 if dataset == "weibo" else 16,
                marker="x",
                c="#111111",
                alpha=0.75,
                linewidths=0.55,
                rasterized=True,
            )
            style_point_cloud_axis(axis, projection, dataset)
            if column_index != 0:
                axis.texts[-1].set_visible(False)

    evidence_legend_handles = legend_handles + [
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="",
            markersize=5.5,
            markeredgecolor="#111111",
            color="#111111",
            label="Misclassified",
        )
    ]
    figure.legend(
        handles=evidence_legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.065),
        handletextpad=0.35,
        columnspacing=1.25,
    )
    figure.tight_layout(rect=(0.012, 0.175, 1, 0.99), w_pad=0.45, h_pad=0.10)
    for column_index, model in enumerate(MODELS):
        position = axes[1, column_index].get_position()
        figure.text(
            position.x0 + position.width / 2.0,
            0.145,
            PANEL_LABELS[model],
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="normal",
        )
    figure.text(
        0.5,
        0.012,
        "Fig. 2. Three-dimensional t-SNE visualization of test-set features.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        fontweight="normal",
    )
    figure.savefig(
        output_dir / "fig_model_comparison_decision_evidence_tsne_icassp.png",
        dpi=500,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "fig_model_comparison_decision_evidence_tsne_icassp.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    fieldnames = list(rows[0].keys())
    with (output_dir / "model_comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "figures": {
            "latent": (
                "2x4 independent 3D t-SNE projections of the final latent "
                "decision embeddings"
            ),
            "decision_evidence": (
                "2x4 independent 3D t-SNE projections of [final probability, "
                "routed branch contributions], with errors marked"
            ),
        },
        "projection_protocol": {
            "preprocessing": "StandardScaler -> PCA(50) -> t-SNE(3)",
            "projection_random_state": PROJECTION_RANDOM_STATE,
            "tsne_perplexity": 30,
            "tsne_iterations": 1000,
            "comparison_note": (
                "Each panel is projected independently with identical settings. "
                "Compare class separation within panels, not absolute point coordinates across panels."
            ),
            "silhouette_note": (
                "Silhouette is computed in the standardized PCA-50 space, "
                "not on the displayed t-SNE coordinates."
            ),
        },
        "alignment_audit": alignment,
        "inputs": {
            f"{dataset}_{model}": metadata[(dataset, model)]
            for dataset in DATASETS
            for model in MODELS
        },
        "metrics": rows,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"alignment": alignment, "metrics": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
