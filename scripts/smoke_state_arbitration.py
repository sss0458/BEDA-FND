#!/usr/bin/env python3
"""CPU smoke test for the factorized four-state estimator."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

from model.beda_fnd import BEDAFNDModel  # noqa: E402


def state_arbitrator(mode: str) -> BEDAFNDModel:
    model = BEDAFNDModel.__new__(BEDAFNDModel)
    torch.nn.Module.__init__(model)
    model.is_beda = True
    model.output_mode = "full"
    model.state_mode = mode
    model.interaction_feature_state_calibrator = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 1),
    )
    model.interaction_decision_state_calibrator = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 1),
    )
    torch.nn.init.zeros_(model.interaction_feature_state_calibrator[-1].weight)
    torch.nn.init.zeros_(model.interaction_feature_state_calibrator[-1].bias)
    torch.nn.init.zeros_(model.interaction_decision_state_calibrator[-1].weight)
    torch.nn.init.zeros_(model.interaction_decision_state_calibrator[-1].bias)
    return model


def main() -> None:
    probabilities = torch.tensor(((0.1, 0.9, 0.5), (0.8, 0.7, 0.9)))
    clip_similarity = torch.tensor((0.2, -0.4))
    match_logit = torch.tensor((1.1, -0.7))
    states_by_mode = {}
    for mode in (
        "dual", "feature_only", "decision_only",
        "without_both_channels", "legacy",
    ):
        model = state_arbitrator(mode)
        states = model._interaction_state_weights(
            probabilities, clip_similarity, match_logit
        )
        if not torch.allclose(states.sum(dim=-1), torch.ones(2), atol=1e-6):
            raise AssertionError(f"{mode} memberships do not sum to one")
        if not bool(((states >= 0.0) & (states <= 1.0)).all()):
            raise AssertionError(f"{mode} memberships leave [0, 1]")
        states_by_mode[mode] = states
        print(mode, states.tolist())
    if not torch.equal(states_by_mode["dual"], states_by_mode["legacy"]):
        raise AssertionError("zero-residual dual arbitrator changed the warm start")
    expected_uniform = torch.full_like(
        states_by_mode["without_both_channels"], 0.25
    )
    if not torch.equal(
        states_by_mode["without_both_channels"], expected_uniform
    ):
        raise AssertionError("both-off state cues are not exactly uniform")


if __name__ == "__main__":
    main()
