#!/usr/bin/env python3
"""Check the real decision/output suffix without loading pretrained encoders."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def decision_forward():
    """Compile the unchanged suffix starting at the reference-probability sum."""
    source = Path(__file__).resolve().parents[1] / "src/model/beda_fnd.py"
    tree = ast.parse(source.read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BEDAFNDModel")
    forward = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    start = next(i for i, n in enumerate(forward.body) if (
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "fake_news_sigmoid" for t in n.targets)
        and isinstance(n.value, ast.BinOp)
    ))
    template = ast.parse('''
def decision(self, weight_common, branches):
    domain_aware_text_view, domain_aware_image_view, domain_aware_fusion_view = branches.unbind(-1)
    text_fake_news = image_fake_news = fusion_fake_news = None
    text_multi_domain = image_multi_domain = fusion_multi_domain = None
    match_positive_logit = branches[:, 0] * 0
    match_negative_logit = branches[:, 0] * 0
''')
    template.body[0].body.extend(forward.body[start:])
    ast.fix_missing_locations(template)
    namespace = {"torch": torch}
    exec(compile(template, str(source), "exec"), namespace)
    return namespace["decision"]


def main():
    decision = decision_forward()

    def forbidden(*args, **kwargs):
        raise AssertionError("Refinement must never run in reference-only mode")

    model = SimpleNamespace(
        is_beda=True, ablation="evidence_router", output_mode="reference_only",
        _interaction_moe_correction=forbidden,
        _evidence_logit_correction=forbidden, _correction_acceptance=forbidden,
    )
    logits = torch.tensor([[1.7, -0.5, 0.2], [-0.8, 1.3, 0.5]], requires_grad=True)
    branch_logits = torch.tensor([[-1.5, 1.2, 0.3], [1.4, -0.7, 0.6]], requires_grad=True)
    weights = logits.softmax(-1)
    branches = branch_logits.sigmoid()
    outputs = decision(model, weights, branches)
    expected = (weights[:, 0] * branches[:, 0] + weights[:, 1] * branches[:, 1]
                + weights[:, 2] * branches[:, 2]).clamp(1e-6, 1 - 1e-6)
    assert torch.equal(outputs[0], expected), "Reference p0 must be bitwise exact"
    assert len(outputs) == 13 and outputs[10] is weights
    assert not torch.allclose(outputs[0], branches.mean(-1)), "Refine removal is not equal weighting"
    assert all(p is outputs[0] for p in model.interaction_counterfactual_predictions)
    torch.nn.functional.binary_cross_entropy(outputs[0], torch.tensor([0., 1.])).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert branch_logits.grad is not None and branch_logits.grad.abs().sum() > 0

    # Exercise the retained numerical clamp and singleton batch contract.
    for value in (0., 1.):
        result = decision(model, torch.tensor([[0.7, 0.2, 0.1]]), torch.full((1, 3), value))
        assert torch.equal(result[0], torch.tensor([value]).clamp(1e-6, 1 - 1e-6))
    print("PASS: exact p0, retained attention/branch gradients, clamp, tuple and skipped refinement")


if __name__ == "__main__":
    main()
