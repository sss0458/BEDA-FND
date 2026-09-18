"""Reference-only DCEA intervention on the unchanged historical model graph."""
import types
import torch

VARIANTS = ('dcea_reference_only',)

def _reset_reference(model, inputs):
    model.audit_reference_probability = None
    model.audit_reference_weights = None

def _reference_only(self, base_probability, branch_probabilities, base_weights,
                    category, soft_domain, clip_similarity, analysis_mode):
    # The first invocation receives the original arithmetic reference p0.
    # Later invocations are historical counterfactual calls; none runs experts.
    if self.audit_reference_probability is None:
        self.audit_reference_probability = base_probability
        self.audit_reference_weights = base_weights
    return (torch.zeros_like(base_probability), base_weights,
            base_probability.new_zeros((base_probability.shape[0], 4)))

def _return_reference(model, inputs, outputs):
    assert model.audit_reference_probability is not None
    # Preserve p0 exactly, avoiding even the redundant sigmoid(logit(p0)).
    return (model.audit_reference_probability,) + outputs[1:]

def apply(model, variant):
    assert variant in VARIANTS and model.is_unified
    assert model.v20_signal_mode == 'off' and model.v20_dacg_mode == 'off'
    model.audit_variant = variant
    model.unified_output_mode = 'full'
    model.unified_dir_mode = 'off'
    model._interaction_moe_correction = types.MethodType(_reference_only, model)
    model.audit_handles = [model.register_forward_pre_hook(_reset_reference),
                           model.register_forward_hook(_return_reference)]
    return model
