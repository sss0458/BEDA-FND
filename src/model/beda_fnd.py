import os
import tqdm
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
from transformers import BertModel
import torch.nn as nn
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder, clipdata2gpu
from utils.utils import metricsTrueFalse
from .layers import *
from .pivot import *
from timm.models.vision_transformer import Block
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models




class DomainAwareTransformer(nn.Module):
    def __init__(self, dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.global_proj = nn.Linear(dim, dim)
        self.weight_proj = nn.Linear(dim, 3)
        
        # Add & Norm layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_reps, domain_rep):
        batch_size = domain_rep.shape[0]
        
        # global features
        global_rep = torch.mean(torch.stack(modality_reps, dim=1), dim=1)
        global_rep = self.global_proj(global_rep)
        
        #  domain features plus global features
        q = self.q_proj(domain_rep + global_rep).view(batch_size, self.num_heads, self.head_dim)
        
        # 
        k = torch.stack([self.k_proj(rep) for rep in modality_reps], dim=1)
        v = torch.stack([self.v_proj(rep) for rep in modality_reps], dim=1)
        
        k = k.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        
        # multihead
        attn = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).squeeze(2)
        out = out.reshape(batch_size, -1)

        # Add & Norm 
        domain_rep = domain_rep + self.dropout(out)
        domain_rep = self.norm1(domain_rep)
        
        # FFN
        ffn_output = self.ffn(domain_rep)
        
        # Add & Norm
        domain_rep = domain_rep + self.dropout(ffn_output)
        domain_rep = self.norm2(domain_rep)
        
        # final weights
        weights = self.weight_proj(domain_rep)
        weights = F.softmax(weights, dim=-1)
        
        return weights


class BEDAFNDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout,
                 ablation='beda_full', signed_gate_alpha=0.5,
                 tensor_domain_mode='full', pool_domain_mode='full',
                 signal_mode='off', acceptance_mode='off',
                 signal_domain_mode='full', bem_mode='full',
                 dcea_mode='full', correction_scale=1.0,
                 routing_family='shared', output_mode='full',
                 state_mode='dual', drcm_mode='full'):
        super(BEDAFNDModel, self).__init__()
        self.requested_ablation = ablation
        self.is_beda = ablation == 'beda_full'
        if self.is_beda:
            # The released model uses one computation graph on both datasets.
            ablation = 'evidence_router'
        self.num_expert = 6
        self.task_num = 2
        # self.domain_num = 9
        self.domain_num = self.task_num
        self.gate_num = 3
        self.ablation = ablation
        self.signed_gate_alpha = signed_gate_alpha
        self.tensor_domain_mode = tensor_domain_mode
        self.pool_domain_mode = pool_domain_mode
        self.signal_mode = signal_mode
        self.acceptance_mode = acceptance_mode
        self.signal_domain_mode = signal_domain_mode
        self.bem_mode = bem_mode
        self.drcm_mode = drcm_mode
        self.dcea_mode = dcea_mode
        if state_mode not in (
            'dual', 'feature_only', 'decision_only',
            'without_both_channels', 'legacy'
        ):
            raise ValueError(f'unsupported state_mode: {state_mode}')
        self.state_mode = state_mode
        self.correction_scale = correction_scale
        self.routing_family = routing_family
        self.output_mode = output_mode
        if drcm_mode not in ('full', 'off'):
            raise ValueError(f'unsupported drcm_mode: {drcm_mode}')
        self.branch_diagnostics_available = output_mode != 'plain_ffn_early'
        self.num_share = 1
        self.evidence_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.text_token_len = 197
        self.image_token_len = 197

        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))

            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
        self.text_experts = nn.ModuleList(text_expert_list)

        image_expert_list = []
        for i in range(self.domain_num):
            image_expert = []
            for j in range(self.num_expert):
                image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)
        self.image_experts = nn.ModuleList(image_expert_list)

        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 320),
                                       )
                fusion_expert.append(expert)
            fusion_expert = nn.ModuleList(fusion_expert)
            fusion_expert_list.append(fusion_expert)
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        final_expert_list = []
        for i in range(self.domain_num):
            final_expert = []
            for j in range(self.num_expert):
                final_expert.append(Block(dim=320, num_heads=8))
            final_expert = nn.ModuleList(final_expert)
            final_expert_list.append(final_expert)
        self.final_experts = nn.ModuleList(final_expert_list)

        text_share_expert, image_share_expert, fusion_share_expert, final_share_expert = [], [], [], []
        for i in range(self.num_share):
            text_share = []
            image_share = []
            fusion_share = []
            final_share = []
            for j in range(self.num_expert * 2):
                text_share.append(cnn_extractor(emb_dim, feature_kernel))
                image_share.append(cnn_extractor(self.image_dim, feature_kernel))
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 320),
                                       )
                fusion_share.append(expert)
                final_share.append(Block(dim=320, num_heads=8))
            text_share = nn.ModuleList(text_share)
            text_share_expert.append(text_share)
            image_share = nn.ModuleList(image_share)
            image_share_expert.append(image_share)
            fusion_share = nn.ModuleList(fusion_share)
            fusion_share_expert.append(fusion_share)
            final_share = nn.ModuleList(final_share)
            final_share_expert.append(final_share)
        self.text_share_expert = nn.ModuleList(text_share_expert)
        self.image_share_expert = nn.ModuleList(image_share_expert)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
        self.final_share_expert = nn.ModuleList(final_share_expert)

        image_gate_list, text_gate_list, fusion_gate_list, fusion_gate_list0, final_gate_list = [], [], [], [], []
        for i in range(self.domain_num):
            image_gate = nn.Sequential(nn.Linear(self.evidence_dim, self.evidence_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.evidence_dim, self.num_expert * 3),
                                       nn.Dropout(0.1),
                                       nn.Softmax(dim=1)
                                       )
            text_gate = nn.Sequential(nn.Linear(self.evidence_dim, self.evidence_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.evidence_dim, self.num_expert * 3),
                                      nn.Dropout(0.1),
                                      nn.Softmax(dim=1)
                                      )
            fusion_gate = nn.Sequential(nn.Linear(self.evidence_dim, self.evidence_dim),
                                        nn.SiLU(),
                                        nn.Linear(self.evidence_dim, self.num_expert * 4),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                        )
            fusion_gate0 = nn.Sequential(nn.Linear(320, 160),
                                         nn.SiLU(),
                                         nn.Linear(160, self.num_expert * 3),
                                         nn.Dropout(0.1),
                                         nn.Softmax(dim=1)
                                         )
            final_gate = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 160),
                                       nn.SiLU(),
                                       nn.Linear(160, self.num_expert * 3),
                                       nn.Dropout(0.1),
                                       nn.Softmax(dim=1)
                                       )
            image_gate_list.append(image_gate)
            text_gate_list.append(text_gate)
            fusion_gate_list.append(fusion_gate)
            fusion_gate_list0.append(fusion_gate0)
            final_gate_list.append(final_gate)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.fusion_gate_list = nn.ModuleList(fusion_gate_list)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
        self.final_gate_list = nn.ModuleList(final_gate_list)

        self.text_attention = MaskAttention(self.evidence_dim)
        self.image_attention = TokenAttention(self.evidence_dim)
        self.fusion_attention = TokenAttention(self.evidence_dim * 2)
        self.final_attention = TokenAttention(320)

        self.text_classifier = MLP(320, mlp_dims, dropout)
        self.text_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
        self.image_classifier = MLP(320, mlp_dims, dropout)
        self.image_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
        self.fusion_classifier = MLP(320, mlp_dims, dropout)
        self.fusion_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)

        self.max_classifier = MLP(320 * 1, mlp_dims, dropout)


        self.domain_aware_text_classifier = MLP(320 * 1, mlp_dims, dropout)
        self.domain_aware_image_classifier = MLP(320 * 1, mlp_dims, dropout)
        self.domain_aware_fusion_classifier = MLP(320 * 1, mlp_dims, dropout)

        self.domain_aware_total_classifier = MLP(320 * 1, mlp_dims, dropout)


        share_classifier_list = []

        for i in range(self.domain_num):
            share_classifier = MLP(320, mlp_dims, dropout)
            share_classifier_list.append(share_classifier)
        self.share_classifier_list = nn.ModuleList(share_classifier_list)

        dom_classifier_list = []

        for i in range(self.domain_num):
            dom_classifier = MLP(320, mlp_dims, dropout)
            dom_classifier_list.append(dom_classifier)
        self.dom_classifier_list = nn.ModuleList(dom_classifier_list)

        final_classifier_list = []

        for i in range(self.domain_num):
            final_classifier = MLP(320, mlp_dims, dropout)
            final_classifier_list.append(final_classifier)
        self.final_classifier_list = nn.ModuleList(final_classifier_list)

        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(768 * 2, 768, [348], 0.1)
        self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)
        if self.is_beda:
            # Structural baseline used by the large BEDA ablation.  It sees
            # only pooled raw text/image/cross-modal features and uses one
            # ordinary FFN; no multi-branch evidence heads or router output
            # are reused.
            self.plain_fusion_head = nn.Sequential(
                nn.LayerNorm(768 * 3 + 320),
                nn.Linear(768 * 3 + 320, 384),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(384, 1),
            )
        self.att_mlp_text = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_img = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_mm = MLP_fusion(320, 2, [174], 0.1)



        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        self.image_model.cuda()
        checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False


        self.ClipModel, _ = load_from_name("ViT-B-16", device="cuda", download_root='./')

        self.fake_news_layernorm = LayerNorm(320 * 3, eps=1e-12)
        self.domain_classification_layernorm = LayerNorm(320 * 1, eps=1e-12)
        self.gate_trans = nn.Sequential(
            nn.Linear(320 * 1, 3 * 320, bias=False),
            nn.GELU(),
            nn.Linear(3 * 320, 320 * 1, bias=False),
            nn.GELU(),
        )

        self.query_text = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 1, bias=False)
        )
        self.query_image = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 1, bias=False)
        )
        self.query_fusion = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )

        self.gate_fusion_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )

        # Start at the no_gate residual scale, then learn a bounded correction
        # that may either strengthen or weaken each domain-conditioned feature.
        if self.ablation == 'signed_gate':
            for gate in (
                self.gate_text_prefer,
                self.gate_image_prefer,
                self.gate_fusion_prefer,
            ):
                gate[-1] = nn.Tanh()
                nn.init.zeros_(gate[-2].weight)
                nn.init.zeros_(gate[-2].bias)

        self.attention = DomainAwareTransformer(dim=320, num_heads=8)

        # The specialized variant keeps the original three-view backbone, but
        # gives every view its own residual adapter and exposes the real domain
        # representation to both the view modulation and the decision router.
        # These modules are created only for the new variant so that pretrained
        # checkpoints remain strictly loadable for reproduction/ablation runs.
        if self.ablation in (
            'branch_evidence', 'identity_branch_evidence', 'calibrated_branch_evidence',
            'token_relation_evidence', 'domain_tensor_router',
            'logit_evidence_router',
            'uncertainty_evidence_router',
            'evidence_router',
        ):
            def residual_adapter():
                return nn.Sequential(
                    nn.Linear(320, 320),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(320, 320),
                )

            self.text_specializer = residual_adapter()
            self.image_specializer = residual_adapter()
            self.fusion_specializer = residual_adapter()
            self.text_domain_adapter = residual_adapter()
            self.image_domain_adapter = residual_adapter()
            self.fusion_domain_adapter = residual_adapter()
            if self.ablation in (
                'identity_branch_evidence', 'calibrated_branch_evidence',
                'token_relation_evidence', 'domain_tensor_router',
                'logit_evidence_router',
                'uncertainty_evidence_router',
                'evidence_router',
            ):
                self.text_view_norm = nn.Identity()
                self.image_view_norm = nn.Identity()
                self.fusion_view_norm = nn.Identity()
            else:
                self.text_view_norm = nn.LayerNorm(320)
                self.image_view_norm = nn.LayerNorm(320)
                self.fusion_view_norm = nn.LayerNorm(320)

            # |text-image| and text*image make cross-modal agreement explicit;
            # the CLIP pair representation supplies a second semantic signal.
            self.consistency_adapter = nn.Sequential(
                nn.Linear(960, 320),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(320, 320),
            )
            self.match_classifier = nn.Sequential(
                nn.Linear(960, 160),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(160, 1),
            )
            self.reliability_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(321, 160),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(160, 1),
                )
                for _ in range(3)
            ])
            if self.ablation in (
                'identity_branch_evidence', 'calibrated_branch_evidence',
                'token_relation_evidence', 'domain_tensor_router',
                'logit_evidence_router',
                'uncertainty_evidence_router',
                'evidence_router',
            ):
                self.domain_router_bias = nn.Sequential(
                    nn.Linear(320, 160),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(160, 3),
                )
                # Zero-output initialization makes warm-start exactly preserve the
                # warm-started discriminator and router before fine-tuning.
                zero_output_modules = [
                    self.text_specializer, self.image_specializer,
                    self.fusion_specializer, self.text_domain_adapter,
                    self.image_domain_adapter, self.fusion_domain_adapter,
                    self.consistency_adapter, self.domain_router_bias,
                    *self.reliability_heads,
                ]
                for module in zero_output_modules:
                    nn.init.zeros_(module[-1].weight)
                    if module[-1].bias is not None:
                        nn.init.zeros_(module[-1].bias)

            if self.ablation == 'token_relation_evidence':
                # Four shared queries independently collect token-level text
                # and image evidence.  Query-wise difference/product features
                # expose local agreement without replacing the frozen encoders.
                relation_dim = 160
                relation_queries = 4
                self.relation_queries = nn.Parameter(
                    torch.empty(relation_queries, relation_dim)
                )
                nn.init.normal_(self.relation_queries, std=0.02)
                self.relation_text_projection = nn.Sequential(
                    nn.LayerNorm(self.text_dim),
                    nn.Linear(self.text_dim, relation_dim),
                )
                self.relation_image_projection = nn.Sequential(
                    nn.LayerNorm(self.image_dim),
                    nn.Linear(self.image_dim, relation_dim),
                )
                self.relation_text_attention = nn.MultiheadAttention(
                    relation_dim, num_heads=4, dropout=dropout,
                    batch_first=True,
                )
                self.relation_image_attention = nn.MultiheadAttention(
                    relation_dim, num_heads=4, dropout=dropout,
                    batch_first=True,
                )
                self.relation_token_mlp = nn.Sequential(
                    nn.Linear(4 * relation_dim, 320),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(320, 320),
                )
                self.relation_pool = nn.Linear(320, 1)
                self.relation_output = nn.Sequential(
                    nn.LayerNorm(320),
                    nn.Linear(320, 320),
                )
                self.relation_match_classifier = nn.Sequential(
                    nn.LayerNorm(320),
                    nn.Linear(320, 160),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(160, 1),
                )
                # The first forward pass is exactly the warm-started
                # detector; only training can introduce the relation residual.
                nn.init.zeros_(self.relation_output[-1].weight)
                nn.init.zeros_(self.relation_output[-1].bias)

            if self.ablation == 'domain_tensor_router':
                # A compact CP/Tucker-style interaction models the joint
                # question "is this text-image relation plausible in this
                # domain?".  The frozen warm-start detector supplies three branch
                # features; the tensor block only learns a residual log-odds
                # correction, so its causal contribution is easy to ablate.
                tensor_rank = 96
                self.tensor_text_projection = nn.Sequential(
                    nn.LayerNorm(320),
                    nn.Linear(320, tensor_rank, bias=False),
                )
                self.tensor_image_projection = nn.Sequential(
                    nn.LayerNorm(320),
                    nn.Linear(320, tensor_rank, bias=False),
                )
                self.tensor_fusion_projection = nn.Sequential(
                    nn.LayerNorm(320),
                    nn.Linear(320, tensor_rank, bias=False),
                )
                # 9 explicit-domain coordinates + 9 softly inferred ones.
                # No bias: a zero domain context is a genuine no-domain case.
                self.tensor_domain_projection = nn.Linear(
                    18, tensor_rank, bias=False
                )
                self.tensor_interaction_norm = nn.LayerNorm(tensor_rank)
                self.tensor_correction = nn.Sequential(
                    nn.Linear(tensor_rank, 48),
                    nn.GELU(),
                    nn.Linear(48, 1),
                )
                # Exact warm-start equivalence.  Training can only add a
                # bounded correction after seeing labels from the train split.
                nn.init.zeros_(self.tensor_correction[-1].weight)
                nn.init.zeros_(self.tensor_correction[-1].bias)

            if self.ablation == 'calibrated_branch_evidence':
                # Unlike the legacy latent "domain feature", this path starts
                # from the explicitly supervised nine-domain probabilities.
                # FiLM supplies signed scale/shift residuals, so a domain can
                # suppress as well as amplify a branch.  Zero initialization
                # preserves the warm-start checkpoint exactly before fine-tuning.
                self.semantic_domain_encoder = nn.Sequential(
                    nn.Linear(9, 64),
                    nn.GELU(),
                    nn.Linear(64, 320),
                    nn.LayerNorm(320),
                )
                self.semantic_domain_film = nn.ModuleList([
                    nn.Linear(320, 640) for _ in range(3)
                ])
                self.semantic_domain_router = nn.Linear(320, 3)
                for module in (
                    *self.semantic_domain_film,
                    self.semantic_domain_router,
                ):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        # A compact, explicitly semantic alternative.  It consumes the known
        # news-domain metadata (never the truth label) and learns low-rank
        # per-domain feature scales.  The zero-initialized projection makes the
        # model exactly equivalent to no_gate before the first update.
        if self.ablation == 'explicit_domain_gate':
            self.explicit_domain_embedding = nn.Embedding(9, 32)
            self.explicit_domain_scale = nn.Linear(32, 3 * 320)
            nn.init.zeros_(self.explicit_domain_scale.weight)
            nn.init.zeros_(self.explicit_domain_scale.bias)

        if self.ablation == 'domain_log_pool':
            # A logarithmic opinion pool combines Bernoulli experts by adding
            # their log-odds.  A tiny explicit/soft-domain controller decides
            # how much the product-of-experts answer may correct the released
            # arithmetic probability pool.  Bias-free layers make the
            # no-domain intervention exactly zero.
            pool_hidden = 32
            self.pool_domain_encoder = nn.Linear(
                18, pool_hidden, bias=False
            )
            self.pool_weight_delta = nn.Linear(
                pool_hidden, 3, bias=False
            )
            self.pool_strength = nn.Linear(
                pool_hidden, 1, bias=False
            )
            self.pool_bias = nn.Linear(pool_hidden, 1, bias=False)
            for module in (
                self.pool_weight_delta,
                self.pool_strength,
                self.pool_bias,
            ):
                nn.init.zeros_(module.weight)

        if self.ablation in (
            'probability_evidence_router',
            'logit_evidence_router',
            'uncertainty_evidence_router',
            'probability_router_with_signal_control',
            'evidence_router',
        ):
            # Four deterministic interaction gates represent
            # agreement/alignment, agreement/misalignment,
            # disagreement/alignment and disagreement/misalignment.  Each
            # expert maps the explicit+inferred domain context to branch
            # reliability logits.  This is a compact hierarchical MoE: the
            # interaction state chooses an expert, while the domain decides
            # how that expert may reweight text/image/fusion evidence.
            interaction_hidden = 32
            self.interaction_domain_encoder = nn.Linear(
                18, interaction_hidden, bias=False
            )
            self.interaction_reliability_experts = nn.ModuleList([
                nn.Linear(interaction_hidden, 3, bias=False)
                for _ in range(4)
            ])
            self.interaction_strength_experts = nn.ModuleList([
                nn.Linear(interaction_hidden, 1, bias=False)
                for _ in range(4)
            ])
            self.interaction_bias_experts = nn.ModuleList([
                nn.Linear(interaction_hidden, 1, bias=False)
                for _ in range(4)
            ])
            # The feature channel calibrates vector-level image-text matching
            # evidence without consuming any branch decision.  It is a
            # residual around the stable CLIP prior; zero-output
            # initialization therefore preserves warm-start predictions.
            self.interaction_feature_state_calibrator = nn.Sequential(
                nn.Linear(2, 8),
                nn.GELU(),
                nn.Linear(8, 1),
            )
            self.interaction_decision_state_calibrator = nn.Sequential(
                nn.Linear(2, 8),
                nn.GELU(),
                nn.Linear(8, 1),
            )
            nn.init.zeros_(self.interaction_feature_state_calibrator[-1].weight)
            nn.init.zeros_(self.interaction_feature_state_calibrator[-1].bias)
            nn.init.zeros_(self.interaction_decision_state_calibrator[-1].weight)
            nn.init.zeros_(self.interaction_decision_state_calibrator[-1].bias)
            if self.ablation == 'uncertainty_evidence_router':
                # Domain-conditioned confidence calibration.  The absolute
                # Bernoulli log-odds are a compact evidential certainty score;
                # centering them across branches makes this term change only
                # relative reliability.  Bias-free, zero-initialized experts
                # retain exact warm-start behavior before training and when domain
                # information is removed.
                self.interaction_certainty_experts = nn.ModuleList([
                    nn.Linear(interaction_hidden, 3, bias=False)
                    for _ in range(4)
                ])
            # Exact warm start.  Unlike fixed domain pooling, the trained reliability logits
            # can span enough range to revive a branch whose released router
            # assigned almost zero probability.
            for module in (
                *self.interaction_reliability_experts,
                *self.interaction_strength_experts,
                *self.interaction_bias_experts,
            ):
                nn.init.zeros_(module.weight)
            if self.ablation == 'uncertainty_evidence_router':
                for module in self.interaction_certainty_experts:
                    nn.init.zeros_(module.weight)

        if self.ablation in (
            'probability_router_with_signal_control',
            'evidence_router',
        ):
            # signal-control changes signal processing before the three branch decisions.
            # The six domain-conditioned gains act on common and differential
            # text/image signals.  Zero initialization is an exact identity,
            # which lets a validated probability-pool/logit-pool checkpoint be loaded without an
            # architectural accuracy penalty.
            self.signal_domain_controller = nn.Linear(18, 6, bias=False)
            nn.init.zeros_(self.signal_domain_controller.weight)

            # The correction-acceptance gate observes domain, branch conflict,
            # uncertainty, alignment and correction size.  It can only attenuate
            # the already learned interaction correction.  A zero final layer
            # gives acceptance=1 exactly, so the warm-start prediction is kept.
            self.dacg_domain_encoder = nn.Linear(18, 16, bias=False)
            self.dacg_gate = nn.Sequential(
                nn.Linear(21, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            nn.init.zeros_(self.dacg_gate[-1].weight)
            nn.init.zeros_(self.dacg_gate[-1].bias)

            if self.is_beda:
                # CDSD has two explicit decision residuals.  The common head
                # sees shared confidence/alignment; the difference head sees
                # branch conflict/misalignment.  Their last layers start at
                # zero, so the complete unified model is exactly the backbone
                # before training and can be ablated without checkpoint drift.
                self.signal_common_correction = nn.Sequential(
                    nn.Linear(24, 16), nn.GELU(), nn.Linear(16, 1)
                )
                self.signal_difference_correction = nn.Sequential(
                    nn.Linear(26, 16), nn.GELU(), nn.Linear(16, 1)
                )
                for head in (
                    self.signal_common_correction,
                    self.signal_difference_correction,
                ):
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)

        self.tau = 0.5

    def _relation_query_feature(self, text_tokens, image_tokens, text_mask):
        text_tokens = self.relation_text_projection(text_tokens)
        image_tokens = self.relation_image_projection(image_tokens)
        queries = self.relation_queries.unsqueeze(0).expand(
            text_tokens.shape[0], -1, -1
        )
        text_evidence, _ = self.relation_text_attention(
            queries, text_tokens, text_tokens,
            key_padding_mask=~text_mask.bool(),
            need_weights=False,
        )
        image_evidence, _ = self.relation_image_attention(
            queries, image_tokens, image_tokens,
            need_weights=False,
        )
        relation_tokens = self.relation_token_mlp(torch.cat((
            text_evidence,
            image_evidence,
            torch.abs(text_evidence - image_evidence),
            text_evidence * image_evidence,
        ), dim=-1))
        relation_weights = torch.softmax(
            self.relation_pool(relation_tokens).squeeze(-1), dim=1
        )
        return torch.sum(
            relation_weights.unsqueeze(-1) * relation_tokens, dim=1
        )

    def _tensor_domain_context(
        self, category, soft_domain, analysis_mode
    ):
        hard_domain = F.one_hot(
            category.long(), num_classes=9
        ).to(dtype=soft_domain.dtype)
        mode = self.tensor_domain_mode
        if analysis_mode == 'no_tensor_domain':
            mode = 'none'
        elif analysis_mode == 'tensor_hard_only':
            mode = 'hard'
        elif analysis_mode == 'tensor_soft_only':
            mode = 'soft'

        if mode == 'none':
            hard_domain = torch.zeros_like(hard_domain)
            soft_domain = torch.zeros_like(soft_domain)
        elif mode == 'hard':
            soft_domain = torch.zeros_like(soft_domain)
        elif mode == 'soft':
            hard_domain = torch.zeros_like(hard_domain)
        elif mode != 'full':
            raise ValueError(f'unknown tensor_domain_mode: {mode}')

        context = torch.cat((hard_domain, soft_domain), dim=-1)
        if analysis_mode.startswith('shuffled_tensor_domain'):
            parts = analysis_mode.rsplit('_', 1)
            shift = int(parts[-1]) if parts[-1].isdigit() else 1
            context = torch.roll(context, shifts=shift, dims=0)
        return context

    def _domain_tensor_correction(
        self, text_feature, image_feature, fusion_feature,
        category, soft_domain, analysis_mode,
    ):
        text_factor = torch.tanh(
            self.tensor_text_projection(text_feature)
        )
        image_factor = torch.tanh(
            self.tensor_image_projection(image_feature)
        )
        fusion_factor = torch.tanh(
            self.tensor_fusion_projection(fusion_feature)
        )
        domain_context = self._tensor_domain_context(
            category, soft_domain, analysis_mode
        )
        domain_factor = torch.tanh(
            self.tensor_domain_projection(domain_context)
        )
        # Low-rank bilinear text-image interaction, modulated around identity
        # by the fusion branch and domain.  This avoids a full 320^4 tensor.
        interaction = (
            text_factor
            * image_factor
            * (1.0 + 0.5 * fusion_factor)
            * (1.0 + 0.5 * domain_factor)
        )
        interaction = self.tensor_interaction_norm(interaction)
        raw_correction = self.tensor_correction(interaction).squeeze(-1)
        return 2.0 * torch.tanh(raw_correction)

    def _pool_domain_context(
        self, category, soft_domain, analysis_mode
    ):
        hard_domain = F.one_hot(
            category.long(), num_classes=9
        ).to(dtype=soft_domain.dtype)
        mode = self.pool_domain_mode
        if analysis_mode == 'no_pool_domain':
            mode = 'none'
        elif analysis_mode == 'pool_hard_only':
            mode = 'hard'
        elif analysis_mode == 'pool_soft_only':
            mode = 'soft'
        if mode == 'none':
            hard_domain = torch.zeros_like(hard_domain)
            soft_domain = torch.zeros_like(soft_domain)
        elif mode == 'hard':
            soft_domain = torch.zeros_like(soft_domain)
        elif mode == 'soft':
            hard_domain = torch.zeros_like(hard_domain)
        elif mode != 'full':
            raise ValueError(f'unknown pool_domain_mode: {mode}')
        context = torch.cat((hard_domain, soft_domain), dim=-1)
        if analysis_mode.startswith('shuffled_pool_domain'):
            parts = analysis_mode.rsplit('_', 1)
            shift = int(parts[-1]) if parts[-1].isdigit() else 1
            context = torch.roll(context, shifts=shift, dims=0)
        return context

    def _domain_log_pool_correction(
        self, base_probability, branch_probabilities, base_weights,
        category, soft_domain, analysis_mode,
    ):
        context = self._pool_domain_context(
            category, soft_domain, analysis_mode
        )
        hidden = torch.tanh(self.pool_domain_encoder(context))
        weight_delta = 0.5 * torch.tanh(
            self.pool_weight_delta(hidden)
        )
        pool_weights = torch.softmax(
            torch.log(base_weights.clamp_min(1e-8)) + weight_delta,
            dim=-1,
        )
        branch_logits = torch.logit(
            branch_probabilities.clamp(1e-6, 1.0 - 1e-6)
        )
        pooled_logit = torch.sum(pool_weights * branch_logits, dim=-1)
        base_logit = torch.logit(
            base_probability.clamp(1e-6, 1.0 - 1e-6)
        )
        strength = 0.5 * torch.tanh(
            self.pool_strength(hidden).squeeze(-1)
        )
        domain_bias = 0.5 * torch.tanh(
            self.pool_bias(hidden).squeeze(-1)
        )
        correction = strength * (pooled_logit - base_logit) + domain_bias
        return correction, pool_weights

    def _interaction_domain_context(
        self, category, soft_domain, analysis_mode
    ):
        if self.is_beda and self.drcm_mode == 'off':
            analysis_mode = 'no_interaction_domain'
        translated_mode = analysis_mode
        if analysis_mode == 'no_interaction_domain':
            translated_mode = 'no_pool_domain'
        elif analysis_mode == 'interaction_hard_only':
            translated_mode = 'pool_hard_only'
        elif analysis_mode == 'interaction_soft_only':
            translated_mode = 'pool_soft_only'
        elif analysis_mode.startswith('shuffled_interaction_domain'):
            translated_mode = analysis_mode.replace(
                'shuffled_interaction_domain',
                'shuffled_pool_domain',
                1,
            )
        return self._pool_domain_context(
            category, soft_domain, translated_mode
        )

    def _signal_domain_context(
        self, category, soft_domain, analysis_mode, component
    ):
        translated_mode = analysis_mode
        if analysis_mode == f'no_{component}_domain':
            translated_mode = 'no_pool_domain'
        elif analysis_mode == f'{component}_hard_only':
            translated_mode = 'pool_hard_only'
        elif analysis_mode == f'{component}_soft_only':
            translated_mode = 'pool_soft_only'
        elif analysis_mode.startswith(f'shuffled_{component}_domain'):
            translated_mode = analysis_mode.replace(
                f'shuffled_{component}_domain',
                'shuffled_pool_domain',
                1,
            )
        context = self._pool_domain_context(
            category, soft_domain, translated_mode
        )
        if self.signal_domain_mode == 'none':
            context = torch.zeros_like(context)
        elif self.signal_domain_mode == 'shuffled':
            context = torch.roll(context, shifts=1, dims=0)
        elif self.signal_domain_mode != 'full':
            raise ValueError(
                f'unknown signal_domain_mode: {self.signal_domain_mode}'
            )
        return context

    def _signal_decomposition(
        self, text_feature, image_feature, fusion_feature,
        category, soft_domain, analysis_mode,
    ):
        context = self._signal_domain_context(
            category, soft_domain, analysis_mode, 'signal'
        )
        gains = 0.25 * torch.tanh(
            self.signal_domain_controller(context)
        )
        if self.signal_mode == 'off' or analysis_mode == 'no_signal':
            gains = torch.zeros_like(gains)
        elif self.signal_mode == 'common_only':
            gains = gains * gains.new_tensor(
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
            )
        elif self.signal_mode == 'difference_only':
            gains = gains * gains.new_tensor(
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
            )
        elif self.signal_mode != 'full':
            raise ValueError(
                f'unknown signal_mode: {self.signal_mode}'
            )

        common = F.layer_norm(
            0.5 * (text_feature + image_feature),
            (text_feature.shape[-1],),
        )
        text_residual = F.layer_norm(
            text_feature - 0.5 * (text_feature + image_feature),
            (text_feature.shape[-1],),
        )
        image_residual = F.layer_norm(
            image_feature - 0.5 * (text_feature + image_feature),
            (image_feature.shape[-1],),
        )
        discrepancy = F.layer_norm(
            torch.abs(text_feature - image_feature),
            (text_feature.shape[-1],),
        )
        text_feature = (
            text_feature
            + gains[:, 0:1] * common
            + gains[:, 1:2] * text_residual
        )
        image_feature = (
            image_feature
            + gains[:, 2:3] * common
            + gains[:, 3:4] * image_residual
        )
        fusion_feature = (
            fusion_feature
            + gains[:, 4:5] * common
            + gains[:, 5:6] * discrepancy
        )
        self.last_signal_gains = gains
        return text_feature, image_feature, fusion_feature

    def _correction_acceptance(
        self, base_probability, branch_probabilities, correction,
        category, soft_domain, clip_similarity, analysis_mode,
    ):
        context = self._signal_domain_context(
            category, soft_domain, analysis_mode, 'dacg'
        )
        domain_feature = torch.tanh(
            self.dacg_domain_encoder(context)
        )
        conflict = (
            branch_probabilities.max(dim=-1).values
            - branch_probabilities.min(dim=-1).values
        )
        uncertainty = (
            1.0 - torch.abs(2.0 * branch_probabilities - 1.0)
        ).mean(dim=-1)
        alignment = ((clip_similarity + 1.0) / 2.0).clamp(0.0, 1.0)
        base_uncertainty = (
            1.0 - torch.abs(2.0 * base_probability - 1.0)
        )
        correction_size = torch.tanh(torch.abs(correction))
        evidence = torch.stack((
            conflict,
            uncertainty,
            alignment,
            base_uncertainty,
            correction_size,
        ), dim=-1)
        raw_acceptance = self.dacg_gate(torch.cat((
            domain_feature, evidence
        ), dim=-1)).squeeze(-1)
        self.last_dacg_raw_acceptance = raw_acceptance

        # One-sided bounded residual: the new gate may reject an unsafe
        # correction, but cannot amplify it beyond the validated warm start.
        acceptance = torch.clamp(
            1.0 + torch.tanh(raw_acceptance), min=0.0, max=1.0
        )
        if self.acceptance_mode == 'off' or analysis_mode == 'no_dacg':
            acceptance = torch.ones_like(acceptance)
        self.last_dacg_acceptance = acceptance
        return acceptance

    def _evidence_logit_correction(
        self, base_probability, branch_probabilities, category,
        soft_domain, clip_similarity, analysis_mode,
    ):
        context = self._signal_domain_context(
            category, soft_domain, analysis_mode, 'signal'
        )
        gains = self.last_signal_gains
        alignment = ((clip_similarity + 1.0) / 2.0).clamp(0.0, 1.0)
        branch_uncertainty = (
            1.0 - torch.abs(2.0 * branch_probabilities - 1.0)
        ).mean(dim=-1)
        common_input = torch.cat((
            context,
            gains[:, (0, 2, 4)],
            branch_probabilities.mean(dim=-1, keepdim=True),
            branch_uncertainty.unsqueeze(-1),
            alignment.unsqueeze(-1),
        ), dim=-1)
        pairwise_difference = torch.stack((
            torch.abs(branch_probabilities[:, 0] - branch_probabilities[:, 1]),
            torch.abs(branch_probabilities[:, 0] - branch_probabilities[:, 2]),
            torch.abs(branch_probabilities[:, 1] - branch_probabilities[:, 2]),
        ), dim=-1)
        conflict = (
            branch_probabilities.max(dim=-1).values
            - branch_probabilities.min(dim=-1).values
        )
        difference_input = torch.cat((
            context,
            gains[:, (1, 3, 5)],
            pairwise_difference,
            conflict.unsqueeze(-1),
            (1.0 - alignment).unsqueeze(-1),
        ), dim=-1)
        common_correction = self.correction_scale * torch.tanh(
            self.signal_common_correction(common_input).squeeze(-1)
        )
        difference_correction = self.correction_scale * torch.tanh(
            self.signal_difference_correction(difference_input).squeeze(-1)
        )
        if self.signal_mode == 'off' or analysis_mode == 'no_signal':
            common_correction = torch.zeros_like(base_probability)
            difference_correction = torch.zeros_like(base_probability)
        elif self.signal_mode == 'common_only':
            difference_correction = torch.zeros_like(base_probability)
        elif self.signal_mode == 'difference_only':
            common_correction = torch.zeros_like(base_probability)
        self.last_signal_common_correction = common_correction
        self.last_signal_difference_correction = difference_correction
        return common_correction + difference_correction

    def _interaction_state_weights(
        self, branch_probabilities, clip_similarity,
        match_positive_logit=None,
    ):
        """Factor feature compatibility and decision agreement into 4 states."""
        if self.is_beda and self.output_mode == 'shared_evidence':
            # The shared-evidence intervention has no active BEM matcher.
            match_positive_logit = None
        legacy_agreement = (
            1.0
            - torch.abs(
                branch_probabilities[:, 0]
                - branch_probabilities[:, 1]
            )
        ).clamp(0.0, 1.0)
        clip_compatibility = (
            (clip_similarity + 1.0) / 2.0
        ).clamp(0.0, 1.0)

        # Decision channel: all three branch outcomes contribute.  The sum
        # of the three pairwise distances is at most two for probabilities in
        # [0, 1], so this remains a continuous quantity in [0, 1].
        pairwise_distance = (
            torch.abs(branch_probabilities[:, 0] - branch_probabilities[:, 1])
            + torch.abs(branch_probabilities[:, 0] - branch_probabilities[:, 2])
            + torch.abs(branch_probabilities[:, 1] - branch_probabilities[:, 2])
        )
        tri_branch_agreement = (
            1.0 - pairwise_distance / 2.0
        ).clamp(0.0, 1.0)
        decision_input = torch.stack((
            tri_branch_agreement.detach(),
            legacy_agreement.detach(),
        ), dim=-1)
        decision_residual = 0.5 * torch.tanh(
            self.interaction_decision_state_calibrator(
                decision_input
            ).squeeze(-1)
        )
        decision_agreement = (
            legacy_agreement
            + decision_residual
            * 4.0 * legacy_agreement * (1.0 - legacy_agreement)
        ).clamp(0.0, 1.0)

        # Feature channel: the supervised vector-matching logit refines the
        # CLIP compatibility prior.  Detaching the matching logit keeps the
        # feature and decision objectives modular; its own matching loss still
        # trains the underlying vector relation model.
        feature_compatibility = clip_compatibility
        if match_positive_logit is not None:
            feature_input = torch.stack((
                torch.tanh(match_positive_logit.detach()),
                clip_similarity.clamp(-1.0, 1.0),
            ), dim=-1)
            feature_residual = 0.5 * torch.tanh(
                self.interaction_feature_state_calibrator(
                    feature_input
                ).squeeze(-1)
            )
            feature_compatibility = (
                clip_compatibility
                + feature_residual
                * 4.0 * clip_compatibility * (1.0 - clip_compatibility)
            ).clamp(0.0, 1.0)

        if self.state_mode == 'legacy':
            agreement = legacy_agreement
            compatibility = clip_compatibility
        elif self.state_mode == 'feature_only':
            agreement = torch.full_like(decision_agreement, 0.5)
            compatibility = feature_compatibility
        elif self.state_mode == 'decision_only':
            agreement = decision_agreement
            compatibility = torch.full_like(feature_compatibility, 0.5)
        elif self.state_mode == 'without_both_channels':
            # Remove both internal state cues while retaining the surrounding
            # DCEA computation.  The four interaction states consequently
            # receive equal mass, which is the clean 2x2 ablation corner.
            agreement = torch.full_like(decision_agreement, 0.5)
            compatibility = torch.full_like(feature_compatibility, 0.5)
        else:
            agreement = decision_agreement
            compatibility = feature_compatibility

        disagreement = 1.0 - agreement
        incompatibility = 1.0 - compatibility
        state_weights = torch.stack((
            agreement * compatibility,
            agreement * incompatibility,
            disagreement * compatibility,
            disagreement * incompatibility,
        ), dim=-1)
        self.last_feature_compatibility = feature_compatibility
        self.last_decision_agreement = decision_agreement
        return state_weights

    def _interaction_moe_correction(
        self, base_probability, branch_probabilities, base_weights,
        category, soft_domain, clip_similarity, analysis_mode,
        match_positive_logit=None,
    ):
        context = self._interaction_domain_context(
            category, soft_domain, analysis_mode
        )
        hidden = torch.tanh(self.interaction_domain_encoder(context))
        state_weights = self._interaction_state_weights(
            branch_probabilities, clip_similarity, match_positive_logit
        )
        reliability_by_state = torch.stack([
            expert(hidden)
            for expert in self.interaction_reliability_experts
        ], dim=1)
        reliability_delta = torch.sum(
            state_weights.unsqueeze(-1) * reliability_by_state,
            dim=1,
        )
        # log(400) is about 6.0.  A range of +/-8 is intentionally wide
        # enough to recover a useful image expert from the released 0.2%
        # routing weight, while tanh keeps the correction finite.
        reliability_delta = 8.0 * torch.tanh(reliability_delta)
        interaction_weights = torch.softmax(
            torch.log(base_weights.clamp_min(1e-8)) + reliability_delta,
            dim=-1,
        )
        branch_logits = torch.logit(
            branch_probabilities.clamp(1e-6, 1.0 - 1e-6)
        )
        evidential_correction = torch.zeros_like(base_probability)
        if hasattr(self, 'interaction_certainty_experts'):
            certainty_by_state = torch.stack([
                expert(hidden)
                for expert in self.interaction_certainty_experts
            ], dim=1)
            certainty_scale = torch.sum(
                state_weights.unsqueeze(-1) * certainty_by_state,
                dim=1,
            )
            branch_certainty = torch.tanh(
                torch.abs(branch_logits) / 2.0
            )
            centered_certainty = (
                branch_certainty
                - branch_certainty.mean(dim=-1, keepdim=True)
            )
            reliability_delta = (
                reliability_delta
                + 4.0 * torch.tanh(certainty_scale)
                * centered_certainty
            )
            interaction_weights = torch.softmax(
                torch.log(base_weights.clamp_min(1e-8))
                + reliability_delta,
                dim=-1,
            )
            # Domain/state-specific vector scaling of branch log-odds.  This
            # is an evidential calibration residual: confident evidence may
            # be trusted in one domain and discounted in another.  It has a
            # direct gradient even while the opinion-pool strength is still
            # zero, and remains exactly zero without a domain context.
            evidential_correction = 0.75 * torch.tanh(torch.sum(
                torch.tanh(certainty_scale)
                * torch.tanh(branch_logits / 2.0),
                dim=-1,
            ))
        pooled_logit = torch.sum(
            interaction_weights * branch_logits, dim=-1
        )
        base_logit = torch.logit(
            base_probability.clamp(1e-6, 1.0 - 1e-6)
        )
        strength_by_state = torch.cat([
            expert(hidden)
            for expert in self.interaction_strength_experts
        ], dim=-1)
        bias_by_state = torch.cat([
            expert(hidden)
            for expert in self.interaction_bias_experts
        ], dim=-1)
        strength = 1.5 * torch.tanh(torch.sum(
            state_weights * strength_by_state, dim=-1
        ))
        domain_bias = torch.tanh(torch.sum(
            state_weights * bias_by_state, dim=-1
        ))
        correction = (
            strength * (pooled_logit - base_logit)
            + domain_bias
            + evidential_correction
        )
        return correction, interaction_weights, state_weights

    @staticmethod
    def _same_domain_hard_negative(
        clip_text_feature, clip_image_feature, category
    ):
        """Choose the most similar non-paired image, preferring same domain."""
        with torch.no_grad():
            similarity = clip_text_feature @ clip_image_feature.transpose(0, 1)
            batch_size = similarity.shape[0]
            not_self = ~torch.eye(
                batch_size, dtype=torch.bool, device=similarity.device
            )
            same_domain = category.reshape(-1, 1).eq(
                category.reshape(1, -1)
            ) & not_self
            has_same_domain = same_domain.any(dim=1, keepdim=True)
            candidates = torch.where(
                has_same_domain, same_domain, not_self
            )
            mask_value = torch.finfo(similarity.dtype).min
            return similarity.masked_fill(
                ~candidates, mask_value
            ).argmax(dim=1)

    def forward(self, **kwargs):
        analysis_mode = kwargs.get('analysis_mode', 'normal')
        if self.is_beda and self.drcm_mode == 'off':
            # Preserve semantic experts/allocation; remove the domain inputs
            # consumed by BEM. DCEA context is cleared separately above.
            analysis_mode = 'no_domain'
        if self.is_beda and self.output_mode == 'no_arbitration':
            # This is a training-time structural ablation, not a test-time
            # weight replacement: domain inputs are removed everywhere that
            # consumes the common analysis mode.
            analysis_mode = 'no_domain'
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        text_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])
        image = kwargs['image']
        image_feature = self.image_model.forward_ying(image)  # ([64, 197, 768])
        clip_image = kwargs['clip_image']
        clip_text = kwargs['clip_text']
        with torch.no_grad():
            clip_image_feature = self.ClipModel.encode_image(clip_image)  # ([64, 512])
            clip_text_feature = self.ClipModel.encode_text(clip_text)  # ([64, 512])
            clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
            clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)
        clip_similarity = torch.sum(
            clip_image_feature * clip_text_feature, dim=-1
        ).float()
        clip_fusion_feature = torch.cat((clip_image_feature, clip_text_feature), dim=-1)  # torch.Size([64, 1024])
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())  # torch.Size([64, 320])

        text_atn_feature = self.text_attention(text_feature, masks)
        image_atn_feature, _ = self.image_attention(image_feature)
        fusion_feature = torch.cat((image_feature, text_feature), dim=-1)
        fusion_atn_feature, _ = self.fusion_attention(fusion_feature)  # ([64, 1536])
        fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature)
        plain_fusion_probability = None
        if self.is_beda and self.output_mode in ('plain_ffn', 'plain_ffn_early'):
            plain_fusion_input = torch.cat((
                text_atn_feature,
                image_atn_feature,
                fusion_atn_feature,
                clip_fusion_feature,
            ), dim=-1)
            plain_fusion_probability = torch.sigmoid(
                self.plain_fusion_head(plain_fusion_input).squeeze(-1)
            )
            if self.output_mode == 'plain_ffn_early':
                # Match the archived all-off intervention: stop before any
                # expert, domain, branch-evidence or arbitration module runs.
                self.last_plain_fusion_input = plain_fusion_input.detach()
                n = plain_fusion_probability.shape[0]
                domain = plain_fusion_probability.new_zeros((n, 9))
                auxiliary = plain_fusion_probability.new_full((n, 1), 0.5)
                view = plain_fusion_probability.new_full((n,), 0.5)
                weights = plain_fusion_probability.new_full((n, 3), 1.0 / 3.0)
                match = plain_fusion_probability.new_zeros(n)
                self.last_signal_gains = plain_fusion_probability.new_zeros((n, 6))
                self.last_dacg_acceptance = plain_fusion_probability.new_ones(n)
                # Detached constants only maintain the Trainer tuple contract;
                # they are not branch predictions or routing diagnostics.
                return (
                    plain_fusion_probability, auxiliary, domain, auxiliary,
                    domain, auxiliary, domain, view, view, view, weights, match, match,
                )

        text_gate_input = text_atn_feature  # ([64, 1536])
        image_gate_input = image_atn_feature
        fusion_gate_input = fusion_atn_feature


        # Multi-view Features Extraction and Aggregation

        text_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list[i](text_gate_input)
            text_gate_out_list.append(gate_out)
        self.text_gate_out_list = text_gate_out_list

        image_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.image_gate_list[i](image_gate_input)
            image_gate_out_list.append(gate_out)
        self.image_gate_out_list = image_gate_out_list

        fusion_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.fusion_gate_list[i](fusion_gate_input)
            fusion_gate_out_list.append(gate_out)
        self.fusion_gate_out_list = fusion_gate_out_list

        text_gate_expert_value = []
        text_experts_feature = 0
        text_gate_share_expert_value = []
        for i in range(1):
            gate_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.text_experts[i][j](text_feature)  # ([64, 320])
                gate_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert * 2):
                tmp_expert = self.text_share_expert[0][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
            text_experts_feature = gate_expert
            text_gate_share_expert_value.append(gate_share_expert)

        att = F.softmax(self.att_mlp_text(text_experts_feature), dim=-1)
        text_experts_feature0 = att[:, 0].view(-1, 1)*text_experts_feature
        text_experts_feature1 = att[:, 1].view(-1, 1)*text_experts_feature
        text_gate_expert_value.append(text_experts_feature0)
        text_gate_expert_value.append(text_experts_feature1)


        image_gate_expert_value = []
        image_experts_feature = 0
        image_gate_share_expert_value = []
        for i in range(1):
            gate_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.image_experts[i][j](image_feature)  # ([64, 320])
                gate_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert * 2):
                tmp_expert = self.image_share_expert[0][j](image_feature)
                gate_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
            image_experts_feature = gate_expert
            image_gate_share_expert_value.append(gate_share_expert)

        att = F.softmax(self.att_mlp_img(image_experts_feature), dim=-1)
        image_experts_feature0 = att[:, 0].view(-1, 1)*image_experts_feature
        image_experts_feature1 = att[:, 1].view(-1, 1)*image_experts_feature
        image_gate_expert_value.append(image_experts_feature0)
        image_gate_expert_value.append(image_experts_feature1)


        # clip_fusion_feature
        # fusion

        text = text_gate_share_expert_value[0]
        image = image_gate_share_expert_value[0]
        fusion_share_feature = torch.cat((clip_fusion_feature, text, image), dim=-1)

        fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        fusion_gate_input0 = self.domain_fusion(fusion_share_feature)
        fusion_gate_out_list0 = []
        for k in range(self.domain_num):
            gate_out = self.fusion_gate_list0[k](fusion_gate_input0)
            fusion_gate_out_list0.append(gate_out)
        self.fusion_gate_out_list0 = fusion_gate_out_list0

        fusion_gate_expert_value0 = []
        fusion_experts_feature = 0
        fusion_gate_share_expert_value0 = []
        for m in range(1):
            share_gate_expert0 = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for n in range(self.num_expert):
                fusion_tmp_expert0 = self.fusion_experts[m][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
            for n in range(self.num_expert * 2):
                fusion_tmp_expert0 = self.fusion_share_expert[0][n](fusion_share_feature)
                share_gate_expert0 += (
                            fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
                gate_share_expert += (
                            fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
            #fusion_gate_expert_value0.append(share_gate_expert0)
            fusion_gate_share_expert_value0.append(gate_share_expert)
            # The released implementation accidentally kept only the final
            # shared expert here.  The specialized variant uses the gated sum.
            fusion_experts_feature = (
                share_gate_expert0
                if self.ablation == 'branch_evidence'
                else fusion_tmp_expert0
            )

        att = F.softmax(self.att_mlp_mm(fusion_experts_feature), dim=-1)
        fusion_experts_feature0 = att[:, 0].view(-1, 1)*fusion_experts_feature
        fusion_experts_feature1 = att[:, 1].view(-1, 1)*fusion_experts_feature
        fusion_gate_expert_value0.append(fusion_experts_feature0)
        fusion_gate_expert_value0.append(fusion_experts_feature1)


        # text
        text_two_task = []
        image_two_task = []
        fusion_two_task = []
        text_two_task.append(self.text_classifier(text_gate_expert_value[0]).squeeze(1))
        text_two_task.append(self.text_classifier_Mu(text_gate_expert_value[1]).squeeze(1))
        image_two_task.append(self.image_classifier(image_gate_expert_value[0]).squeeze(1))
        image_two_task.append(self.image_classifier_Mu(image_gate_expert_value[1]).squeeze(1))
        fusion_two_task.append(self.fusion_classifier(fusion_gate_expert_value0[0]).squeeze(1))
        fusion_two_task.append(self.fusion_classifier_Mu(fusion_gate_expert_value0[1]).squeeze(1))

        
        
        # Domain Disentanglement
        if self.ablation in (
            'branch_evidence', 'identity_branch_evidence', 'calibrated_branch_evidence',
            'token_relation_evidence', 'domain_tensor_router',
            'logit_evidence_router',
            'uncertainty_evidence_router',
            'evidence_router',
        ):
            # Keep logits for the multi-label domain objective.  Applying
            # softmax before BCEWithLogits (the released path) double-activates
            # the values and also prevents two-domain samples from fitting.
            text_fake_news = torch.sigmoid(text_two_task[0])
            image_fake_news = torch.sigmoid(image_two_task[0])
            fusion_fake_news = torch.sigmoid(fusion_two_task[0])
            text_multi_domain = text_two_task[1]
            image_multi_domain = image_two_task[1]
            fusion_multi_domain = fusion_two_task[1]
        else:
            text_fake_news = torch.softmax(text_two_task[0],-1)
            image_fake_news = torch.softmax(image_two_task[0],-1)
            fusion_fake_news = torch.softmax(fusion_two_task[0],-1)
            text_multi_domain = torch.softmax(text_two_task[1], -1) # [64, 9]
            image_multi_domain = torch.softmax(image_two_task[1], -1)
            fusion_multi_domain = torch.softmax(fusion_two_task[1], -1)


        if self.ablation in (
            'probability_router_with_signal_control',
            'evidence_router',
        ):
            if self.ablation == 'evidence_router':
                signal_soft_domain = (
                    torch.sigmoid(text_multi_domain)
                    + torch.sigmoid(image_multi_domain)
                    + torch.sigmoid(fusion_multi_domain)
                ).detach() / 3.0
            else:
                signal_soft_domain = (
                    text_multi_domain
                    + image_multi_domain
                    + fusion_multi_domain
                ).detach() / 3.0
            (
                text_gate_expert_value[0],
                image_gate_expert_value[0],
                fusion_gate_expert_value0[0],
            ) = self._signal_decomposition(
                text_gate_expert_value[0],
                image_gate_expert_value[0],
                fusion_gate_expert_value0[0],
                kwargs['category'],
                signal_soft_domain,
                analysis_mode,
            )

        multi_label_feature = text_gate_expert_value[0] + image_gate_expert_value[0] + fusion_gate_expert_value0[0]
        fake_news_feature = text_gate_expert_value[1] + image_gate_expert_value[1] + fusion_gate_expert_value0[1]


        # Domain-Aware Multi-View Discriminator
        match_positive_logit = None
        match_negative_logit = None
        use_branch_evidence = self.ablation in (
            'branch_evidence', 'identity_branch_evidence', 'calibrated_branch_evidence',
            'token_relation_evidence', 'domain_tensor_router',
            'logit_evidence_router',
            'uncertainty_evidence_router',
            'evidence_router',
        ) and not (
            self.is_beda and self.bem_mode == 'off'
        )
        if use_branch_evidence:
            text_news_feature = text_gate_expert_value[0]
            image_news_feature = image_gate_expert_value[0]
            fusion_news_feature = fusion_gate_expert_value0[0]
            text_domain_feature = text_gate_expert_value[1]
            image_domain_feature = image_gate_expert_value[1]
            fusion_domain_feature = fusion_gate_expert_value0[1]
            if analysis_mode == 'no_domain':
                text_domain_feature = torch.zeros_like(text_domain_feature)
                image_domain_feature = torch.zeros_like(image_domain_feature)
                fusion_domain_feature = torch.zeros_like(fusion_domain_feature)
            elif analysis_mode.startswith('shuffled_domain'):
                parts = analysis_mode.rsplit('_', 1)
                shift = int(parts[-1]) if parts[-1].isdigit() else 1
                text_domain_feature = torch.roll(
                    text_domain_feature, shifts=shift, dims=0
                )
                image_domain_feature = torch.roll(
                    image_domain_feature, shifts=shift, dims=0
                )
                fusion_domain_feature = torch.roll(
                    fusion_domain_feature, shifts=shift, dims=0
                )
            domain_feature = (
                text_domain_feature + image_domain_feature + fusion_domain_feature
            )

            if self.ablation in (
                'identity_branch_evidence', 'calibrated_branch_evidence',
                'token_relation_evidence', 'domain_tensor_router',
                'logit_evidence_router',
                'uncertainty_evidence_router',
                'evidence_router',
            ):
                text_base_feature = (
                    text_news_feature
                    + self.gate_text_prefer(domain_feature) * text_news_feature
                )
                image_base_feature = (
                    image_news_feature
                    + self.gate_image_prefer(domain_feature) * image_news_feature
                )
                text_view_feature = (
                    text_base_feature
                    + self.text_specializer(text_base_feature)
                    + self.gate_text_prefer(domain_feature)
                    * self.text_domain_adapter(text_domain_feature)
                )
                image_view_feature = (
                    image_base_feature
                    + self.image_specializer(image_base_feature)
                    + self.gate_image_prefer(domain_feature)
                    * self.image_domain_adapter(image_domain_feature)
                )
            else:
                text_view_feature = self.text_view_norm(
                    text_news_feature
                    + self.text_specializer(text_news_feature)
                    + self.gate_text_prefer(domain_feature)
                    * self.text_domain_adapter(text_domain_feature)
                )
                image_view_feature = self.image_view_norm(
                    image_news_feature
                    + self.image_specializer(image_news_feature)
                    + self.gate_image_prefer(domain_feature)
                    * self.image_domain_adapter(image_domain_feature)
                )

            pair_feature = torch.cat((
                torch.abs(text_news_feature - image_news_feature),
                text_news_feature * image_news_feature,
                clip_fusion_feature,
            ), dim=-1)
            consistency_feature = self.consistency_adapter(pair_feature)
            if analysis_mode == 'no_consistency':
                consistency_feature = torch.zeros_like(consistency_feature)
            if self.ablation in (
                'identity_branch_evidence', 'calibrated_branch_evidence',
                'token_relation_evidence', 'domain_tensor_router',
                'logit_evidence_router',
                'uncertainty_evidence_router',
                'evidence_router',
            ):
                fusion_base_feature = (
                    fusion_news_feature
                    + self.gate_fusion_prefer(domain_feature)
                    * fusion_news_feature
                )
                fusion_view_feature = (
                    fusion_base_feature
                    + self.fusion_specializer(fusion_base_feature)
                    + consistency_feature
                    + self.gate_fusion_prefer(domain_feature)
                    * self.fusion_domain_adapter(fusion_domain_feature)
                )
            else:
                fusion_view_feature = self.fusion_view_norm(
                    fusion_news_feature
                    + self.fusion_specializer(fusion_news_feature)
                    + consistency_feature
                    + self.gate_fusion_prefer(domain_feature)
                    * self.fusion_domain_adapter(fusion_domain_feature)
                )

            relation_feature = None
            if self.ablation == 'token_relation_evidence':
                relation_feature = self._relation_query_feature(
                    text_feature, image_feature, masks
                )
                if analysis_mode == 'no_relation_queries':
                    relation_residual = torch.zeros_like(relation_feature)
                else:
                    relation_residual = self.relation_output(relation_feature)
                fusion_view_feature = fusion_view_feature + relation_residual

            semantic_router_bias = 0.0
            if self.ablation == 'calibrated_branch_evidence':
                semantic_distribution = (
                    torch.sigmoid(text_multi_domain)
                    + torch.sigmoid(image_multi_domain)
                    + torch.sigmoid(fusion_multi_domain)
                ).detach() / 3.0
                use_semantic_domain = analysis_mode != 'no_semantic_domain'
                if analysis_mode.startswith('shuffled_semantic_domain'):
                    parts = analysis_mode.rsplit('_', 1)
                    shift = int(parts[-1]) if parts[-1].isdigit() else 1
                    semantic_distribution = torch.roll(
                        semantic_distribution, shifts=shift, dims=0
                    )
                semantic_feature = self.semantic_domain_encoder(
                    semantic_distribution
                )
                if use_semantic_domain:
                    semantic_residuals = []
                    for film, news_feature in zip(
                        self.semantic_domain_film,
                        (text_news_feature, image_news_feature, fusion_news_feature),
                    ):
                        scale, shift = film(semantic_feature).chunk(2, dim=-1)
                        semantic_residuals.append(
                            0.25 * torch.tanh(scale)
                            * F.layer_norm(news_feature, (news_feature.shape[-1],))
                            + 0.25 * torch.tanh(shift)
                        )
                    text_view_feature = text_view_feature + semantic_residuals[0]
                    image_view_feature = image_view_feature + semantic_residuals[1]
                    fusion_view_feature = fusion_view_feature + semantic_residuals[2]
                    semantic_router_bias = self.semantic_domain_router(
                        semantic_feature
                    )

            domain_aware_text_view = torch.sigmoid(
                self.domain_aware_text_classifier(text_view_feature).squeeze(-1)
            )
            domain_aware_image_view = torch.sigmoid(
                self.domain_aware_image_classifier(image_view_feature).squeeze(-1)
            )
            domain_aware_fusion_view = torch.sigmoid(
                self.domain_aware_fusion_classifier(fusion_view_feature).squeeze(-1)
            )

            # Paired samples are positives; a cyclically shifted image is a
            # cheap in-batch negative.  This forces the fusion view to model
            # text-image agreement instead of becoming a third copy.
            negative_image_feature = torch.roll(image_news_feature, shifts=1, dims=0)
            negative_clip_fusion = self.clip_fusion(torch.cat((
                torch.roll(clip_image_feature, shifts=1, dims=0),
                clip_text_feature,
            ), dim=-1).float())
            negative_pair_feature = torch.cat((
                torch.abs(text_news_feature - negative_image_feature),
                text_news_feature * negative_image_feature,
                negative_clip_fusion,
            ), dim=-1)
            match_positive_logit = self.match_classifier(pair_feature).squeeze(-1)
            match_negative_logit = self.match_classifier(negative_pair_feature).squeeze(-1)
            if self.ablation == 'token_relation_evidence':
                negative_indices = self._same_domain_hard_negative(
                    clip_text_feature, clip_image_feature, kwargs['category']
                )
                negative_relation_feature = self._relation_query_feature(
                    text_feature, image_feature[negative_indices], masks
                )
                match_positive_logit = self.relation_match_classifier(
                    relation_feature
                ).squeeze(-1)
                match_negative_logit = self.relation_match_classifier(
                    negative_relation_feature
                ).squeeze(-1)
            router_features = [
                text_view_feature, image_view_feature, fusion_view_feature
            ]
        else:
            text_domain_features = text_gate_expert_value[0]
            image_domain_features = image_gate_expert_value[0]
            fusion_domain_features = fusion_gate_expert_value0[0]

            if self.ablation == 'explicit_domain_gate':
                if analysis_mode == 'no_domain':
                    explicit_multipliers = (
                        torch.ones_like(text_domain_features),
                        torch.ones_like(image_domain_features),
                        torch.ones_like(fusion_domain_features),
                    )
                else:
                    explicit_category = kwargs['category']
                    if analysis_mode.startswith('shuffled_domain'):
                        parts = analysis_mode.rsplit('_', 1)
                        shift = int(parts[-1]) if parts[-1].isdigit() else 1
                        explicit_category = torch.roll(
                            explicit_category, shifts=shift, dims=0
                        )
                    explicit_scale = self.explicit_domain_scale(
                        self.explicit_domain_embedding(explicit_category)
                    )
                    explicit_scale = explicit_scale.reshape(
                        explicit_scale.shape[0], 3, 320
                    )
                    explicit_multipliers = tuple(
                        1.0
                        + self.signed_gate_alpha
                        * torch.tanh(explicit_scale[:, index, :])
                        for index in range(3)
                    )
                self.last_explicit_domain_multipliers = explicit_multipliers
                text_domain_features = (
                    explicit_multipliers[0] * text_domain_features
                )
                image_domain_features = (
                    explicit_multipliers[1] * image_domain_features
                )
                fusion_domain_features = (
                    explicit_multipliers[2] * fusion_domain_features
                )
            elif self.ablation == 'signed_gate':
                if analysis_mode == 'no_domain':
                    signed_multipliers = (
                        torch.ones_like(text_domain_features),
                        torch.ones_like(image_domain_features),
                        torch.ones_like(fusion_domain_features),
                    )
                else:
                    gate_context = fake_news_feature
                    if analysis_mode.startswith('shuffled_domain'):
                        parts = analysis_mode.rsplit('_', 1)
                        shift = int(parts[-1]) if parts[-1].isdigit() else 1
                        gate_context = torch.roll(
                            gate_context, shifts=shift, dims=0
                        )
                    signed_multipliers = (
                        1.0
                        + self.signed_gate_alpha
                        * self.gate_text_prefer(gate_context),
                        1.0
                        + self.signed_gate_alpha
                        * self.gate_image_prefer(gate_context),
                        1.0
                        + self.signed_gate_alpha
                        * self.gate_fusion_prefer(gate_context),
                    )
                self.last_signed_gate_multipliers = signed_multipliers
                text_domain_features = (
                    signed_multipliers[0] * text_domain_features
                )
                image_domain_features = (
                    signed_multipliers[1] * image_domain_features
                )
                fusion_domain_features = (
                    signed_multipliers[2] * fusion_domain_features
                )
            elif self.ablation != 'no_gate':
                if analysis_mode != 'no_domain':
                    gate_context = fake_news_feature
                    if analysis_mode.startswith('shuffled_domain'):
                        parts = analysis_mode.rsplit('_', 1)
                        shift = int(parts[-1]) if parts[-1].isdigit() else 1
                        gate_context = torch.roll(
                            gate_context, shifts=shift, dims=0
                        )
                    text_domain_features = self.gate_text_prefer(gate_context) * text_domain_features
                    image_domain_features = self.gate_image_prefer(gate_context) * image_domain_features
                    fusion_domain_features = self.gate_fusion_prefer(gate_context) * fusion_domain_features

            domain_aware_text_view = torch.sigmoid(
                self.domain_aware_text_classifier(
                    text_gate_expert_value[0] + text_domain_features
                ).squeeze(-1)
            )
            domain_aware_image_view = torch.sigmoid(
                self.domain_aware_image_classifier(
                    image_gate_expert_value[0] + image_domain_features
                ).squeeze(-1)
            )
            domain_aware_fusion_view = torch.sigmoid(
                self.domain_aware_fusion_classifier(
                    fusion_gate_expert_value0[0] + fusion_domain_features
                ).squeeze(-1)
            )

        if self.is_beda and self.output_mode == 'shared_evidence':
            # BEM structural ablation: erase modality-specific evidence before
            # the three prediction heads, while leaving the domain-aware
            # decision router intact. The heads may calibrate the common
            # evidence differently, but none receives a private modality view.
            shared_evidence_feature = torch.stack((
                text_gate_expert_value[0],
                image_gate_expert_value[0],
                fusion_gate_expert_value0[0],
            ), dim=0).mean(dim=0)
            shared_domain_feature = torch.stack((
                text_gate_expert_value[1],
                image_gate_expert_value[1],
                fusion_gate_expert_value0[1],
            ), dim=0).mean(dim=0)
            if analysis_mode == 'no_domain':
                shared_domain_feature = torch.zeros_like(
                    shared_domain_feature
                )
            elif analysis_mode.startswith('shuffled_domain'):
                parts = analysis_mode.rsplit('_', 1)
                shift = int(parts[-1]) if parts[-1].isdigit() else 1
                shared_domain_feature = torch.roll(
                    shared_domain_feature, shifts=shift, dims=0
                )
            shared_domain_gate = (
                self.gate_text_prefer(shared_domain_feature)
                + self.gate_image_prefer(shared_domain_feature)
                + self.gate_fusion_prefer(shared_domain_feature)
            ) / 3.0
            shared_view_feature = (
                shared_evidence_feature
                + shared_domain_gate * shared_domain_feature
            )
            domain_aware_text_view = torch.sigmoid(
                self.domain_aware_text_classifier(
                    shared_view_feature
                ).squeeze(-1)
            )
            domain_aware_image_view = torch.sigmoid(
                self.domain_aware_image_classifier(
                    shared_view_feature
                ).squeeze(-1)
            )
            domain_aware_fusion_view = torch.sigmoid(
                self.domain_aware_fusion_classifier(
                    shared_view_feature
                ).squeeze(-1)
            )
            router_features = [shared_view_feature] * 3

        # Domain-Enhanced Multi-view Decision Layer 
        if self.ablation == 'equal_weight':
            weight_common = torch.full(
                (multi_label_feature.shape[0], 3),
                1.0 / 3.0,
                dtype=multi_label_feature.dtype,
                device=multi_label_feature.device,
            )
        elif use_branch_evidence:
            if self.ablation in (
                'identity_branch_evidence', 'calibrated_branch_evidence',
                'token_relation_evidence', 'domain_tensor_router',
                'logit_evidence_router',
                'uncertainty_evidence_router',
                'evidence_router',
            ):
                # Preserve the released router at initialization, then learn a
                # zero-initialized domain-conditioned correction.
                attention_features = [
                    text_news_feature, image_news_feature, fusion_news_feature
                ]
                if self.is_beda and self.output_mode == 'shared_evidence':
                    # Apply the archived attention-input intervention as well
                    # as the shared prediction-head input above.
                    shared_semantic = torch.stack(attention_features, dim=0).mean(dim=0)
                    attention_features = [shared_semantic] * 3
                domain_weights = self.attention(
                    attention_features, multi_label_feature,
                )
                domain_router_bias = (
                    self.domain_router_bias(domain_feature)
                    + semantic_router_bias
                )
            else:
                domain_weights = self.attention(router_features, domain_feature)
                domain_router_bias = 0.0
            view_probabilities = torch.stack((
                domain_aware_text_view,
                domain_aware_image_view,
                domain_aware_fusion_view,
            ), dim=1)
            reliability_logits = torch.cat([
                head(torch.cat((feature, view_probabilities[:, index:index + 1]), dim=1))
                for index, (head, feature) in enumerate(
                    zip(self.reliability_heads, router_features)
                )
            ], dim=1)
            weight_common = torch.softmax(
                torch.log(domain_weights.clamp_min(1e-8))
                + reliability_logits
                + domain_router_bias,
                dim=1,
            )
            if analysis_mode == 'uniform_route':
                weight_common = torch.full_like(weight_common, 1.0 / 3.0)
            elif analysis_mode == 'fusion_only':
                weight_common = torch.zeros_like(weight_common)
                weight_common[:, 2] = 1.0
        else:
            weight_common = self.attention([text_gate_expert_value[0], image_gate_expert_value[0], fusion_gate_expert_value0[0]], multi_label_feature)

        fake_news_sigmoid = (
            weight_common[:, 0] * domain_aware_text_view.reshape(-1)
            + weight_common[:, 1] * domain_aware_image_view.reshape(-1)
            + weight_common[:, 2] * domain_aware_fusion_view.reshape(-1)
        )

        fake_news_sigmoid = torch.clamp(
            fake_news_sigmoid, min=1e-6, max=1.0 - 1e-6
        )
        if self.ablation == 'domain_tensor_router':
            soft_domain = (
                torch.sigmoid(text_multi_domain)
                + torch.sigmoid(image_multi_domain)
                + torch.sigmoid(fusion_multi_domain)
            ).detach() / 3.0
            tensor_correction = self._domain_tensor_correction(
                text_news_feature,
                image_news_feature,
                fusion_news_feature,
                kwargs['category'],
                soft_domain,
                analysis_mode,
            )
            if analysis_mode == 'no_tensor':
                tensor_correction = torch.zeros_like(tensor_correction)
            self.last_tensor_correction = tensor_correction
            fake_news_sigmoid = torch.sigmoid(
                torch.logit(fake_news_sigmoid) + tensor_correction
            )
        elif self.ablation == 'domain_log_pool':
            soft_domain = (
                text_multi_domain
                + image_multi_domain
                + fusion_multi_domain
            ).detach() / 3.0
            branch_probabilities = torch.stack((
                domain_aware_text_view.reshape(-1),
                domain_aware_image_view.reshape(-1),
                domain_aware_fusion_view.reshape(-1),
            ), dim=-1)
            pool_correction, pool_weights = (
                self._domain_log_pool_correction(
                    fake_news_sigmoid,
                    branch_probabilities,
                    weight_common,
                    kwargs['category'],
                    soft_domain,
                    analysis_mode,
                )
            )
            if analysis_mode == 'no_pool':
                pool_correction = torch.zeros_like(pool_correction)
            self.last_pool_correction = pool_correction
            self.last_pool_weights = pool_weights
            base_pool_logit = torch.logit(fake_news_sigmoid)
            fake_news_sigmoid = torch.sigmoid(
                base_pool_logit + pool_correction
            )
            # Counterfactuals reuse the same frozen branch evidence.  They are
            # consumed only by the optional training-time margin objective;
            # no extra encoder pass and no test label are involved.
            no_domain_correction, _ = self._domain_log_pool_correction(
                torch.sigmoid(base_pool_logit),
                branch_probabilities,
                weight_common,
                kwargs['category'],
                soft_domain,
                'no_pool_domain',
            )
            shuffled_correction, _ = self._domain_log_pool_correction(
                torch.sigmoid(base_pool_logit),
                branch_probabilities,
                weight_common,
                kwargs['category'],
                soft_domain,
                'shuffled_pool_domain_1',
            )
            self.pool_counterfactual_predictions = (
                torch.sigmoid(base_pool_logit + no_domain_correction),
                torch.sigmoid(base_pool_logit + shuffled_correction),
            )
        elif self.is_beda and self.output_mode == 'reference_only':
            # w/o Refine: return the original, clamped arithmetic p0 exactly.
            # Keep reference attention/domain/reliability gates; never invoke
            # state experts, pw, bounded fusion or their counterfactual paths.
            self.last_base_probability = fake_news_sigmoid
            self.last_interaction_weights = weight_common
            self.last_interaction_correction = torch.zeros_like(fake_news_sigmoid)
            self.last_interaction_state_weights = fake_news_sigmoid.new_zeros(
                (fake_news_sigmoid.shape[0], 4)
            )
            self.interaction_counterfactual_predictions = (
                fake_news_sigmoid, fake_news_sigmoid,
            )
        elif self.ablation in (
            'probability_evidence_router',
            'logit_evidence_router',
            'uncertainty_evidence_router',
            'probability_router_with_signal_control',
            'evidence_router',
        ):
            if self.ablation in (
                'logit_evidence_router',
                'uncertainty_evidence_router',
                'evidence_router',
            ):
                soft_domain = (
                    torch.sigmoid(text_multi_domain)
                    + torch.sigmoid(image_multi_domain)
                    + torch.sigmoid(fusion_multi_domain)
                ).detach() / 3.0
            else:
                soft_domain = (
                    text_multi_domain
                    + image_multi_domain
                    + fusion_multi_domain
                ).detach() / 3.0
            branch_probabilities = torch.stack((
                domain_aware_text_view.reshape(-1),
                domain_aware_image_view.reshape(-1),
                domain_aware_fusion_view.reshape(-1),
            ), dim=-1)
            interaction_correction, interaction_weights, state_weights = (
                self._interaction_moe_correction(
                    fake_news_sigmoid,
                    branch_probabilities,
                    weight_common,
                    kwargs['category'],
                    soft_domain,
                    clip_similarity,
                    analysis_mode,
                    match_positive_logit,
                )
            )
            if (
                analysis_mode == 'no_interaction'
                or (self.is_beda and self.dcea_mode == 'off')
            ):
                interaction_correction = torch.zeros_like(
                    interaction_correction
                )
            if self.ablation in (
                'probability_router_with_signal_control',
                'evidence_router',
            ):
                signal_logit_correction = torch.zeros_like(
                    fake_news_sigmoid
                )
                if self.is_beda:
                    signal_logit_correction = (
                        self._evidence_logit_correction(
                            fake_news_sigmoid,
                            branch_probabilities,
                            kwargs['category'],
                            soft_domain,
                            clip_similarity,
                            analysis_mode,
                        )
                    )
                self.last_dir_correction = interaction_correction
                # DIR is an already validated part of the unified baseline.
                # DACG must judge only the newly proposed CDSD residual;
                # otherwise the strong DIR gain makes nearly every target an
                # "accept" target and the safety gate collapses to one.
                dir_probability = torch.sigmoid(
                    torch.logit(fake_news_sigmoid)
                    + interaction_correction
                )
                self.last_base_probability = dir_probability
                self.last_dacg_base_probability = dir_probability
                self.last_dacg_full_correction_probability = torch.sigmoid(
                    torch.logit(dir_probability)
                    + signal_logit_correction
                )
                correction_acceptance = self._correction_acceptance(
                    dir_probability,
                    branch_probabilities,
                    signal_logit_correction,
                    kwargs['category'],
                    soft_domain,
                    clip_similarity,
                    analysis_mode,
                )
                interaction_correction = (
                    interaction_correction
                    + correction_acceptance * signal_logit_correction
                )
            self.last_interaction_correction = interaction_correction
            self.last_interaction_weights = interaction_weights
            self.last_interaction_state_weights = state_weights
            base_interaction_logit = torch.logit(fake_news_sigmoid)
            fake_news_sigmoid = torch.sigmoid(
                base_interaction_logit + interaction_correction
            )
            no_domain_correction, _, _ = (
                self._interaction_moe_correction(
                    torch.sigmoid(base_interaction_logit),
                    branch_probabilities,
                    weight_common,
                    kwargs['category'],
                    soft_domain,
                    clip_similarity,
                    'no_interaction_domain',
                    match_positive_logit,
                )
            )
            shuffled_correction, _, _ = (
                self._interaction_moe_correction(
                    torch.sigmoid(base_interaction_logit),
                    branch_probabilities,
                    weight_common,
                    kwargs['category'],
                    soft_domain,
                    clip_similarity,
                    'shuffled_interaction_domain_1',
                    match_positive_logit,
                )
            )
            self.interaction_counterfactual_predictions = (
                torch.sigmoid(
                    base_interaction_logit + no_domain_correction
                ),
                torch.sigmoid(
                    base_interaction_logit + shuffled_correction
                ),
            )

        if self.is_beda:
            if self.output_mode in ('fixed_mean', 'no_arbitration'):
                # Large router ablation: all three branches receive the same
                # fixed weight, with no domain- or conflict-aware decision.
                fake_news_sigmoid = torch.stack((
                    domain_aware_text_view,
                    domain_aware_image_view,
                    domain_aware_fusion_view,
                ), dim=-1).mean(dim=-1)
                self.last_interaction_weights = fake_news_sigmoid.new_full(
                    (fake_news_sigmoid.shape[0], 3), 1.0 / 3.0
                )
            elif self.output_mode == 'fusion_only':
                # Large signal-modeling ablation: discard branch comparison
                # and use the conventional single fused-feature FFN only.
                fake_news_sigmoid = domain_aware_fusion_view.squeeze(-1)
            elif self.output_mode == 'plain_ffn':
                fake_news_sigmoid = plain_fusion_probability
            elif self.output_mode not in ('full', 'shared_evidence', 'reference_only'):
                raise ValueError(
                    f'unknown output_mode: '
                    f'{self.output_mode}'
                )

        outputs = (
            fake_news_sigmoid, text_fake_news, text_multi_domain,
            image_fake_news, image_multi_domain, fusion_fake_news,
            fusion_multi_domain, domain_aware_text_view,
            domain_aware_image_view, domain_aware_fusion_view,
        )
        if self.ablation in (
            'branch_evidence', 'identity_branch_evidence', 'calibrated_branch_evidence',
            'token_relation_evidence', 'domain_tensor_router',
        ):
            outputs += (
                weight_common, match_positive_logit, match_negative_logit
            )
        elif self.ablation in (
            'logit_evidence_router',
            'uncertainty_evidence_router',
            'evidence_router',
        ):
            outputs += (
                self.last_interaction_weights,
                match_positive_logit,
                match_negative_logit,
            )
        elif self.ablation == 'domain_log_pool':
            zero_match_logit = torch.zeros_like(fake_news_sigmoid)
            outputs += (
                self.last_pool_weights,
                zero_match_logit,
                zero_match_logit,
            )
        elif self.ablation in (
            'probability_evidence_router',
            'probability_router_with_signal_control',
        ):
            zero_match_logit = torch.zeros_like(fake_news_sigmoid)
            outputs += (
                self.last_interaction_weights,
                zero_match_logit,
                zero_match_logit,
            )
        return outputs



class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 loss_weight=[1, 0.006, 0.009, 5e-5],
                 early_stop=5,
                 epoches=100,
                 ablation='beda_full',
                 arbitration_loss_weight=0.2,
                 match_loss_weight=0.2,
                 balance_loss_weight=0.02,
                 arbitration_temperature=0.25,
                 domain_loss_weight=1.0,
                 view_loss_weight=1.0,
                 init_checkpoint='',
                 new_module_lr_multiplier=5.0,
                 fusion_view_loss_multiplier=1.0,
                 signed_gate_alpha=0.5,
                 skip_final_test=False,
                 validation_auc_only=False,
                 freeze_base_for_relation=False,
                 freeze_base_for_tensor=False,
                 tensor_domain_mode='full',
                 freeze_base_for_pool=False,
                 pool_domain_mode='full',
                 pool_rank_loss_weight=0.0,
                 pool_rank_margin=0.02,
                 freeze_base_for_signal_control=False,
                 dacg_loss_weight=0.2,
                 signal_mode='off',
                 acceptance_mode='off',
                 signal_domain_mode='full',
                 freeze_backbone=False,
                 train_scope='all',
                 bem_mode='full',
                 dcea_mode='full',
                 state_mode='dual',
                 correction_scale=1.0,
                 routing_family='shared',
                 output_mode='full',
                 hard_sample_loss_weight=0.0,
                 hard_sample_temperature=1.0,
                 selection_metric='auc',
                 drcm_mode='full',
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        self.requested_ablation = ablation
        self.is_beda = ablation == 'beda_full'
        self.ablation = 'evidence_router' if self.is_beda else ablation
        self.arbitration_loss_weight = arbitration_loss_weight
        self.match_loss_weight = match_loss_weight
        self.balance_loss_weight = balance_loss_weight
        self.arbitration_temperature = arbitration_temperature
        self.domain_loss_weight = domain_loss_weight
        self.view_loss_weight = view_loss_weight
        self.init_checkpoint = init_checkpoint
        self.new_module_lr_multiplier = new_module_lr_multiplier
        self.fusion_view_loss_multiplier = fusion_view_loss_multiplier
        self.signed_gate_alpha = signed_gate_alpha
        self.skip_final_test = skip_final_test
        self.validation_auc_only = validation_auc_only
        self.freeze_base_for_relation = freeze_base_for_relation
        self.freeze_base_for_tensor = freeze_base_for_tensor
        self.tensor_domain_mode = tensor_domain_mode
        self.freeze_base_for_pool = freeze_base_for_pool
        self.pool_domain_mode = pool_domain_mode
        self.pool_rank_loss_weight = pool_rank_loss_weight
        self.pool_rank_margin = pool_rank_margin
        self.freeze_base_for_signal_control = freeze_base_for_signal_control
        self.dacg_loss_weight = dacg_loss_weight
        self.signal_mode = signal_mode
        self.acceptance_mode = acceptance_mode
        self.signal_domain_mode = signal_domain_mode
        self.freeze_backbone = freeze_backbone
        self.train_scope = train_scope
        self.bem_mode = bem_mode
        self.drcm_mode = drcm_mode
        self.dcea_mode = dcea_mode
        self.state_mode = state_mode
        self.correction_scale = correction_scale
        self.routing_family = routing_family
        self.output_mode = output_mode
        self.hard_sample_loss_weight = hard_sample_loss_weight
        self.hard_sample_temperature = hard_sample_temperature
        self.selection_metric = selection_metric
        os.makedirs(save_param_dir, exist_ok=True)
        self.save_param_dir = save_param_dir

    def train(self):
        self.model = BEDAFNDModel(
            self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout,
            ablation=self.requested_ablation,
            signed_gate_alpha=self.signed_gate_alpha,
            tensor_domain_mode=self.tensor_domain_mode,
            pool_domain_mode=self.pool_domain_mode,
            signal_mode=self.signal_mode,
            acceptance_mode=self.acceptance_mode,
            signal_domain_mode=self.signal_domain_mode,
            bem_mode=self.bem_mode,
            drcm_mode=self.drcm_mode,
            dcea_mode=self.dcea_mode,
            state_mode=self.state_mode,
            correction_scale=self.correction_scale,
            routing_family=self.routing_family,
            output_mode=self.output_mode,
        )
        if self.use_cuda:
            self.model = self.model.cuda()
        if self.init_checkpoint:
            checkpoint_state = torch.load(
                self.init_checkpoint,
                map_location='cuda' if self.use_cuda else 'cpu',
            )
            load_result = self.model.load_state_dict(
                checkpoint_state, strict=False
            )
            print(
                'Warm start checkpoint:', self.init_checkpoint,
                'missing keys:', len(load_result.missing_keys),
                'unexpected keys:', len(load_result.unexpected_keys),
            )
        loss_fn = torch.nn.BCELoss()
        if self.freeze_backbone:
            if not self.is_beda:
                raise ValueError(
                    'freeze_backbone requires beda_full'
                )
            trainable_prefixes = []
            if self.train_scope == 'state':
                if self.dcea_mode == 'off':
                    raise ValueError('state scope requires DCEA')
                trainable_prefixes.extend((
                    'interaction_feature_state_calibrator',
                    'interaction_decision_state_calibrator',
                ))
            if (
                self.train_scope == 'all'
                and self.bem_mode != 'off'
            ):
                trainable_prefixes.extend((
                    'text_specializer', 'image_specializer',
                    'fusion_specializer', 'text_domain_adapter',
                    'image_domain_adapter', 'fusion_domain_adapter',
                    'consistency_adapter', 'match_classifier',
                    'reliability_heads', 'domain_router_bias',
                ))
            if (
                self.train_scope == 'all'
                and self.dcea_mode != 'off'
            ):
                trainable_prefixes.append('interaction_')
            if self.signal_mode != 'off':
                trainable_prefixes.append('signal_')
            if self.acceptance_mode != 'off':
                trainable_prefixes.append('dacg_')
            if not trainable_prefixes:
                raise ValueError(
                    'unified training needs at least one enabled module'
                )
            trainable_prefixes = tuple(trainable_prefixes)
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(name.startswith(trainable_prefixes))
            trainable_parameters = [
                parameter for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.lr * self.new_module_lr_multiplier,
                weight_decay=self.weight_decay,
            )
            print(
                'Frozen-backbone unified training; trainable parameters:',
                sum(parameter.numel() for parameter in trainable_parameters),
                'scope:', self.train_scope,
                'BSM:', self.bem_mode,
                'CDSD:', self.signal_mode,
                'DCEA:', self.dcea_mode,
                'DACG:', self.acceptance_mode,
            )
        elif self.freeze_base_for_signal_control:
            if self.ablation not in (
                'probability_router_with_signal_control',
                'evidence_router',
            ):
                raise ValueError(
                    'freeze_base_for_signal_control requires a signal-control signal/DACG variant'
                )
            trainable_prefixes = []
            if self.signal_mode != 'off':
                trainable_prefixes.append('signal_')
            if self.acceptance_mode != 'off':
                trainable_prefixes.append('dacg_')
            if not trainable_prefixes:
                raise ValueError(
                    'signal-control frozen-base training needs signal or DACG enabled'
                )
            trainable_prefixes = tuple(trainable_prefixes)
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(name.startswith(trainable_prefixes))
            trainable_parameters = [
                parameter for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.lr * self.new_module_lr_multiplier,
                weight_decay=self.weight_decay,
            )
            print(
                'Frozen-base signal-control signal/DACG training; trainable parameters:',
                sum(parameter.numel() for parameter in trainable_parameters),
                'variant:', self.ablation,
            )
        elif self.freeze_base_for_pool:
            if self.ablation not in (
                'domain_log_pool',
                'probability_evidence_router',
                'logit_evidence_router',
                'uncertainty_evidence_router',
            ):
                raise ValueError(
                    'freeze_base_for_pool requires '
                    'domain_log_pool or '
                    'probability_evidence_router or '
                    'logit_evidence_router or '
                    'uncertainty_evidence_router'
                )
            trainable_prefix = (
                'pool_'
                if self.ablation == 'domain_log_pool'
                else 'interaction_'
            )
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(name.startswith(trainable_prefix))
            trainable_parameters = [
                parameter for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.lr * self.new_module_lr_multiplier,
                weight_decay=self.weight_decay,
            )
            print(
                'Frozen-base domain fusion training; trainable parameters:',
                sum(parameter.numel() for parameter in trainable_parameters),
                'variant:', self.ablation,
                'domain mode:', self.pool_domain_mode,
            )
        elif self.freeze_base_for_tensor:
            if self.ablation != 'domain_tensor_router':
                raise ValueError(
                    'freeze_base_for_tensor requires '
                    'domain_tensor_router'
                )
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(name.startswith('tensor_'))
            trainable_parameters = [
                parameter for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.lr * self.new_module_lr_multiplier,
                weight_decay=self.weight_decay,
            )
            print(
                'Frozen-base domain-tensor training; trainable parameters:',
                sum(parameter.numel() for parameter in trainable_parameters),
                'domain mode:', self.tensor_domain_mode,
            )
        elif self.freeze_base_for_relation:
            if self.ablation != 'token_relation_evidence':
                raise ValueError(
                    'freeze_base_for_relation requires token_relation_evidence'
                )
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(name.startswith('relation_'))
            trainable_parameters = [
                parameter for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.lr * self.new_module_lr_multiplier,
                weight_decay=self.weight_decay,
            )
            print(
                'Frozen-base relation training; trainable parameters:',
                sum(parameter.numel() for parameter in trainable_parameters),
            )
        elif self.init_checkpoint and self.ablation in (
            'branch_evidence', 'identity_branch_evidence', 'calibrated_branch_evidence',
            'token_relation_evidence', 'domain_tensor_router',
        ):
            if self.ablation == 'calibrated_branch_evidence':
                new_module_prefixes = (
                    'semantic_domain_encoder',
                    'semantic_domain_film',
                    'semantic_domain_router',
                )
            else:
                new_module_prefixes = (
                    'text_specializer', 'image_specializer',
                    'fusion_specializer', 'text_domain_adapter',
                    'image_domain_adapter', 'fusion_domain_adapter',
                    'text_view_norm', 'image_view_norm', 'fusion_view_norm',
                    'consistency_adapter', 'match_classifier',
                    'reliability_heads', 'domain_router_bias',
                )
                if self.ablation == 'token_relation_evidence':
                    new_module_prefixes += (
                        'relation_queries',
                        'relation_text_projection',
                        'relation_image_projection',
                        'relation_text_attention',
                        'relation_image_attention',
                        'relation_token_mlp',
                        'relation_pool',
                        'relation_output',
                        'relation_match_classifier',
                    )
                elif self.ablation == 'domain_tensor_router':
                    new_module_prefixes += (
                        'tensor_text_projection',
                        'tensor_image_projection',
                        'tensor_fusion_projection',
                        'tensor_domain_projection',
                        'tensor_interaction_norm',
                        'tensor_correction',
                    )
            base_parameters, new_parameters = [], []
            for name, parameter in self.model.named_parameters():
                target = (
                    new_parameters
                    if name.startswith(new_module_prefixes)
                    else base_parameters
                )
                target.append(parameter)
            optimizer = torch.optim.Adam(
                [
                    {'params': base_parameters, 'lr': self.lr},
                    {
                        'params': new_parameters,
                        'lr': self.lr * self.new_module_lr_multiplier,
                    },
                ],
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop)
        best_validation_auc = float('-inf')
        best_validation_accuracy = float('-inf')
        best_validation_accuracy_auc = float('-inf')
        best_validation_routing_match = float('-inf')
        best_validation_fusion_accuracy = float('-inf')
        if self.init_checkpoint:
            initial_results = self.test(self.val_loader)
            results0 = initial_results
            if not self.validation_auc_only:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_param_dir,
                        'parameter_beda_fnd_initial.pkl',
                    ),
                )
            best_validation_auc = initial_results.get(
                'auc', float('-inf')
            )
            best_validation_routing_match = initial_results.get(
                'routing_match', float('-inf')
            )
            best_validation_fusion_accuracy = initial_results.get(
                'fusion_accuracy', float('-inf')
            )
            best_validation_accuracy = initial_results.get(
                'acc', float('-inf')
            )
            best_validation_accuracy_auc = initial_results.get(
                'auc', float('-inf')
            )
            if self.selection_metric == 'auc' or not self.validation_auc_only:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_param_dir,
                        'parameter_beda_fnd_best_auc.pkl',
                    ),
                )
            if self.selection_metric == 'accuracy':
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_param_dir,
                        'parameter_beda_fnd_best_accuracy.pkl',
                    ),
                )
            if not self.validation_auc_only:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_param_dir,
                        'parameter_beda_fnd_best_arbitration.pkl',
                    ),
                )
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_param_dir,
                        'parameter_beda_fnd_best_fusion.pkl',
                    ),
                )
            if recorder.add(initial_results) == 'save':
                if not self.validation_auc_only:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.save_param_dir,
                            'parameter_beda_fnd.pkl',
                        ),
                    )
        for epoch in range(self.epoches):
            # Frozen-backbone experiments must also freeze BatchNorm running
            # statistics and dropout behavior.  Trainable tensor parameters
            # still receive gradients in eval mode.
            if (
                self.freeze_base_for_tensor
                or self.freeze_base_for_pool
                or self.freeze_base_for_signal_control
                or self.freeze_backbone
            ):
                self.model.eval()
            else:
                self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = clipdata2gpu(batch)
                label = batch_data['label']
                category = batch_data['multi_category']
                labels_domain = category

                model_outputs = self.model(**batch_data)
                label0, text_fake_news, text_multi_domain, image_fake_news, image_multi_domain, fusion_fake_news, fusion_multi_domain, domain_aware_text_view, domain_aware_image_view, domain_aware_fusion_view = model_outputs[:10]
                loss0 = loss_fn(label0, label.float())


                loss11 = torch.nn.functional.binary_cross_entropy_with_logits(text_multi_domain, labels_domain.float())
                loss21 = torch.nn.functional.binary_cross_entropy_with_logits(image_multi_domain, labels_domain.float())
                loss31 = torch.nn.functional.binary_cross_entropy_with_logits(fusion_multi_domain,
                                                                              labels_domain.float())


                loss12_aux = loss_fn(domain_aware_text_view.squeeze(), label.float())
                loss22_aux = loss_fn(domain_aware_image_view.squeeze(), label.float())
                loss32_auc = loss_fn(domain_aware_fusion_view.squeeze(), label.float())

                if self.ablation in (
                    'branch_evidence', 'identity_branch_evidence', 'calibrated_branch_evidence',
                    'token_relation_evidence', 'domain_tensor_router',
                    'logit_evidence_router',
                    'uncertainty_evidence_router',
                    'evidence_router',
                ):
                    # The corrected domain objective directly supervises the
                    # multi-label logits.  The old scalar-softmax KL term is
                    # mathematically constant and is intentionally omitted.
                    domain_disentanglement_loss = (
                        loss11 + loss21 + loss31
                    ) / 3.0
                else:
                    uniform_target = torch.ones_like(text_fake_news, dtype=torch.float).cuda() / 9
                    loss12 = F.kl_div(text_fake_news, uniform_target.float())
                    loss22 = F.kl_div(image_fake_news, uniform_target.float())
                    loss32 = F.kl_div(fusion_fake_news, uniform_target.float())
                    domain_disentanglement_loss = (
                        loss11 + loss12 + loss21 + loss22 + loss31 + loss32
                    ) / 6
                if self.ablation == 'no_dd':
                    domain_disentanglement_loss = 0.0
                loss = (
                    loss0
                    + self.domain_loss_weight * domain_disentanglement_loss
                    + self.view_loss_weight
                    * (
                        loss12_aux
                        + loss22_aux
                        + self.fusion_view_loss_multiplier * loss32_auc
                    )
                    / (2.0 + self.fusion_view_loss_multiplier)
                )

                if (
                    self.freeze_backbone
                    and self.bem_mode != 'off'
                    and self.match_loss_weight > 0.0
                ):
                    match_positive_logit = model_outputs[11]
                    match_negative_logit = model_outputs[12]
                    bem_match_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(
                            match_positive_logit,
                            torch.ones_like(match_positive_logit),
                        )
                        + F.binary_cross_entropy_with_logits(
                            match_negative_logit,
                            torch.zeros_like(match_negative_logit),
                        )
                    )
                    loss = (
                        loss
                        + self.match_loss_weight * bem_match_loss
                    )

                if (
                    self.ablation in (
                        'probability_router_with_signal_control',
                        'evidence_router',
                    )
                    and self.dacg_loss_weight > 0.0
                    and self.acceptance_mode != 'off'
                ):
                    base_loss = F.binary_cross_entropy(
                        self.model.last_dacg_base_probability,
                        label.float(),
                        reduction='none',
                    )
                    full_correction_loss = F.binary_cross_entropy(
                        self.model.last_dacg_full_correction_probability,
                        label.float(),
                        reduction='none',
                    )
                    acceptance_target = (
                        full_correction_loss < base_loss
                    ).detach().to(dtype=label0.dtype)
                    raw_gate_loss = F.binary_cross_entropy_with_logits(
                        self.model.last_dacg_raw_acceptance,
                        acceptance_target,
                        reduction='none',
                    )
                    positive_rate = acceptance_target.mean()
                    if 0.0 < positive_rate < 1.0:
                        balanced_weight = torch.where(
                            acceptance_target > 0.5,
                            0.5 / positive_rate,
                            0.5 / (1.0 - positive_rate),
                        )
                        dacg_loss = (
                            balanced_weight * raw_gate_loss
                        ).mean()
                    else:
                        dacg_loss = raw_gate_loss.mean()
                    loss = loss + self.dacg_loss_weight * dacg_loss

                if (
                    self.freeze_backbone
                    and self.hard_sample_loss_weight > 0.0
                ):
                    base_probability = (
                        self.model.last_base_probability
                        .detach()
                        .clamp(1e-6, 1.0 - 1e-6)
                    )
                    base_margin = torch.abs(torch.logit(base_probability))
                    hard_weight = torch.exp(
                        -base_margin / self.hard_sample_temperature
                    )
                    per_sample_final_loss = F.binary_cross_entropy(
                        label0,
                        label.float(),
                        reduction='none',
                    )
                    hard_loss = (
                        hard_weight * per_sample_final_loss
                    ).sum() / hard_weight.sum().clamp_min(1e-6)
                    loss = (
                        loss
                        + self.hard_sample_loss_weight * hard_loss
                    )

                if (
                    self.ablation in (
                        'domain_log_pool',
                        'probability_evidence_router',
                        'logit_evidence_router',
                        'uncertainty_evidence_router',
                        'probability_router_with_signal_control',
                        'evidence_router',
                    )
                    and self.pool_rank_loss_weight > 0.0
                ):
                    correct_loss = F.binary_cross_entropy(
                        label0,
                        label.float(),
                        reduction='none',
                    )
                    rank_losses = []
                    counterfactuals = (
                        self.model.pool_counterfactual_predictions
                        if self.ablation == 'domain_log_pool'
                        else self.model.interaction_counterfactual_predictions
                    )
                    for counterfactual in counterfactuals:
                        counterfactual_loss = F.binary_cross_entropy(
                            counterfactual,
                            label.float(),
                            reduction='none',
                        )
                        rank_losses.append(torch.relu(
                            self.pool_rank_margin
                            + correct_loss
                            - counterfactual_loss
                        ).mean())
                    domain_rank_loss = torch.stack(rank_losses).mean()
                    loss = (
                        loss
                        + self.pool_rank_loss_weight * domain_rank_loss
                    )

                if (
                    self.ablation in (
                        'probability_evidence_router',
                        'logit_evidence_router',
                        'uncertainty_evidence_router',
                        'probability_router_with_signal_control',
                        'evidence_router',
                    )
                    and self.arbitration_loss_weight > 0.0
                ):
                    interaction_weights = model_outputs[10]
                    view_probabilities = torch.stack((
                        domain_aware_text_view,
                        domain_aware_image_view,
                        domain_aware_fusion_view,
                    ), dim=1).clamp(1e-6, 1.0 - 1e-6)
                    expanded_labels = label.float().unsqueeze(1).expand_as(
                        view_probabilities
                    )
                    correct_experts = (
                        (view_probabilities >= 0.5)
                        == expanded_labels.bool()
                    ).detach().to(dtype=interaction_weights.dtype)
                    correct_mass = torch.sum(
                        interaction_weights * correct_experts,
                        dim=1,
                    )
                    has_correct_expert = correct_experts.sum(dim=1) > 0
                    # Set-likelihood routing: assigning weight to any correct
                    # branch is rewarded.  If every branch is correct, the
                    # mass is one and no needless rerouting occurs; if all are
                    # wrong, the sample cannot teach the frozen router and is
                    # skipped.
                    if has_correct_expert.any():
                        interaction_arbitration_loss = -torch.log(
                            correct_mass[has_correct_expert].clamp_min(1e-8)
                        ).mean()
                    else:
                        interaction_arbitration_loss = (
                            interaction_weights.sum() * 0.0
                        )
                    loss = (
                        loss
                        + self.arbitration_loss_weight
                        * interaction_arbitration_loss
                    )

                if self.ablation in (
                    'branch_evidence', 'identity_branch_evidence', 'calibrated_branch_evidence',
                    'token_relation_evidence', 'domain_tensor_router',
                ):
                    weight_common, match_positive_logit, match_negative_logit = model_outputs[10:13]
                    view_probabilities = torch.stack((
                        domain_aware_text_view,
                        domain_aware_image_view,
                        domain_aware_fusion_view,
                    ), dim=1).clamp(1e-6, 1.0 - 1e-6)
                    expanded_labels = label.float().unsqueeze(1).expand_as(
                        view_probabilities
                    )
                    per_view_loss = F.binary_cross_entropy(
                        view_probabilities,
                        expanded_labels,
                        reduction='none',
                    )
                    arbitration_target = torch.softmax(
                        -per_view_loss.detach() / self.arbitration_temperature,
                        dim=1,
                    )
                    arbitration_loss = -(
                        arbitration_target
                        * torch.log(weight_common.clamp_min(1e-8))
                    ).sum(dim=1).mean()
                    match_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(
                            match_positive_logit,
                            torch.ones_like(match_positive_logit),
                        )
                        + F.binary_cross_entropy_with_logits(
                            match_negative_logit,
                            torch.zeros_like(match_negative_logit),
                        )
                    )
                    uniform_weights = torch.full_like(
                        weight_common.mean(dim=0), 1.0 / 3.0
                    )
                    balance_loss = torch.sum(
                        (weight_common.mean(dim=0) - uniform_weights) ** 2
                    )
                    loss = (
                        loss
                        + self.arbitration_loss_weight * arbitration_loss
                        + self.match_loss_weight * match_loss
                        + self.balance_loss_weight * balance_loss
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            print("----- self.save_param_dir", self.save_param_dir)
            results0 = self.test(self.val_loader)
            if results0.get('auc', float('-inf')) > best_validation_auc:
                best_validation_auc = results0['auc']
                if (
                    self.selection_metric == 'auc'
                    or not self.validation_auc_only
                ):
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.save_param_dir,
                            'parameter_beda_fnd_best_auc.pkl',
                        ),
                    )
            current_accuracy = results0.get('acc', float('-inf'))
            current_auc = results0.get('auc', float('-inf'))
            selection_tolerance = 1e-12
            if (
                current_accuracy
                > best_validation_accuracy + selection_tolerance
                or (
                    abs(
                        current_accuracy - best_validation_accuracy
                    ) <= selection_tolerance
                    and current_auc
                    > best_validation_accuracy_auc + selection_tolerance
                )
            ):
                best_validation_accuracy = current_accuracy
                best_validation_accuracy_auc = current_auc
                if self.selection_metric == 'accuracy':
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.save_param_dir,
                            'parameter_beda_fnd_best_accuracy.pkl',
                        ),
                    )
            if (
                not self.validation_auc_only
                and results0.get('routing_match', float('-inf'))
                > best_validation_routing_match
            ):
                best_validation_routing_match = results0['routing_match']
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_param_dir,
                        'parameter_beda_fnd_best_arbitration.pkl',
                    ),
                )
            if (
                not self.validation_auc_only
                and results0.get('fusion_accuracy', float('-inf'))
                > best_validation_fusion_accuracy
            ):
                best_validation_fusion_accuracy = results0['fusion_accuracy']
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_param_dir,
                        'parameter_beda_fnd_best_fusion.pkl',
                    ),
                )
            mark = recorder.add(results0)
            if mark == 'save':
                if not self.validation_auc_only:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.save_param_dir, 'parameter_beda_fnd.pkl'
                        ),
                    )
            elif mark == 'esc':
                break
            else:
                continue
        if self.validation_auc_only:
            checkpoint_name = (
                'parameter_beda_fnd_best_accuracy.pkl'
                if self.selection_metric == 'accuracy'
                else 'parameter_beda_fnd_best_auc.pkl'
            )
        else:
            checkpoint_name = 'parameter_beda_fnd.pkl'
        selected_checkpoint = os.path.join(self.save_param_dir, checkpoint_name)
        if self.skip_final_test:
            print(
                'Final test skipped by protocol; checkpoint selection used '
                'validation data only.'
            )
            return results0, selected_checkpoint
        self.model.load_state_dict(torch.load(selected_checkpoint))
        print("开始进行最后的测试")
        results0 = self.test(self.test_loader)
        print("final: ", results0)

        return results0, selected_checkpoint

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        view_predictions = [[], [], []]
        decision_weights = []
        signal_gains = []
        dacg_acceptance = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu(batch)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                model_outputs = self.model(**batch_data)
                batch_label_pred = model_outputs[0]

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                if self.ablation in (
                    'branch_evidence', 'identity_branch_evidence', 'calibrated_branch_evidence',
                    'token_relation_evidence', 'domain_tensor_router',
                    'domain_log_pool',
                    'probability_evidence_router',
                    'logit_evidence_router',
                    'uncertainty_evidence_router',
                    'probability_router_with_signal_control',
                    'evidence_router',
                ):
                    for view_index, output_index in enumerate((7, 8, 9)):
                        view_predictions[view_index].extend(
                            model_outputs[output_index]
                            .reshape(-1)
                            .detach()
                            .cpu()
                            .numpy()
                            .tolist()
                        )
                    decision_weights.extend(
                        model_outputs[10].detach().cpu().numpy().tolist()
                    )
                if self.ablation in (
                    'probability_router_with_signal_control',
                    'evidence_router',
                ):
                    signal_gains.extend(
                        self.model.last_signal_gains
                        .detach().cpu().numpy().tolist()
                    )
                    dacg_acceptance.extend(
                        self.model.last_dacg_acceptance
                        .detach().cpu().numpy().tolist()
                    )

        metric_res = metricsTrueFalse(label, pred, category, self.category_dict)
        if not self.model.branch_diagnostics_available:
            metric_res['branch_diagnostics_available'] = False
            return metric_res
        if self.ablation in (
            'branch_evidence', 'identity_branch_evidence', 'calibrated_branch_evidence',
            'token_relation_evidence', 'domain_tensor_router',
            'domain_log_pool',
            'probability_evidence_router',
            'logit_evidence_router',
            'uncertainty_evidence_router',
            'probability_router_with_signal_control',
            'evidence_router',
        ):
            labels_tensor = torch.tensor(label, dtype=torch.float32)
            view_scores = torch.stack([
                torch.tensor(values, dtype=torch.float32)
                for values in view_predictions
            ], dim=1).clamp(1e-6, 1.0 - 1e-6)
            weights_tensor = torch.tensor(
                decision_weights, dtype=torch.float32
            )
            per_view_loss = F.binary_cross_entropy(
                view_scores,
                labels_tensor.unsqueeze(1).expand_as(view_scores),
                reduction='none',
            )
            metric_res['routing_match'] = float(
                (
                    torch.argmax(weights_tensor, dim=1)
                    == torch.argmin(per_view_loss, dim=1)
                ).float().mean()
            )
            if self.ablation in (
                'probability_router_with_signal_control',
                'evidence_router',
            ):
                gains_tensor = torch.tensor(
                    signal_gains, dtype=torch.float32
                )
                acceptance_tensor = torch.tensor(
                    dacg_acceptance, dtype=torch.float32
                )
                metric_res['signal_gain_abs_mean'] = float(
                    gains_tensor.abs().mean()
                )
                metric_res['signal_gain_mean_by_path'] = (
                    gains_tensor.mean(dim=0).tolist()
                )
                metric_res['dacg_acceptance_mean'] = float(
                    acceptance_tensor.mean()
                )
                metric_res['dacg_rejection_rate'] = float(
                    (acceptance_tensor < 0.999).float().mean()
                )
            view_correct = (
                (view_scores >= 0.5)
                == labels_tensor.unsqueeze(1).bool()
            )
            metric_res['oracle_accuracy'] = float(
                view_correct.any(dim=1).float().mean()
            )
            metric_res['text_accuracy'] = float(
                view_correct[:, 0].float().mean()
            )
            metric_res['image_accuracy'] = float(
                view_correct[:, 1].float().mean()
            )
            metric_res['fusion_accuracy'] = float(
                view_correct[:, 2].float().mean()
            )
        return metric_res
