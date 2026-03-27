"""
This file keeps the router logic but removes internal model part. This is used to help understand the router design.
This is not an official Adobe product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import nn


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention."""
    if n_rep == 1:
        return x
    num_tokens, num_kv_heads, head_dim = x.shape
    x = x[:, :, None, :].expand(num_tokens, num_kv_heads, n_rep, head_dim)
    return x.reshape(num_tokens, num_kv_heads * n_rep, head_dim)


def expert_selection_per_sequence(
    scores: torch.Tensor,
    valid_seqlens: list[int] | None,
    topk: int,
    n_routed_experts: int,
) -> torch.Tensor:
    """Select expert-assigned tokens independently for each sequence."""
    if valid_seqlens is None:
        valid_seqlens = [scores.size(0)]

    sequence_scores = scores.split(valid_seqlens, dim=0)
    chosen_indices = []
    offset = 0

    for seq_len, seq_score in zip(valid_seqlens, sequence_scores):
        local_topk = int(math.floor(topk * seq_len / n_routed_experts + 0.5))
        local_topk = max(local_topk, 1)
        _, local_indices = torch.topk(seq_score, k=local_topk, dim=0)
        chosen_indices.append(local_indices + offset)
        offset += seq_len

    return torch.cat(chosen_indices, dim=0)


class RMSNorm(nn.Module):
    """Minimal RMSNorm implementation used by the public router."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_norm.to(dtype=x.dtype)) * self.weight


@dataclass
class AttentionGuidedGateConfig:
    dim: int = 192
    attention_hidden_dim: int = 2048
    n_routed_experts: int = 2
    n_activated_experts: int = 1
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: str = "softmax"
    route_scale: float = 1.0
    with_bias: bool = False
    gate_type: Literal["token_choice", "expert_choice"] = "token_choice"
    moe_warm_up_steps: int = 0
    target_token_budget: int = 256
    use_attention_guidance: bool = True
    attention_guidance_timestep_threshold: int = 606
    n_heads: int = 16
    n_kv_heads: int = 16
    qkv_bias: bool = True
    use_qknorm: bool = True
    selected_attention_heads: list[int] | None = None
    router_mlp_hidden_dim: int = 1024
    router_mlp_num_layers: int = 3
    router_mlp_activation: Literal["relu", "gelu", "silu"] = "gelu"
    router_mlp_dropout: float = 0.1


class AttentionGuidedGate(nn.Module):
    """Attention-guided router with an MLP gate and a lightweight attention probe."""

    def __init__(self, config: AttentionGuidedGateConfig):
        super().__init__()
        if config.gate_type != "token_choice":
            raise ValueError("AttentionGuidedGate only supports token_choice routing.")
        if config.n_activated_experts != 1:
            raise ValueError("This implementation assumes top-1 routing.")
        if config.attention_hidden_dim % config.n_heads != 0:
            raise ValueError("attention_hidden_dim must be divisible by n_heads.")
        if config.router_mlp_num_layers < 2:
            raise ValueError("router_mlp_num_layers must be at least 2.")

        self.config = config
        self.dim = config.dim
        self.attention_hidden_dim = config.attention_hidden_dim
        self.n_routed_experts = config.n_routed_experts
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.gate_type = config.gate_type
        self.moe_warm_up_steps = config.moe_warm_up_steps
        self.target_token_budget = config.target_token_budget
        self.use_attention_guidance = config.use_attention_guidance
        self.attention_guidance_timestep_threshold = config.attention_guidance_timestep_threshold
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.attention_hidden_dim // config.n_heads
        self.qkv_bias = config.qkv_bias
        self.use_qknorm = config.use_qknorm
        self.selected_attention_heads = config.selected_attention_heads

        self.attention_norm = RMSNorm(config.attention_hidden_dim)
        self.wq = nn.Linear(config.attention_hidden_dim, config.n_heads * self.head_dim, bias=config.qkv_bias)
        self.wk = nn.Linear(config.attention_hidden_dim, config.n_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.wv = nn.Linear(config.attention_hidden_dim, config.n_kv_heads * self.head_dim, bias=config.qkv_bias)

        if self.use_qknorm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        activation_map: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        self.router_mlp_activation = activation_map[config.router_mlp_activation]
        self.router_mlp_dropout = (
            nn.Dropout(config.router_mlp_dropout) if config.router_mlp_dropout > 0 else nn.Identity()
        )

        mlp_layers: list[nn.Module] = [nn.Linear(config.dim, config.router_mlp_hidden_dim)]
        for _ in range(config.router_mlp_num_layers - 2):
            mlp_layers.append(nn.Linear(config.router_mlp_hidden_dim, config.router_mlp_hidden_dim))
        mlp_layers.append(nn.Linear(config.router_mlp_hidden_dim, config.n_routed_experts))
        self.router_mlp_layers = nn.ModuleList(mlp_layers)

        self._attention_importance: torch.Tensor | None = None
        self._soft_routing_weights: torch.Tensor | None = None
        self._per_head_attention: torch.Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for index, layer in enumerate(self.router_mlp_layers):
            gain = 0.1 if index == len(self.router_mlp_layers) - 1 else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.normal_(self.wq.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wk.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wv.weight, mean=0.0, std=0.02)
        if self.qkv_bias:
            nn.init.zeros_(self.wq.bias)
            nn.init.zeros_(self.wk.bias)
            nn.init.zeros_(self.wv.bias)

    def load_attention_probe(
        self,
        attention_norm_state: dict[str, torch.Tensor],
        q_proj_state: dict[str, torch.Tensor],
        k_proj_state: dict[str, torch.Tensor],
        v_proj_state: dict[str, torch.Tensor],
        q_norm_state: dict[str, torch.Tensor] | None = None,
        k_norm_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Load attention probe parameters from any compatible backbone."""
        self.attention_norm.load_state_dict(attention_norm_state)
        self.wq.load_state_dict(q_proj_state)
        self.wk.load_state_dict(k_proj_state)
        self.wv.load_state_dict(v_proj_state)
        if self.use_qknorm:
            if q_norm_state is None or k_norm_state is None:
                raise ValueError("q_norm_state and k_norm_state are required when use_qknorm=True.")
            self.q_norm.load_state_dict(q_norm_state)
            self.k_norm.load_state_dict(k_norm_state)

    def _project_qk(self, x_projected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x_projected.shape[0]
        x_norm = self.attention_norm(x_projected)
        xq = self.wq(x_norm).view(num_tokens, -1, self.head_dim)
        xk = self.wk(x_norm).view(num_tokens, -1, self.head_dim)

        if self.use_qknorm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        n_rep = xq.shape[1] // xk.shape[1]
        if n_rep > 1:
            xk = repeat_kv(xk, n_rep)
        return xq, xk

    @staticmethod
    def _normalize_per_sample(values: torch.Tensor, batch_size: int, spatial_regions: int) -> torch.Tensor:
        values = values.view(batch_size, spatial_regions, *values.shape[1:])
        flat_dim = values.shape[2:]
        normalized = torch.zeros_like(values)

        if flat_dim:
            for batch_idx in range(batch_size):
                for channel_idx in range(values.shape[-1]):
                    current = values[batch_idx, :, channel_idx]
                    min_val = current.min()
                    max_val = current.max()
                    normalized[batch_idx, :, channel_idx] = (current - min_val) / (max_val - min_val + 1e-8)
        else:
            for batch_idx in range(batch_size):
                current = values[batch_idx]
                min_val = current.min()
                max_val = current.max()
                normalized[batch_idx] = (current - min_val) / (max_val - min_val + 1e-8)

        return normalized.view(batch_size * spatial_regions, *flat_dim)

    def compute_attention_importance(
        self,
        x_projected: torch.Tensor,
        batch_info: dict[str, int] | None = None,
    ) -> torch.Tensor:
        """Compute a normalized attention-based importance score for each region."""
        with torch.no_grad():
            xq, xk = self._project_qk(x_projected)
            attn_scores = torch.matmul(xq.transpose(0, 1), xk.transpose(0, 1).transpose(-2, -1))
            attn_scores = attn_scores / math.sqrt(self.head_dim)
            attn_probs = F.softmax(attn_scores, dim=-1)

            if self.selected_attention_heads is not None:
                selected = torch.as_tensor(self.selected_attention_heads, device=attn_probs.device)
                attention_importance = attn_probs.index_select(0, selected).sum(dim=1).mean(dim=0)
            else:
                attention_importance = attn_probs.sum(dim=1).mean(dim=0)

            if batch_info is not None:
                attention_importance = self._normalize_per_sample(
                    attention_importance,
                    batch_size=batch_info["batch_size"],
                    spatial_regions=batch_info["spatial_regions"],
                )

        return attention_importance

    def compute_per_head_attention_importance(
        self,
        x_projected: torch.Tensor,
        batch_info: dict[str, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-head and averaged attention importance for analysis."""
        with torch.no_grad():
            xq, xk = self._project_qk(x_projected)
            attn_scores = torch.matmul(xq.transpose(0, 1), xk.transpose(0, 1).transpose(-2, -1))
            attn_scores = attn_scores / math.sqrt(self.head_dim)
            attn_probs = F.softmax(attn_scores, dim=-1)

            per_head_importance = attn_probs.sum(dim=1).transpose(0, 1)
            aggregated_importance = per_head_importance.mean(dim=1)

            if batch_info is not None:
                per_head_importance = self._normalize_per_sample(
                    per_head_importance,
                    batch_size=batch_info["batch_size"],
                    spatial_regions=batch_info["spatial_regions"],
                )
                aggregated_importance = self._normalize_per_sample(
                    aggregated_importance,
                    batch_size=batch_info["batch_size"],
                    spatial_regions=batch_info["spatial_regions"],
                )

        return per_head_importance, aggregated_importance

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.router_mlp_layers):
            x = layer(x)
            if index < len(self.router_mlp_layers) - 1:
                x = self.router_mlp_activation(x)
                x = self.router_mlp_dropout(x)
        return x

    def forward(
        self,
        x_raw: torch.Tensor,
        x_projected: torch.Tensor,
        valid_seqlens: list[int] | None = None,
        batch_info: dict[str, int] | None = None,
        timestep: float | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del valid_seqlens

        if self.use_attention_guidance and timestep is not None:
            self._attention_importance = self.compute_attention_importance(x_projected, batch_info=batch_info)
        else:
            self._attention_importance = None

        scores = self.forward_mlp(x_raw)

        if self.training:
            soft_weights = F.softmax(scores, dim=-1)
            indices = soft_weights.argmax(dim=-1, keepdim=True)
            hard_weights = torch.zeros_like(soft_weights).scatter(-1, indices, 1.0)
            ste_weights = (hard_weights - soft_weights).detach() + soft_weights
            weights = ste_weights.gather(dim=-1, index=indices)
            self._soft_routing_weights = soft_weights
        else:
            probs = F.softmax(scores, dim=-1, dtype=torch.float32)
            indices = probs.argmax(dim=-1, keepdim=True)
            weights = torch.ones_like(indices, dtype=scores.dtype)
            self._soft_routing_weights = probs

        weights = (weights * self.route_scale).type_as(x_raw)
        return weights, indices

    def compute_attention_similarity_loss(
        self,
        timestep: float | None = None,
        batch_info: dict[str, int] | None = None,
        loss_type: Literal["mse", "smooth_l1", "cosine", "hinge"] = "cosine",
        target_fine_ratio: float = 0.60,
        mask_temperature: float = 20.0,
    ) -> torch.Tensor:
        """Regularize routing so that important regions prefer finer stages."""
        if self._attention_importance is None or self._soft_routing_weights is None:
            return torch.zeros((), device=self.wq.weight.device)

        attention_importance = self._attention_importance.detach()
        soft_routing = self._soft_routing_weights
        num_tokens, num_stages = soft_routing.shape

        stage_indices = torch.arange(num_stages, device=soft_routing.device, dtype=soft_routing.dtype)
        expected_stage = soft_routing @ stage_indices
        fineness = 1.0 - expected_stage / (num_stages - 1 + 1e-8)
        target_stage = (1.0 - attention_importance) * (num_stages - 1)

        mse_loss = F.mse_loss(expected_stage, target_stage)
        smooth_l1_loss = F.smooth_l1_loss(expected_stage, target_stage)
        cosine_loss = 1.0 - F.cosine_similarity(
            expected_stage.view(1, -1),
            target_stage.view(1, -1),
            dim=1,
        ).squeeze(0)

        if batch_info is not None and num_tokens % batch_info["batch_size"] == 0:
            batch_size = batch_info["batch_size"]
        else:
            batch_size = 1
        tokens_per_sample = num_tokens // batch_size

        attention_per_sample = attention_importance.view(batch_size, tokens_per_sample)
        fineness_per_sample = fineness.view(batch_size, tokens_per_sample)
        threshold_q = 1.0 - target_fine_ratio

        foreground_weights = []
        background_weights = []
        for batch_idx in range(batch_size):
            threshold = torch.quantile(attention_per_sample[batch_idx], threshold_q)
            w_fg = torch.sigmoid((attention_per_sample[batch_idx] - threshold) * mask_temperature)
            foreground_weights.append(w_fg)
            background_weights.append(1.0 - w_fg)

        w_fg = torch.stack(foreground_weights, dim=0)
        w_bg = torch.stack(background_weights, dim=0)
        mu_fg = (w_fg * fineness_per_sample).sum() / w_fg.sum().clamp_min(1e-8)
        mu_bg = (w_bg * fineness_per_sample).sum() / w_bg.sum().clamp_min(1e-8)
        margin = max(0.05, 0.5 / (num_stages - 1 + 1e-8)) if num_stages <= 3 else 0.075
        hinge_loss = F.relu(margin - (mu_fg - mu_bg))

        selected_loss = {
            "mse": mse_loss,
            "smooth_l1": smooth_l1_loss,
            "cosine": cosine_loss,
            "hinge": hinge_loss,
        }[loss_type]

        if timestep is None:
            return selected_loss

        time_weight = max(0.0, 1.0 - timestep / self.attention_guidance_timestep_threshold)
        return time_weight * selected_loss


@dataclass
class GateConfig:
    dim: int = 4096
    n_routed_experts: int = 32
    n_activated_experts: int = 1
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    with_bias: bool = False
    gate_type: Literal["token_choice", "expert_choice"] = "token_choice"
    moe_warm_up_steps: int = 0
    target_token_budget: int = 256


class Gate(nn.Module):
    """Baseline router without attention guidance."""

    def __init__(self, config: GateConfig):
        super().__init__()
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        self.target_topk = config.n_activated_experts
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.with_bias = config.with_bias
        self.gate_type = config.gate_type
        self.moe_warm_up_steps = config.moe_warm_up_steps
        self.target_token_budget = config.target_token_budget

        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.dim))
        self.bias = nn.Parameter(torch.empty(config.n_routed_experts)) if config.with_bias else None

        if self.gate_type not in {"token_choice", "expert_choice"}:
            raise ValueError(f"Unsupported gate_type: {self.gate_type}")
        if self.gate_type == "expert_choice" and self.bias is not None:
            raise ValueError("Bias is not supported for expert_choice routing.")

        self._soft_routing_weights: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            bias = torch.zeros(self.n_routed_experts)
            bias[0] = 1.5
            self.bias.data.copy_(bias)

    def update_top_k(self, cur_train_steps: int | None) -> None:
        if self.moe_warm_up_steps <= 0 or cur_train_steps is None:
            return
        self.topk = max(
            self.target_topk,
            self.n_routed_experts
            - cur_train_steps * (self.n_routed_experts - self.target_topk) // self.moe_warm_up_steps,
        )

    @staticmethod
    def gumbel_softmax_pair(
        logits: torch.Tensor,
        temperature: float = 0.3,
        dim: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gumbels = -torch.empty_like(logits).exponential_().log()
        y_soft = F.softmax((logits + gumbels) / temperature, dim=dim)
        indices = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter(dim, indices, 1.0)
        return y_soft, y_hard, indices

    def token_choice_forward(
        self,
        x: torch.Tensor,
        valid_seqlens: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del valid_seqlens
        if self.topk != 1:
            raise ValueError("This public implementation assumes top-1 token-choice routing.")

        scores = F.linear(x, self.weight, self.bias)

        if self.training:
            y_soft, y_hard, indices = self.gumbel_softmax_pair(scores, temperature=0.3, dim=-1)
            ste_weights = (y_hard - y_soft).detach() + y_soft
            weights = ste_weights.gather(dim=-1, index=indices)
            self._soft_routing_weights = y_soft
        else:
            probs = F.softmax(scores, dim=-1, dtype=torch.float32)
            indices = probs.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(probs).scatter(-1, indices, 1.0)
            ste_weights = (y_hard - probs).detach() + probs
            weights = ste_weights.gather(dim=-1, index=indices)
            self._soft_routing_weights = probs

        return (weights * self.route_scale).type_as(x), indices

    def expert_choice_forward(
        self,
        x: torch.Tensor,
        valid_seqlens: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x.size(0)
        scores = F.linear(x, self.weight, self.bias)
        scores = scores.softmax(dim=-1, dtype=torch.float32) if self.score_func == "softmax" else scores.sigmoid()
        original_scores = scores

        if self.n_groups > 1:
            scores = scores.view(num_tokens, self.n_groups, -1)
            group_scores = scores.amax(dim=-1)
            group_indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, group_indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)

        indices = expert_selection_per_sequence(scores, valid_seqlens, self.topk, self.n_routed_experts)
        if self.score_func == "softmax":
            weights = original_scores.gather(0, indices)
        else:
            mask = torch.zeros_like(scores, dtype=torch.bool)
            for expert_idx in range(self.n_routed_experts):
                mask[indices[:, expert_idx], expert_idx] = True
            sums = (scores * mask).sum(dim=1, keepdim=True).clamp_min(1e-9)
            normalized_scores = torch.where(mask, scores / sums, scores.new_zeros(()))
            weights = normalized_scores.gather(0, indices)

        if weights.dtype != torch.bfloat16:
            weights = weights.to(torch.bfloat16)
        return (weights * self.route_scale).type_as(x), indices

    def forward(
        self,
        x: torch.Tensor,
        valid_seqlens: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gate_type == "token_choice":
            return self.token_choice_forward(x, valid_seqlens=valid_seqlens)
        return self.expert_choice_forward(x, valid_seqlens=valid_seqlens)


@dataclass
class ChunkRoutingOutput:
    weights: torch.Tensor
    indices: torch.Tensor
    soft_routing_weights: torch.Tensor | None = None
    attention_importance: torch.Tensor | None = None
    per_head_attention: torch.Tensor | None = None


def route_chunked_motion_regions(
    router: AttentionGuidedGate,
    raw_region_inputs: torch.Tensor,
    projected_region_fn: Callable[[torch.Tensor], torch.Tensor],
    routing_chunk_size: int,
    timestep: float | None = None,
    include_per_head_attention: bool = False,
) -> ChunkRoutingOutput:
    """Apply the router with the same chunk policy used in the experiment.

    Args:
        router: Attention-guided gate.
        raw_region_inputs: Tensor of shape [batch, frames, regions, raw_dim].
        projected_region_fn: Callable mapping a flattened tensor
            [batch * regions, raw_dim] to [batch * regions, attention_hidden_dim].
        routing_chunk_size: Number of frames per chunk.
        timestep: Diffusion timestep used for attention-loss weighting.
        include_per_head_attention: Whether to export per-head importance.

    Returns:
        ChunkRoutingOutput with routing decisions for all frames. The first frame
        of each chunk is treated as a key frame and forced to stage 0. The
        remaining frames in the chunk share one routing decision computed from
        the temporal average of their region features.
    """
    if raw_region_inputs.dim() != 4:
        raise ValueError("raw_region_inputs must have shape [batch, frames, regions, raw_dim].")
    if routing_chunk_size <= 0:
        raise ValueError("routing_chunk_size must be positive.")

    batch_size, num_frames, num_regions, raw_dim = raw_region_inputs.shape
    dtype = raw_region_inputs.dtype
    device = raw_region_inputs.device

    final_weights: list[torch.Tensor] = []
    final_indices: list[torch.Tensor] = []
    soft_weights_per_chunk: list[torch.Tensor] = []
    attention_per_chunk: list[torch.Tensor] = []
    per_head_per_chunk: list[torch.Tensor] = []

    for start_frame in range(0, num_frames, routing_chunk_size):
        end_frame = min(start_frame + routing_chunk_size, num_frames)
        num_motion_frames = max(0, end_frame - start_frame - 1)

        keyframe_weights = torch.ones(batch_size, num_regions, device=device, dtype=dtype)
        keyframe_indices = torch.zeros(batch_size, num_regions, device=device, dtype=torch.long)
        final_weights.append(keyframe_weights)
        final_indices.append(keyframe_indices)

        if num_motion_frames == 0:
            continue

        motion_average = raw_region_inputs[:, start_frame + 1 : end_frame].mean(dim=1)
        motion_average_flat = motion_average.reshape(batch_size * num_regions, raw_dim)
        projected_flat = projected_region_fn(motion_average_flat)
        batch_info = {"batch_size": batch_size, "spatial_regions": num_regions}

        motion_weights, motion_indices = router(
            x_raw=motion_average_flat,
            x_projected=projected_flat,
            batch_info=batch_info,
            timestep=timestep,
        )

        if router._soft_routing_weights is not None:
            soft_weights_per_chunk.append(router._soft_routing_weights.view(batch_size, num_regions, -1))
        if router._attention_importance is not None:
            attention_per_chunk.append(router._attention_importance.view(batch_size, num_regions))
        if include_per_head_attention:
            per_head, _ = router.compute_per_head_attention_importance(projected_flat, batch_info=batch_info)
            per_head_per_chunk.append(per_head.view(batch_size, num_regions, -1))

        motion_weights = motion_weights.view(batch_size, num_regions)
        motion_indices = motion_indices.view(batch_size, num_regions)
        expanded_weights = motion_weights[:, None, :].expand(-1, num_motion_frames, -1).reshape(batch_size, -1)
        expanded_indices = motion_indices[:, None, :].expand(-1, num_motion_frames, -1).reshape(batch_size, -1)

        final_weights.append(expanded_weights)
        final_indices.append(expanded_indices)

    weights = torch.cat(final_weights, dim=1)
    indices = torch.cat(final_indices, dim=1)

    soft_routing_weights = None
    if soft_weights_per_chunk:
        soft_routing_weights = torch.cat(soft_weights_per_chunk, dim=1).reshape(-1, router.n_routed_experts)
        router._soft_routing_weights = soft_routing_weights

    attention_importance = None
    if attention_per_chunk:
        attention_importance = torch.cat(attention_per_chunk, dim=1).reshape(-1)
        router._attention_importance = attention_importance

    per_head_attention = None
    if per_head_per_chunk:
        per_head_attention = torch.stack(per_head_per_chunk, dim=1)
        per_head_attention = per_head_attention.view(batch_size, -1, per_head_attention.size(-1)).reshape(
            -1, per_head_attention.size(-1)
        )
        router._per_head_attention = per_head_attention

    return ChunkRoutingOutput(
        weights=weights,
        indices=indices,
        soft_routing_weights=soft_routing_weights,
        attention_importance=attention_importance,
        per_head_attention=per_head_attention,
    )

