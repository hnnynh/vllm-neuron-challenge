# SPDX-License-Identifier: Apache-2.0
"""
Qwen3Embedding BF16 Implementation
====================================

Annotated implementation of Qwen3Embedding for the Neuron backend.
Adapted from GPT-OSS BF16 model for dense (non-MoE) architecture.

Supported parallelism: TP, SP, DP.
Starting model: Qwen3Embedding-3.2-1B (dense, GQA 32Q/8KV, SiLU, tied embeddings).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Qwen3Embedding-specific. Change when porting.

PARALLELISM SHARDING:
  TP-only:  Attention heads sharded, MLP intermediate sharded
  TP+DP:    Independent replicas per DP rank, same sharding as TP-only
"""

import logging
import math

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from nkilib.core.utils.common_types import NormType
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
)
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import (
    FP8_CLAMP_MAX,
    validate_fp8_segmented_supported,
)
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding

from .config import Qwen3EmbeddingConfig
from .quantization import QuantScheme

logger = logging.getLogger(__name__)


# Reserved top-level attn_metadata key: forward_decode caches the layer-invariant
# decode mask here on layer 0 and reuses it, so the graph holds one gen_mask_tkg
# kernel. Safe because the sole full attn_metadata.items() walk (spec-decode slot
# correction) is a prologue that runs before any layer writes this key. Assumes
# one mask geometry across layers (true for qwen3_embedding's single full-attention group;
# a hybrid/SWA model with per-group block_size/window would need a per-group key).
_DECODE_MASK_CACHE_KEY = "_cached_decode_mask"


def _validate_dcp_prefill_kv_cache(
    k_cache: torch.Tensor, v_cache: torch.Tensor
) -> None:
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if k_cache.dtype in fp8_dtypes or v_cache.dtype in fp8_dtypes:
        raise ValueError(
            "Qwen3Embedding DCP prefill requires a non-FP8 KV cache because "
            "segmented_attention_cp does not support FP8 KV dequantization; "
            "use --kv-cache-dtype auto"
        )


def _validate_dcp_decode_sequence_length(dcp_size: int, sequence_length: int) -> None:
    if dcp_size > 1 and sequence_length != 1:
        raise ValueError(
            "Qwen3Embedding DCP decode supports exactly one token per request; "
            f"got {sequence_length}. Speculative decoding with DCP is not supported."
        )


def _shard_kv_by_dcp_rank(
    kv: torch.Tensor, dcp_size: int, dcp_rank: int, block_size: int
) -> torch.Tensor:
    """Slice a [n_heads, tokens, head_dim] K/V projection to this DCP rank's
    interleaved token shard (positions where (pos // block_size) % dcp_size ==
    dcp_rank), returned contiguous as [n_heads, tokens // dcp_size, head_dim]."""
    n_heads, tokens, head_dim = kv.shape
    return (
        kv.reshape(
            n_heads, tokens // (dcp_size * block_size), dcp_size, block_size, head_dim
        )[:, :, dcp_rank, :, :]
        .reshape(n_heads, -1, head_dim)
        .contiguous()
    )


def _segmented_attention_cp_prefill(
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_seq_len: torch.Tensor | None,
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    dcp_group: object,
    scale: float,
) -> torch.Tensor:
    """DCP prefill attention: AllGather head-sharded Q, attend against this
    rank's local KV shard, LSE-correct across the group, ReduceScatter heads
    back. prior_tokens is the cached prefix per rank (cached_seq_len // dcp_size)."""
    local_prior = cached_seq_len // dcp_size if cached_seq_len is not None else None
    return NF.segmented_attention_cp(
        q=q,
        k_local=k_local,
        v_local=v_local,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_table,
        prior_tokens=local_prior,
        block_size=block_size,
        cp_rank=torch.full((1, 1), dcp_rank, dtype=torch.int32, device=q.device),
        cp_world_size=dcp_size,
        cp_kv_cache_interleave_size=block_size,
        cp_group=dcp_group,
        scale=scale,
        tp_q=True,
        tp_out=True,
    )


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Qwen3Embedding uses standard RMSNorm (no hidden dim padding)
# =============================================================================
class Qwen3EmbeddingRMSNorm(nn.Module):
    """RMS Normalization.

    <-- MODEL-SPECIFIC: Standard RMSNorm without padding. Other models may use
    LayerNorm or RMSNorm with padded hidden dim.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (Qwen3Embedding RoPE)
# <-- MODEL-SPECIFIC: Qwen3Embedding uses Qwen3Embedding-style RoPE scaling
# =============================================================================


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """<-- MODEL-SPECIFIC: Interleaved RoPE (rotate_half style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    <-- MODEL-SPECIFIC: Qwen3Embedding uses rotate_half style (interleaved). GPT-OSS uses
    non-interleaved (split in half). The RoPE variant must match the checkpoint.

    Args:
        q: [Nh, T, Dh], k: [Nkv, T, Dh]
        cos: [T, Dh], sin: [T, Dh]

    Returns:
        Rotated (q, k) tensors
    """
    cos = cos.unsqueeze(0)  # [1, T, Dh]
    sin = sin.unsqueeze(0)  # [1, T, Dh]
    return _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)


class Qwen3EmbeddingRotaryEmbedding(nn.Module):
    """표준 RoPE (theta = rope_theta).

    항목 3 — Llama3 의 piecewise 주파수 스케일링은 Llama 3.1+ 전용이라 삭제했다.
    Qwen3 는 rope_scaling 이 null 인 표준 RoPE 를 쓴다.
    """

    def __init__(self, config: Qwen3EmbeddingConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        if config.rope_scaling is not None:
            raise NotImplementedError(
                f"rope_scaling={config.rope_scaling!r} is not supported."
            )

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings for given positions.

        Returns:
            cos, sin: both of shape [T, head_dim]
        """
        inv_freq = self._compute_inv_freq(device)  # [Hd/2]
        inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (GQA config, no bias, no sinks, no sliding window)
# =============================================================================


class Qwen3EmbeddingAttention(nn.Module):
    """Multi-head attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - KV heads are replicated when fewer than TP size (GQA)
    - Prefill: all-gather input → QKV proj → attention → O proj → reduce-scatter
    - Decode: fused megakernel with TP all-reduce

    <-- MODEL-SPECIFIC:
    - GQA with separate Q and KV head counts
    - No attention bias, no sinks, no sliding window
    - Qwen3Embedding RoPE (rotate_half style)
    """

    def __init__(self, config: Qwen3EmbeddingConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # 항목 5 — Qwen3 의 QK-norm. Llama 에는 없다.
        # hidden_size 가 아니라 head_dim 기준 per-head RMSNorm 이며,
        # projection 과 RoPE 사이에 적용된다.
        self.q_norm = Qwen3EmbeddingRMSNorm(
            config.head_dim, config.rms_norm_eps, config.torch_dtype
        )
        self.k_norm = Qwen3EmbeddingRMSNorm(
            config.head_dim, config.rms_norm_eps, config.torch_dtype
        )

        # >>> PARALLELISM: TP group setup <<<
        from vllm.distributed.parallel_state import get_dcp_group

        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_group = dcp_group if self.dcp_size > 1 else None

        # >>> PARALLELISM: DCP — prefill uses the SAME scheme as decode <<<
        # Both prefill and decode:
        #   - Shard Q/K/V/O weights across the FULL TP group (no duplicated
        #     Q heads). Each rank owns num_q_heads/TP heads + a replicated KV
        #     head shard.
        #   - Token-shard the KV cache across the DCP group (vllm `_DCP`),
        #     interleaved by cp_rank = tp_rank % dcp_size.
        #   - Compute attention by AllGathering Q across heads, attending the
        #     gathered Q against the local KV shard, LSE-correcting across the
        #     DCP group, and ReduceScattering heads back to their owner.
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # DCP rank: position within the DCP group = which interleaved token
        # slice this rank owns (matches decode: tp_rank % dcp_size).
        self.cp_chunk_idx = self.dcp_group.rank_in_group if self.dcp_size > 1 else 0

        # >>> PARALLELISM: Dependent DP setup (decode-only Q/O sharding across DP) <<<
        self.attention_dp_size = (
            config.neuron_config.attention_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_attention_dp_rank,
        )

        self.attention_dp_group = get_neuron_attention_dp_group()
        self.attention_dp_rank = get_neuron_attention_dp_rank()

        # >>> PARALLELISM: Attention TP supergroup (TP * attn_dp) <<<
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_tp_group,
        )

        self.attn_tp_group = get_neuron_attention_tp_group()

        # Effective sharding degree for Q/O (TP for standard, TP*DDP for attention DP)
        effective_q_shards = self.world_size * self.attention_dp_size

        # >>> PARALLELISM: Head sharding calculation <<<
        # Q heads per rank: sharded across TP*DDP for decode
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // effective_q_shards
        )

        # KV heads per rank: determines cache size (always full TP amount).
        # When num_kv_heads > TP and attention DP is enabled, KV weights are also sharded
        # across attention DP (fewer projected heads). The cache stores the gathered result
        # (full TP heads) so attention reads locally without a2a'ing prior context.
        self.kv_needs_a2a = (
            self.attention_dp_size > 1
            and self.num_key_value_heads > self.world_size
            and self.num_key_value_heads % effective_q_shards == 0
        )

        # Cache sizing: full TP heads (unchanged by attention DP)
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        # Weight sizing: may be smaller when KV is dependent-DP-sharded
        self.num_kv_heads_for_weight = (
            self.num_key_value_heads // effective_q_shards
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_kv_heads_for_weight
        )

        # Q/KV heads after all-to-all: each rank receives heads from all attention DP peers
        self.num_q_heads_after_a2a = (
            self.num_attention_heads_per_rank * self.attention_dp_size
        )
        self.num_kv_heads_after_a2a = (
            self.num_kv_heads_for_weight * self.attention_dp_size
            if self.kv_needs_a2a
            else self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_kv_heads_for_weight * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // effective_q_shards

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        # Packed FP8 K cache layout. Auto-detected in bind_kv_cache() from the
        # cache rank (packed K is one dim larger than V); see _write_kv_cache /
        # forward_decode for the swizzle handling.
        self.fp8_packed = False

        # KV cache quantization scales are set during weight loading.
        # We also need floats since tensor.item() causes a graph break,
        # and kernels currently use floats for perf reasons.
        self.register_buffer("k_scale", None, persistent=False)
        self.register_buffer("v_scale", None, persistent=False)

        self.k_scale_float = 1.0
        self.v_scale_float = 1.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach weight loaders for checkpoint → parameter transformation.

        >>> PARALLELISM: Weight loaders handle TP sharding of checkpoint tensors.
        <-- MODEL-SPECIFIC: Qwen3Embedding stores separate Q, K, V weights (no bias).

        With attention DP: Q/O are sharded across TP*attention DP (not just TP). KV may also
        be sharded across attention DP when num_kv_heads > TP (auto-detected via
        kv_needs_a2a). The effective rank for sharded weights is
        attention_dp_rank + tp_rank * attention_dp_size.
        """

        # Effective rank for weight sharding: when attention DP is enabled,
        # Q/O are sharded across TP * attention_dp using an interleaved rank.
        # When disabled (ddp_size=1), effective_rank == tp_rank (no-op).
        ddp = self.attention_dp_size
        effective_q_rank = self.attention_dp_rank + self.rank * ddp
        o_shard_size = (self.num_attention_heads * self.head_dim) // (
            self.world_size * ddp
        )

        # QKV: Q and KV need different shard ranks with attention DP
        # (Q uses effective_rank, KV uses tp_rank). The loader handles this internally.
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
                attention_dp_rank=self.attention_dp_rank,
                attention_dp_size=ddp,
                kv_sharded_across_attention_dp=self.kv_needs_a2a,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            with_rank_override(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=o_shard_size,
                    num_shards=self.world_size * ddp,
                    is_storage_transposed=True,
                ),
                rank=effective_q_rank,
            ),
        )

    # ── Forward dispatch ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Dispatch to prefill or decode path based on metadata.

        >>> PARALLELISM: Dispatch logic <<<
        - Prefill: all-gather for SP before attention, reduce-scatter after
        - Decode: fused megakernel handles TP internally
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
            )
        else:
            # >>> PARALLELISM: All-gather from SP before attention <<<
            if self.world_size > 1:
                # >>> PARALLELISM: SP gather (tp_group): S/TP → S <<<
                # DCP uses the same full-S gather as standard prefill; the
                # per-rank KV-shard split happens inside forward_prefill so the
                # query layout matches decode (full Q, sharded KV).
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

            return self.forward_prefill(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
            )

    # ── Prefill path ─────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: full-sequence attention with flash attention.

        Pipeline:
        1. QKV projection          >>> PARALLELISM: TP sharded heads <<<
        2. RoPE                    <-- MODEL-SPECIFIC: Qwen3Embedding RoPE
        3. KV cache update         >>> PARALLELISM: per-rank cache <<<
        4. Flash attention          <-- MODEL-SPECIFIC: no sinks, no sliding window
        5. Output projection       >>> PARALLELISM: reduce-scatter after O proj <<<
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)
        if self.dcp_size > 1:
            _validate_dcp_prefill_kv_cache(self.k_cache, self.v_cache)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # ── Step 1: QKV Projection (with fused RoPE) ─────────────────────
        # <-- MODEL-SPECIFIC: Qwen3Embedding RoPE — fused into the qkv_proj kernel.
        cos, sin = position_embeddings

        # ── KV cache metadata (pulled up so segmented branch can fold the
        # post-RoPE K/V cache write into the qkv kernel). ──
        # >>> PARALLELISM: Cache is per-rank (TP sharded KV heads) <<<
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        kv_is_fp8 = self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

        def _write_kv_cache(k_to_cache, v_to_cache, slot_map):
            """Write K/V to paged cache using slot mapping."""
            blk_idx = slot_map // block_size
            pos_idx = slot_map % block_size
            num_slots = slot_map.shape[0]
            nkh = self.num_key_value_heads_per_rank

            if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                k_f = (
                    (k_to_cache.reshape(-1, self.head_dim) * self.k_scale)
                    .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                    .to(self.k_cache.dtype)
                )
                v_f = (
                    (v_to_cache.reshape(-1, self.head_dim) * self.v_scale)
                    .clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX)
                    .to(self.k_cache.dtype)
                )
            else:
                k_f = k_to_cache.reshape(-1, self.head_dim).to(self.k_cache.dtype)
                v_f = v_to_cache.reshape(-1, self.head_dim).to(self.v_cache.dtype)

            h_idx = torch.arange(
                nkh, dtype=torch.long, device=k_to_cache.device
            ).repeat_interleave(num_slots)
            index = (blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh))
            # V is never packed → scatters directly.
            self.v_cache.index_put_(index, v_f)
            if self.fp8_packed:
                # Packed K cache [blocks, Nkh, block_size // 2, Dh, 2]: the
                # swizzle interleaves adjacent token positions into the trailing
                # size-2 dim, so a per-token scatter isn't expressible directly.
                # Un-swizzle to the standard layout, scatter, then re-swizzle
                # back in place (matching the decode kernel's write).
                k_unpacked = _unswizzle_packed_k(self.k_cache)
                k_unpacked.index_put_(index, k_f)
                self.k_cache.copy_(_swizzle_packed_k(k_unpacked))
            else:
                self.k_cache.index_put_(index, k_f)

        # >>> PARALLELISM: Each rank computes only its sharded Q/K/V heads <<<
        # Phase 2 — QKV projection + QK-norm + RoPE 를 커널 하나로 융합한다.
        #
        # NF.qkv_proj 는 qk_norm_pre_rope_* 인자로 projection 과 RoPE **사이**의
        # QK-norm 을 커널 안에서 처리한다. 따라서 Qwen3 도 융합 경로를 쓸 수 있고,
        # 중간 결과(Q/K)를 HBM 에 내렸다 올리는 왕복이 사라진다.
        #
        # VLLM_NEURON_QWEN3_EMBEDDING_UNFUSED=1 이면 융합 없이 PyTorch 연산으로
        # 같은 순서를 재현한다 (A/B 대조용, Phase 1 경로).
        import os

        if os.environ.get("VLLM_NEURON_QWEN3_EMBEDDING_UNFUSED") == "1":
            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
            ).squeeze(0)

            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
            q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim)
            k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim)
            v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim)

            q = self.q_norm(q)
            k = self.k_norm(k)

            q = q.transpose(0, 1)  # [Nh_q, S, Dh]
            k = k.transpose(0, 1)  # [Nkv,  S, Dh]
            v = v.transpose(0, 1)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            # 융합 경로. cos/sin 은 커널이 요구하는 [1, S, Dh] 레이아웃으로 만든다.
            cos_cache = cos.unsqueeze(0)
            sin_cache = sin.unsqueeze(0)

            qkv = NF.qkv_proj(
                hidden=hidden_states.unsqueeze(0),
                qkv_weights=self.qkv_proj_weight,
                bias=None,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                num_q_heads=self.num_attention_heads_per_rank,
                num_kv_heads=self.num_key_value_heads_per_rank,
                d_head=self.head_dim,
                qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
                qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
                qk_norm_pre_rope_q_gamma=self.q_norm.weight.unsqueeze(0),
                qk_norm_pre_rope_k_gamma=self.k_norm.weight.unsqueeze(0),
            ).squeeze(0)

            q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
            q = q.view(
                tokens, self.num_attention_heads_per_rank, self.head_dim
            ).transpose(0, 1)
            k = k.view(
                tokens, self.num_key_value_heads_per_rank, self.head_dim
            ).transpose(0, 1)
            v = v.view(
                tokens, self.num_key_value_heads_per_rank, self.head_dim
            ).transpose(0, 1)

        # ── Step 4: Write KV cache + Attention ────────────────────────────
        # <-- MODEL-SPECIFIC: No sinks, no sliding window (standard causal attention)
        # 단일 prefill 경로. DCP(context parallel)·segmented prefill 은 융합 qkv 커널을
        # 전제로 하므로 이 포팅에서는 지원하지 않는다. 진입 시 명시적으로 거부한다.
        if self.dcp_size > 1:
            raise NotImplementedError(
                "DCP (context parallel) prefill is not supported by the "
                "Qwen3-Embedding port; the de-fused QKV path has no CP variant."
            )
        if kv_segment_size:
            raise NotImplementedError(
                "segmented prefill is not supported by the Qwen3-Embedding port "
                "(launch with --no-enable-prefix-caching)."
            )
        _write_kv_cache(k, v, slot_mapping)

        _write_kv_cache(k, v, slot_mapping)

        # Full prefill: standard flash attention
        k = k.repeat_interleave(self.num_key_value_groups, dim=0)
        v = v.repeat_interleave(self.num_key_value_groups, dim=0)

        q_flash = q.transpose(1, 2)  # [Nh, Dh, T]
        k_flash = k.transpose(1, 2)  # [Nh, Dh, T]
        v_flash = v  # [Nh, T, Dh]

        attn_output = NF.flash_attention(
            q_flash,
            k_flash,
            v_flash,
            scale=self.scaling,
            tp_q=False,
            tp_out=True,
        )  # [Nh, Dh, T]

        # ── Step 5: Output Projection ────────────────────────────────────
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode path ──────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode: fused megakernel for single-token generation.

        >>> PARALLELISM: The megakernel handles TP internally. <<<
        >>> PARALLELISM: With attention DP, Q/O are sharded across TP*attention DP. <<<

        <-- MODEL-SPECIFIC: No sink tokens, no sliding window, no attention bias.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        # B_local is from metadata (per-DP-rank batch size).
        # Caller ensures input is gathered to B_local * attn_dp via _dp_transition.
        B_local = block_table.shape[0]
        B = B_local * self.attention_dp_size
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode
        _validate_dcp_decode_sequence_length(self.dcp_size, S_decode)

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        # Prepare RoPE for megakernel format: [T, Hd] → [Hd/2, B, S]
        # With attention DP, RoPE is applied after the a2a on local batch only
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B_local, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        # <-- MODEL-SPECIFIC: Standard causal mask, no sliding window
        # DCP decode engages whenever dcp_size > 1 — prefill now writes the
        # same token-sharded (interleaved) cache, so decode always reads it the
        # DCP way (gather Q heads + LSE correction).
        dcp_decode_active = self.dcp_size > 1
        # With DCP decode, Q is gathered across DCP group → more heads in attention
        mask_q_heads = self.num_q_heads_after_a2a * (
            self.dcp_size if dcp_decode_active else 1
        )
        # With DCP decode, cache is interleaved → use flat slot-based mask.
        # Compute local filled count. The active token (current decode step)
        # is placed at slot s_prior-1 by stage 7b on ALL ranks. To avoid
        # double-counting in LSE correction, only the owning rank includes it
        # by setting local_filled = S_ctx (covering the last slot).
        local_filled = None
        dcp_active_mask = None
        if dcp_decode_active:
            # Per-request prior length: each line's own decode position, not the
            # batch-max cached_seq_len (which over-admits stale KV for shorter
            # lines at bs>1). Mirrors model_static_fp8.forward_decode.
            n = positions.view(B_local, S_decode)[:, 0:1].to(torch.float32)
            W = self.dcp_size
            R = self.dcp_group.rank_in_group
            vblock = block_size * W

            full_vblocks = torch.div(n, vblock, rounding_mode="trunc")
            remaining = n - full_vblocks * vblock
            local_filled = full_vblocks * block_size + torch.clamp(
                remaining - R * block_size,
                min=0,
                max=block_size,
            )

            owner = torch.div(remaining, block_size, rounding_mode="trunc") % W
            dcp_active_mask = (owner == R).float()

        pos_ids = positions.view(1, B_local * S_decode)
        # Layer-invariant here too: mask_q_heads/local_filled/dcp_active_mask are
        # __init__ constants + per-request positions. Cache/reuse (see key def).
        attention_mask = attn_metadata.get(_DECODE_MASK_CACHE_KEY)
        if attention_mask is None:
            attention_mask = NF.gen_attention_decode_mask(
                pos_ids=pos_ids.to(torch.float32),
                bs=B_local,
                q_head=mask_q_heads,
                s_active=S_decode,
                s_prior=S_ctx,
                start_pos=None,
                block_len=block_size,
                local_filled_slots=local_filled,
                dcp_active_mask=dcp_active_mask if dcp_decode_active else None,
            )
            attn_metadata[_DECODE_MASK_CACHE_KEY] = attention_mask

        active_blocks_table = block_table

        # Packed FP8 is selected by server config (neuron_config.fp8_packed_kv):
        # when enabled, the runner allocates the swizzled K layout and
        # bind_kv_cache sets self.fp8_packed, so the decode kernel always reads
        # the packed cache directly. There is no per-bucket auto-adaptation —
        # the layout is fixed by config, and a packed config must use a bucket
        # geometry the kernel can pack (even decode batch >= 2 so the resized
        # block_len stays >= 2).

        # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
        # In-kernel KV cache update (update_cache=True): the kernel writes K/V
        # in place and the FX aliasing pass threads the write back to
        # self.k_cache/self.v_cache (including across the recurrent passes of
        # Eagle3 fused propose). With update_cache=True the API returns only the
        # attention output.
        output = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            # 항목 5 — QK-norm. prefill 과 같은 gamma 를 넘겨야 결과가 일치한다.
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            # Kernel currently requires fusing the K scale dequantization into softmax scale for KV quantization
            softmax_scale=self.scaling / self.k_scale_float,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot_mapping.view(B_local, S_decode).to(torch.uint32),
            fp8_packed=self.fp8_packed,
            # Kernel currently requires fusing the V scale dequantization into W_out for KV quantization
            W_out=self.o_proj_weight / self.v_scale_float,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
            k_scale=self.k_scale
            if self.k_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            else None,
            v_scale=self.v_scale
            if self.v_cache.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
            else None,
            attention_dp=self.attention_dp_size,
            attention_dp_group=self.attention_dp_group.device_group
            if self.attention_dp_group
            else None,
            attention_dp_rank=self.attention_dp_rank,
            kv_needs_a2a=self.kv_needs_a2a,
            dcp_size=self.dcp_size if dcp_decode_active else 1,
            dcp_group=self.dcp_group if dcp_decode_active else None,
        )

        # >>> PARALLELISM: Sum O-proj partials across TP * attn_dp supergroup <<<
        self.attn_tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MLP (Dense)
# Mixed: PARALLELISM (TP intermediate sharding, SP collectives) +
#        MODEL-SPECIFIC (SiLU activation, gate/up/down structure)
# =============================================================================


class Qwen3EmbeddingMLP(nn.Module):
    """Dense MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden → intermediate/TP per rank
    - down_proj: intermediate/TP → hidden per rank
    - Prefill: all-gather before → compute → reduce-scatter after (SP)
    - Decode: compute → all-reduce

    With mlp_dp_size > 1, intermediate is sharded across TP * mlp_dp_size.

    <-- MODEL-SPECIFIC:
    - SiLU activation (gate * silu(up))
    - No bias on any projection
    """

    def __init__(self, config: Qwen3EmbeddingConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: MLP TP group (TP * mlp_dp_size) + DP column <<<
        self.mlp_dp_size = (
            config.neuron_config.mlp_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_mlp_tp_group,
            get_neuron_mlp_dp_group,
        )

        mlp_tp_group = get_neuron_mlp_tp_group()
        self.mlp_tp_group = mlp_tp_group
        self.mlp_tp_size = mlp_tp_group.world_size
        self.mlp_tp_rank = mlp_tp_group.rank_in_group
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        self.hidden_size = config.hidden_size
        # >>> PARALLELISM: Intermediate dim sharded across TP * mlp_dp_size <<<
        self.intermediate_size_per_rank = config.intermediate_size // self.mlp_tp_size

        # <-- MODEL-SPECIFIC: Separate gate and up projections (SwiGLU pattern)
        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        self._setup_weight_loaders(config)

    def _setup_weight_loaders(self, config):
        """>>> PARALLELISM: TP (and MLP DP) sharding of MLP weights. <<<"""
        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.mlp_tp_size,
            is_storage_transposed=True,
        )

        gate_up_loader = with_rank_override(gate_up_loader, rank=self.mlp_tp_rank)
        down_loader = with_rank_override(down_loader, rank=self.mlp_tp_rank)

        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """
        >>> PARALLELISM: SP all-gather/reduce-scatter for prefill,
        TP supergroup all-reduce for decode. Caller handles DP batch transitions. <<<
        """
        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # <-- MODEL-SPECIFIC: SiLU gated MLP (gate * silu(up))
        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        # >>> PARALLELISM: Combine TP (+ MLP DP) shards <<<
        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            self.mlp_tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# Mixed: PARALLELISM (SP residual handling) +
#        MODEL-SPECIFIC (pre-attention norm, residual connections)
# =============================================================================


def _dp_transition(
    x: torch.Tensor,
    current_group,
    target_group,
    dim: int = 0,
) -> torch.Tensor:
    """Transition tensor between DP gathered states.

    Args:
        x: Input tensor, gathered over current_group.world_size DP ranks along dim.
        current_group: GroupCoordinator for the current DP column group.
        target_group: GroupCoordinator for the target DP column group.
        dim: Batch dimension to gather/slice along.

    Returns:
        Tensor at target gathered state.
    """
    current_dp = current_group.world_size
    target_dp = target_group.world_size
    current_dp = current_group.world_size
    target_dp = target_group.world_size
    if current_dp == target_dp:
        return x
    if current_dp > target_dp:
        # Down: slice to keep this rank's target sub-group batches.
        per_dp = x.shape[dim] // current_dp
        start = (current_group.rank_in_group // target_dp) * target_dp * per_dp
        return x.narrow(dim, start, target_dp * per_dp)
    # Up: go through local first to avoid duplicates from overlapping gathered state,
    # then all-gather to target.
    per_dp = x.shape[dim] // current_dp
    x = x.narrow(dim, current_group.rank_in_group * per_dp, per_dp)
    return target_group.all_gather(x, dim=dim)


class Qwen3EmbeddingDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Architecture (MODEL-SPECIFIC):
        hidden_states → RMSNorm → Attention → residual → RMSNorm → MLP → residual

    Parallelism (TP + SP):
        - Input arrives in SP layout (T/world_size tokens per rank) during prefill
        - Decode: _dp_transition handles batch state between modules
        - Residual connection operates at whatever gathered state the module outputs
    """

    def __init__(self, config: Qwen3EmbeddingConfig, layer_idx: int):
        super().__init__()
        # <-- MODEL-SPECIFIC: Pre-attention and pre-MLP RMSNorm
        self.input_layernorm = Qwen3EmbeddingRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3EmbeddingRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        # <-- QUANTIZATION: pick attention/MLP implementations via the
        # quant_spec dispatcher. ``quant_spec=None`` returns the bf16
        # classes so this call site is byte-identical to the pre-dispatch
        # behaviour.
        from .quantization import resolve_attention_mlp_classes

        attn_cls, mlp_cls = resolve_attention_mlp_classes(config.quant_spec, layer_idx)
        self.self_attn = attn_cls(config, layer_idx=layer_idx)
        self.mlp = mlp_cls(config)

        # Record the MLP scheme + RMSNorm eps so the prefill path can
        # branch on the decoder-layer contract (quant MLPs fuse the
        # post-attention RMSNorm internally) without re-resolving.
        self._mlp_scheme = (
            config.quant_spec.get_scheme(layer_idx, prefix="mlp")
            if config.quant_spec is not None
            else QuantScheme.NONE
        )
        self.rms_norm_eps = config.rms_norm_eps

        self.layer_idx = layer_idx

        # >>> PARALLELISM: DP sizes for batch state transitions <<<
        nc = config.neuron_config
        self.attn_dp = nc.attention_dp_size if nc else 1
        self.mlp_dp = nc.mlp_dp_size if nc else 1

        # DP column groups for transitions
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_attention_dp_group,
            get_neuron_mlp_dp_group,
        )

        self.attn_dp_group = get_neuron_attention_dp_group()
        self.mlp_dp_group = get_neuron_mlp_dp_group()

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        if not is_decode:
            return self._forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata
            )

        # ── Decode: batch state transitions between modules ──
        # Input arrives at mlp_dp (from previous layer's MLP or embedding transition).

        # Transition mlp_dp → attn_dp
        hidden_states = _dp_transition(
            hidden_states, self.mlp_dp_group, self.attn_dp_group
        )

        # ── Self Attention ───────────────────────────────────────────────
        residual = hidden_states
        # <-- QUANTIZATION: FP8 attention fuses pre-attention RMSNorm
        # with static activation quant on decode via rmsnorm_X_enabled=True.
        if self._mlp_scheme is QuantScheme.FP8_STATIC_PER_TENSOR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                ln_w=self.input_layernorm.weight,
                eps=self.rms_norm_eps,
            )
        else:
            # <-- MODEL-SPECIFIC: Pre-attention RMSNorm
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )
        # Attention outputs at attn_dp (supergroup all-reduce, stays gathered)

        # Residual add at attn_dp, then transition once to mlp_dp
        hidden_states = residual + hidden_states
        hidden_states = _dp_transition(
            hidden_states, self.attn_dp_group, self.mlp_dp_group
        )

        # ── MLP Feed-Forward ─────────────────────────────────────────────
        residual = hidden_states
        # <-- QUANTIZATION: quant MLPs fuse the pre-MLP RMSNorm with the
        # static activation quant on decode via ``norm_type=RMS_NORM``
        # inside :func:`NF.mlp`. The bf16 path keeps the explicit norm
        # call; the FP8 path must skip it here and forward ``ln_w``/``eps``.
        if self._mlp_scheme is QuantScheme.FP8_STATIC_PER_TENSOR:
            hidden_states = self.mlp(
                hidden_states,
                is_prefill=False,
                ln_w=self.post_attention_layernorm.weight,
                eps=self.rms_norm_eps,
            )
        else:
            # <-- MODEL-SPECIFIC: Pre-MLP RMSNorm
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, is_prefill=False)
        # MLP outputs at mlp_dp (supergroup all-reduce, stays gathered)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        # ── Self Attention ───────────────────────────────────────────────
        residual = hidden_states
        if self._mlp_scheme is QuantScheme.FP8_STATIC_PER_TENSOR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                ln_w=self.input_layernorm.weight,
                eps=self.rms_norm_eps,
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )
        hidden_states = residual + hidden_states

        # ── MLP Feed-Forward ─────────────────────────────────────────────
        residual = hidden_states
        # <-- QUANTIZATION: quant MLPs own the post-attention RMSNorm on
        # prefill (they fuse it with the static activation quant via
        # ``NF.rmsnorm_quant``). The bf16 path keeps the explicit norm call.
        if self._mlp_scheme is QuantScheme.FP8_STATIC_PER_TENSOR:
            hidden_states = self.mlp(
                hidden_states,
                is_prefill=True,
                ln_w=self.post_attention_layernorm.weight,
                eps=self.rms_norm_eps,
            )
        else:
            # <-- MODEL-SPECIFIC: Pre-MLP RMSNorm
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, is_prefill=True)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# Mixed: PARALLELISM (SP chunking / gathering) +
#        MODEL-SPECIFIC (embedding, layer stack, final norm)
# =============================================================================


class Qwen3EmbeddingModel(nn.Module):
    """Qwen3Embedding transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill:
    - After embedding: chunk sequence across TP ranks (each gets T/world_size)
    - After all layers: all-gather to reconstruct full sequence
    During decode:
    - No SP; all ranks process all tokens
    """

    def __init__(self, config: Qwen3EmbeddingConfig):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Embedding TP group (TP * embedding_dp_size) + DP column <<<
        self.embedding_dp_size = (
            config.neuron_config.embedding_dp_size if config.neuron_config else 1
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_embedding_tp_group,
            get_neuron_embedding_dp_group,
            get_neuron_mlp_dp_group,
        )

        emb_tp_group = get_neuron_embedding_tp_group()
        emb_device_group = emb_tp_group.device_group
        self.embedding_dp_group = get_neuron_embedding_dp_group()
        self.embedding_tp_rank = emb_tp_group.rank_in_group
        self.mlp_dp_group = get_neuron_mlp_dp_group()

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=emb_device_group,
        )

        # <-- MODEL-SPECIFIC: Stack of decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3EmbeddingDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final RMSNorm
        self.norm = Qwen3EmbeddingRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3EmbeddingRotaryEmbedding(config)

        # 항목 7 — vocab 151665 는 TP=4 로 나누어떨어지지 않는다 (37916.25).
        # pad_shard 가 없으면 마지막 랭크의 체크포인트 슬라이스가 shard_size 에 못 미쳐
        # 파라미터가 meta 로 남고 .to(device) 에서 죽는다. Llama 템플릿에 이 인자가 없는
        # 이유는 vocab 128256 이 TP 로 나눠떨어져 문제가 드러나지 않기 때문이다.
        emb_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.embed_tokens.vocab_size_per_rank,
            num_shards=self.embed_tokens.tp_size,
            is_storage_transposed=False,
            pad_shard=True,
        )
        emb_loader = with_rank_override(emb_loader, rank=self.embedding_tp_rank)
        set_weight_loader(self.embed_tokens.weight, emb_loader)

        # Eagle3 speculative decoding: layer indices whose hidden states
        # are collected for the draft model.  Empty until the drafter sets them.
        self.aux_hidden_state_layers = []

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: Embedding DP — gather input_ids (tiny, just token IDs) <<<
        if not is_prefill and self.embedding_dp_size > 1:
            input_ids = self.embedding_dp_group.all_gather(input_ids, dim=0)

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        # Pass `rank` so vocab-shard offsets stay as runtime tensor ops; the
        # Python-int fallback would bake per-rank literals into each NEFF and
        # defeat cache reuse across ranks. In production, `rank` is always
        # supplied by the runner (see `neuron_model_runner.py`'s `rank=
        # self.rank_tensor` callsites); the `None` fallback exists only for
        # CPU-mode unit tests that drive this forward directly.
        emb_rank = rank
        if rank is not None and self.embedding_dp_size > 1:
            emb_rank = rank + (self.embedding_tp_rank - self.rank)
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=emb_rank
        )
        # Decode: embedding output at emb_dp gathered state

        # >>> PARALLELISM: Transition emb_dp → mlp_dp for decoder layers <<<
        if not is_prefill:
            hidden_states = _dp_transition(
                hidden_states, self.embedding_dp_group, self.mlp_dp_group
            )

        # >>> PARALLELISM: SP prompt-embed path <<<
        # Shard inputs_embeds/is_token_ids to match SP layout before merging.
        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        # <-- MODEL-SPECIFIC: Qwen3Embedding uses hidden dim as-is, so no prompt-embed
        # dim padding step is needed before merge.
        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        aux_hidden_states = []
        for idx, decoder_layer in enumerate(self.layers):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP - all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            # TODO: make Eagle3 drafter accept SP-partitioned aux states to avoid this all-gather
            aux_hidden_states = [
                self.tp_group.all_gather(aux, dim=0) for aux in aux_hidden_states
            ]

        return hidden_states, aux_hidden_states


# =============================================================================
# Section 7: Language Model Head
# Mixed: PARALLELISM (column-parallel LM head) +
#        MODEL-SPECIFIC (tied embeddings, sampling)
# =============================================================================
