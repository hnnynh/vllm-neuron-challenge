# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Embedding model (pooling / embedding runner).

Step 0 체크리스트 항목 7(lm_head 제거 -> last-token pooling + L2)과
항목 8(DispatchPooler LAST + L2 요구)에 해당한다.

    생성:   backbone -> [T, H] -> index_select + lm_head -> [B, vocab]
    임베딩: backbone -> [T, H]                            (그대로 반환)

반환된 [T, H] post-norm 은닉상태를 NeuronModelRunner 의 ``_pool`` 경로가 받아
LAST-token gather + L2 normalize 를 수행한다.

체크포인트 사실 (텐서 398개 전수 조회):
  - ``model.`` 접두사 없음 -> 매핑의 체크포인트 쪽 키는 접두사를 뺀다
  - ``lm_head`` 텐서 자체가 부재 (항목 7 의 tie_word_embeddings=False 와 정합)
  - ``layers.N.self_attn.{q,k}_norm.weight`` 존재 -> per-head 라 TP 샤딩하지 않는다
"""

import logging

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

from .config import Qwen3EmbeddingConfig
from .model import Qwen3EmbeddingModel

logger = logging.getLogger(__name__)


class Qwen3EmbeddingForEmbedding(nn.Module):
    """Qwen3-Embedding 백본 + pooler 헤드 (lm_head 없음)."""

    is_pooling_model = True

    def __init__(self, config: Qwen3EmbeddingConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3EmbeddingModel(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.pooler import DispatchPooler

        vllm_config = get_current_vllm_config()

        # 임베딩은 prefill 전용이라 양자화 어텐션이 요구하는 k_scale/v_scale 을 로드하지
        # 않는다. fp8 KV 캐시를 받으면 어텐션 깊은 곳에서 죽으므로 먼저 거부한다.
        cache_dtype = getattr(vllm_config.cache_config, "cache_dtype", None)
        if (cache_dtype or "").startswith("fp8"):
            raise ValueError(
                f"kv_cache_dtype={cache_dtype!r} is not supported for the "
                "Qwen3-Embedding pooling model; use the default (auto/bf16)."
            )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None, (
            "pooler_config is None — Qwen3EmbeddingForEmbedding requires the "
            "pooling runner (launch with --runner pooling)."
        )
        # 항목 8 — LAST-token pooling + L2 normalize (1_Pooling/config.json 과 일치)
        self.pooler = DispatchPooler.for_embedding(pooler_config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """평탄화된 [T, H] post-norm 은닉상태를 그대로 반환한다."""
        hidden_states, _ = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )
        return hidden_states

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Qwen3EmbeddingConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV Cache ──────────────────────────────────────────────────────────
    # 임베딩은 prefill 전용이라 KV 캐시를 되읽지 않는다. 두 훅은 NeuronModelRunner 가
    # 캐시 구성/바인딩 시 무조건 호출하기 때문에 존재한다.

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,   # 항목 1 — 슬라이딩 윈도우 없음
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight Loading ────────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """HF 체크포인트에서 백본 가중치를 로드한다 (lm_head 없음)."""
        tp_rank = self.rank
        tp_size = self.world_size

        mappings = {}
        # 체크포인트에는 "model." 접두사가 없다 (398개 텐서 전수 확인).
        mappings["model.embed_tokens.weight"] = "embed_tokens.weight"
        mappings["model.norm.weight"] = "norm.weight"

        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            ckpt = f"layers.{layer_id}"

            mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{ckpt}.self_attn.q_proj.weight",
                f"{ckpt}.self_attn.k_proj.weight",
                f"{ckpt}.self_attn.v_proj.weight",
            ]
            mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{ckpt}.self_attn.o_proj.weight"
            )
            # 항목 5 — QK-norm 가중치. per-head 라 TP 샤딩하지 않는다.
            mappings[f"{prefix}.self_attn.q_norm.weight"] = (
                f"{ckpt}.self_attn.q_norm.weight"
            )
            mappings[f"{prefix}.self_attn.k_norm.weight"] = (
                f"{ckpt}.self_attn.k_norm.weight"
            )
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{ckpt}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{ckpt}.post_attention_layernorm.weight"
            )
            # 항목 4 — MLP 는 Llama 와 동일 구조 (gate/up/down, SiLU)
            mappings[f"{prefix}.mlp.gate_proj_weight"] = f"{ckpt}.mlp.gate_proj.weight"
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{ckpt}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = f"{ckpt}.mlp.down_proj.weight"

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        load_result = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        )
        rank_sharded = load_result.state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
