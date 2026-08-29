# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Embedding Config
======================

Qwen/Qwen3-Embedding-8B 의 config.json 실측값을 기본값으로 둔다.
Step 0 체크리스트의 항목 1(head_dim), 3(RoPE 표현), 4(MLP), 6(Llama 전용 필드 제거),
7(vocab·tie_word_embeddings)에 해당하는 변경이 여기에 모여 있다.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .quantization import QuantizationSpec


@dataclass
class Qwen3EmbeddingConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters
    # 항목 7: vocab 151665 (TP=4 로 나누어떨어지지 않는다 — embed_tokens 로더에서 pad_shard 로 처리)
    vocab_size: int = 151665
    hidden_size: int = 4096
    unpadded_hidden_size: int | None = None
    # 항목 4: MLP 는 SiLU 로 Llama 와 동일 — 값만 교체하면 NF.mlp 를 그대로 쓴다
    intermediate_size: int = 12288
    # 항목 2: 36개 레이어가 전부 같은 타입이라 레이어별 분기가 없다
    num_hidden_layers: int = 36
    # 항목 1: GQA 형태 동일(32/8). head_dim 만 128 로 교체
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    # 항목 5: RMSNorm 타입·위치는 동일, eps 만 다르다
    rms_norm_eps: float = 1e-6
    # 항목 3: RoPE 표현 전환. Llama 의 rope_parameters dict(piecewise 스케일링용) 대신
    # 스칼라 theta + rope_scaling 이다.
    rope_theta: float = 1000000.0
    rope_scaling: dict | None = None
    # 항목 7: weight sharing 없음
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    # Quantization spec parsed from the HuggingFace ``quantization_config``
    # (populated by :meth:`from_configs`). ``None`` means "not quantized".
    # Modeling code should query this via
    # ``quant_spec.get_scheme_for_module(prefix)`` to decide weight dtypes
    # and kernel calls on a per-module basis.
    quant_spec: QuantizationSpec | None = field(default=None)

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.unpadded_hidden_size is None:
            self.unpadded_hidden_size = self.hidden_size
        # transformers v5 는 config 를 정규화하며 rope_theta 를 rope_scaling dict 안으로 옮긴다:
        #   rope_scaling = {"rope_theta": 1e6, "rope_type": "default"}
        # 체크포인트 원본의 rope_scaling 은 null 이므로 이 형태는 표준 RoPE 그대로다.
        # 흡수하고 theta 만 꺼낸다. 그 밖의 rope_type 은 미구현이라 거부한다.
        if isinstance(self.rope_scaling, dict):
            if self.rope_scaling.get("rope_type", "default") in (None, "default"):
                if "rope_theta" in self.rope_scaling:
                    self.rope_theta = float(self.rope_scaling["rope_theta"])
                self.rope_scaling = None
        if self.rope_scaling is not None:
            raise NotImplementedError(
                f"rope_scaling={self.rope_scaling!r} is not supported; "
                "only standard RoPE (rope_type=default) is implemented."
            )

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        # Parse optional quantization config. Absent => not quantized.
        # Present and recognized => attach a QuantizationSpec.
        # Present but unsupported => raise (surfaced to the user, not silently
        # treated as bf16).
        filtered_dict["quant_spec"] = QuantizationSpec.from_hf_quantization_config(
            config_dict.get("quantization_config")
        )

        return cls(**filtered_dict)
