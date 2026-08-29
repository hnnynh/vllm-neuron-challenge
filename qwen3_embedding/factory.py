# SPDX-License-Identifier: Apache-2.0
"""Factory for the Qwen3-Embedding pooling model."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3EmbeddingForEmbedding(nn.Module):
    """Factory that validates config and selects the appropriate Qwen3Embedding implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls to it.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        from .model_embedding import Qwen3EmbeddingForEmbedding as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported. Add rules as needed.

        Parses ``hf_config.quantization_config`` through
        :meth:`~vllm_neuron.model.qwen3_embedding.quantization.QuantizationSpec.from_hf_quantization_config`
        so unsupported quantization schemes fail fast — at model
        construction, with a clear message — instead of at the first
        forward call inside a kernel launch.

        The parse result is discarded; :class:`Qwen3EmbeddingConfig.from_configs`
        re-parses it and attaches the :class:`QuantizationSpec` to the
        config object. Re-parsing costs nothing (pure-Python dict walk)
        and keeps the factory stateless.
        """
        del neuron_config  # reserved for future validation rules
        from .quantization import QuantizationSpec

        quant_cfg = None
        if hf_config is not None:
            quant_cfg = getattr(hf_config, "quantization_config", None)
            # HuggingFace configs also store the dict under __dict__ when
            # the attribute wasn't declared on the pretrained class; fall
            # back to dict form so ModelOpt-injected configs are caught.
            if quant_cfg is None and hasattr(hf_config, "to_dict"):
                quant_cfg = hf_config.to_dict().get("quantization_config")
        # Raises ValueError on unsupported scheme / malformed config;
        # returns None when the checkpoint is unquantized.
        QuantizationSpec.from_hf_quantization_config(quant_cfg)

        # 항목 3 — rope_scaling 검증. transformers v5 정규화 형태
        # ({"rope_theta": …, "rope_type": "default"})는 표준 RoPE 이므로 통과시킨다.
        rope_scaling = None
        if hf_config is not None:
            rope_scaling = getattr(hf_config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            if rope_scaling.get("rope_type", "default") in (None, "default"):
                rope_scaling = None
        if rope_scaling:
            raise NotImplementedError(
                f"rope_scaling={rope_scaling!r} is not supported; only standard RoPE."
            )
