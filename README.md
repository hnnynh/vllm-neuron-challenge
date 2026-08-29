# Neuron Onboarding Challenge

Onboarding models into the [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron)
plugin on AWS Trainium, following the [AWS model onboarding
guide](https://awslabs.github.io/accelerated-compute-tutorials/aws-ai-chip/inference/vllm/model-onboarding).
One phase per model.

## Phase 1 — Qwen3-Embedding-8B

`trn2.3xlarge` (4 NeuronCore) · plugin `release-0.24.0.1.1.0` · official Neuron DLC

Ported from the Llama template. Accuracy matches the recipe baseline; throughput saturates at
concurrency 2–4 (36.05 req/s).

- [phase-1.ko.md](phase-1.ko.md) — report (Korean)
- [phase-1.en.md](phase-1.en.md) — report (English)

## Phase 2 — Kernel fusion (optional)

Same setup, targeting the Phase 1 `qwen3_embedding` implementation.

`NF.qkv_proj` already had QK-norm fusion arguments, so no hand-written NKI kernel was needed.
Turning the fusion on recovered +4–5% saturated throughput (37.18 req/s) with accuracy held.

- [phase-2.ko.md](phase-2.ko.md) — report (Korean)
- [phase-2.en.md](phase-2.en.md) — report (English)
