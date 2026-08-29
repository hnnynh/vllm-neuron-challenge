# Neuron Onboarding Challenge

A challenge exercise: onboard a model into the
[vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron) plugin on AWS Trainium,
following the [AWS model onboarding
guide](https://awslabs.github.io/accelerated-compute-tutorials/aws-ai-chip/inference/vllm/model-onboarding).
One phase per model, written up as a report.

## Phase 1 — Qwen3-Embedding-8B

[Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), ported from the Llama
template. The repository already ships an official Qwen3-Embedding implementation; the challenge
asks for the copy-and-adapt path, so that one was left untouched and used only as an oracle for
the accuracy comparison.

`trn2.3xlarge` (4 NeuronCore) · TP=4 · BF16 · plugin `release-0.24.0.1.1.0` · official Neuron DLC

### Approach

Copied the Llama template (`vllm_neuron/model/llama3/`) into `qwen3_embedding/`, then worked
through an 8-item config/module diff. Two changes did the real work.

- **Unfused RoPE.** The template fuses RoPE into the `NF.qkv_proj` kernel. Qwen3 needs QK-norm
  between projection and RoPE, so the fusion had to go. Unfusing kills prefix caching. Embedding
  is prefill-only, so nothing was actually lost.
- **`pad_shard=True`** on the embed_tokens loader. Vocab is 151665, which TP=4 does not divide.

### Results

Startup was clean. ~3m30s to compile and serve. Smoke test passed (dim 4096, ‖v‖ = 1.000001).

| Task | This port | Recipe baseline |
|---|---|---|
| STS12 (Spearman) | 0.8139 | 0.8639 |
| SciFact (NDCG@10) | 0.7906 | 0.7839 |
| NFCorpus (NDCG@10) | 0.4101 | 0.4143 |

The STS12 gap comes from the evaluation conditions. Run the built-in official implementation under
the same setup and it lands on the same value to 16 decimal places. The port is not the problem.

Throughput saturates at concurrency 2–4 and stays there: **36.05 req/s, ~4,650 prompt tok/s**.
Batching buys nothing. Embedding work is all prefill, so there is nothing to overlap.

### Running

```bash
export VLLM_NEURON_QWEN3_EMBEDDING=1
export NEURON_RT_NUM_CORES=4
export NEURON_SKIP_EFA_AFFINITY=1     # trn2.3xlarge has no EFA
vllm serve Qwen/Qwen3-Embedding-8B --runner pooling \
  --hf-overrides '{"architectures": ["Qwen3EmbeddingForEmbedding"]}' \
  --tensor-parallel-size 4 --max-model-len 2048 \
  --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --no-enable-prefix-caching --port 8000
```

The built-in qwen3 factory already owns `Qwen3ForCausalLM`. So the port registers as
`Qwen3EmbeddingForEmbedding`, and you select it with `--hf-overrides`.

Development container. The DLC installs the plugin non-editable, so mount the source over it:

```bash
docker exec vllm-dev bash -c 'pip uninstall -y vllm-neuron && cd /workspace/src && pip install -e . --no-deps'
```

Do not drop `--no-deps`. Without it, pip tears apart the version set the DLC pins. And the
container needs `--device /dev/neuron0`, or the plugin quietly skips registration.

Unsupported: prefix caching, FP8/MXFP4, DCP.

## Reports

- [phase-1.ko.md](phase-1.ko.md): Phase 1 report (Korean) — Step 0 architecture diff through Stage 5 benchmark
- [phase-1.en.md](phase-1.en.md): Phase 1 report (English), same content
