# Challenge — vLLM Neuron Model Onboarding Report

> Phase 1: "Swapping in" a model with NF functions 
> Getting Qwen3-Embedding-8B running on vLLM Neuron

The model gets assembled from the **NF functions** the plugin ships.

| Item | Value |
|---|---|
| Model | Qwen/Qwen3-Embedding-8B (8B-parameter embedding model) |
| Hardware | trn2.3xlarge — AWS Trainium2 chip, 4 NeuronCores |
| Base | vLLM Inference NeuronX **official DLC** (container image) `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04` |
| Plugin | vllm-project/vllm-neuron `release-0.24.0.1.1.0` |
| dtype | BF16 |
| TP | 4 |


## The whole thing in 5 stages

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ 1. Implement    2. Register      3. Compile &    4. Validate    5. Benchmark  │
│ (config/model/   (ModelRegistry)   Smoke Test     Accuracy       & Tune       │
│  factory/weights)                                                             │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Mini glossary

| Term | What it means |
|---|---|
| **NF function** | The building blocks the plugin gives you (`NF.mlp` and friends). Under the hood they call Neuron-optimized kernels |
| **NKI** | Neuron Kernel Interface. The low-level API you use when writing kernels by hand (that's Phase 2 territory) |
| **decode** | The stage that spits out tokens one at a time. **Embedding models only ever prefill** |
| **TP (tensor parallel)** | Splitting the model across cores. Here, four ways across four cores |
| **RoPE** | Encodes token position by rotating the vectors |
| **QK-norm** | A layer that normalizes attention's Q and K. Qwen3 has it, Llama doesn't |
| **GQA** | Attention where several Q heads share a K/V head |
| **pooling** | Collapsing per-token vectors into one sentence vector. This model uses the last token |
| **compile / NEFF** | Neuron compiles the graph ahead of execution. NEFF is what comes out |

## Environment

| Item | State |
|---|---|
| Instance | trn2.3xlarge ✓ (12 vCPU, 124GB RAM, `/dev/neuron0`) |
| Disk | 258GB free ✓ (fits the ~16GB 8B model plus the DLC image) |
| Docker | Installed, 0 images — DLC not pulled yet |
| NeuronCore | Idle ✓ |


## Setting up the DLC

The DLC arrives with the plugin installed non-editable. To add a model you mount your source over it.

```bash
docker run -d --name vllm-dev \
  --device /dev/neuron0 --network host --shm-size 8g \
  -v ~/vllm-neuron/repo:/workspace/src \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash <IMG> -c 'sleep infinity'

docker exec vllm-dev bash -c 'pip uninstall -y vllm-neuron && cd /workspace/src && pip install -e . --no-deps'
```

| Flag | What breaks without it |
|---|---|
| `--no-deps` | pip re-resolves dependencies and disturbs the pinned DLC combination (torch 2.11.0 / torch-xla 2.11.0 / libtorch-neuronx-lite / neuronx-cc 2.27.5334.0 / transformers 5.15.0) |
| `--device /dev/neuron0` | The plugin says "No Neuron devices found" and skips registration |


## Stage 0: Architecture diff

`~/workspace/compare.py` (stdlib AST parsing + json, no weight loading — zero memory)
Script source: [AWS vLLM Neuron model onboarding guide](https://awslabs.github.io/accelerated-compute-tutorials/aws-ai-chip/inference/vllm/model-onboarding/#_1)

### 1. config.json

| Field | Meaning | meta-llama/Llama-3.1-8B | Qwen/Qwen3-Embedding-8B |
|---|---|---|---|
| architectures | Model class name — the lookup key for vLLM's `ModelRegistry` | `['LlamaForCausalLM']` | `['Qwen3ForCausalLM']` |
| model_type | Key that maps to the HF config class | `llama` | `qwen3` |
| bos_token_id | Beginning-of-sequence token ID | 128000 | 151643 |
| eos_token_id | End-of-sequence token ID | 128001 | 151645 |
| vocab_size | Vocabulary size = number of embed_tokens rows | 128256 | 151665 |
| num_hidden_layers | Decoder layer count | 32 | 36 |
| intermediate_size | MLP intermediate dim (gate/up output) | 14336 | 12288 |
| max_position_embeddings | Max context length | 131072 | 40960 |
| rms_norm_eps | RMSNorm denominator stabilizer | 1e-05 | 1e-06 |
| rope_parameters | RoPE settings — even the representation differs | `{'factor': 8.0, 'low_freq_fa…}` (piecewise scaling) | `{'rope_theta': 1000000, 'rop…}` (standard, theta=1e6) |
| layer_types | Per-layer attention type array | — | `['full_attention', …]` (all 36 identical) |
| max_window_layers | How many layers use sliding window | — | 36 |
| sliding_window | Sliding window width | — | `None` (unused) |
| use_sliding_window | Sliding window on/off | — | `False` |
| mlp_bias | Whether MLP Linear has bias | `False` | — (field absent = no bias) |
| pretraining_tp | TP degree at training time (irrelevant for inference) | 1 | — |

### 2. Model structure

| Module | Meaning | LlamaForCausalLM | Qwen3ForCausalLM |
|---|---|---|---|
| embed_tokens | Token ID → vector lookup table | `Embedding(128256, 4096)` | `Embedding(151665, 4096)` |
| layers | Decoder block stack | `32 x LlamaDecoderLayer` | `36 x Qwen3DecoderLayer` |
| q_proj / o_proj | Builds Q / projects attention output back to hidden | `4096 → 4096`, bias=False | `4096 → 4096`, bias=False |
| k_proj / v_proj | Builds K and V — 1024 = 8 KV heads × 128, **GQA (32 Q heads share 4:1)** | `4096 → 1024`, bias=False | `4096 → 1024`, bias=False |
| **q_norm / k_norm** | **Per-head normalization of Q and K over head_dim (128) — keeps the dot-product scale from blowing up. Sits between projection and RoPE** | **absent** | **`RMSNorm((128,), eps=1e-06)`** |
| gate_proj / up_proj | MLP expansion — gate is the gating signal, up is the value | `4096 → 14336` | `4096 → 12288` |
| down_proj | MLP contraction — back to hidden dim | `14336 → 4096` | `12288 → 4096` |
| act_fn | Activation applied to gate (SwiGLU) | `SiLUActivation()` | `SiLUActivation()` |
| input / post_attention_layernorm | The two pre-norms inside a block — before attention, before MLP | `RMSNorm((4096,), eps=1e-05)` | `RMSNorm((4096,), eps=1e-06)` |
| norm | Final normalization after the last block — **this is where the embedding comes out** | `RMSNorm((4096,), eps=1e-05)` | `RMSNorm((4096,), eps=1e-06)` |
| rotary_emb | Builds the RoPE cos/sin cache | `LlamaRotaryEmbedding()` (piecewise scaling) | `Qwen3RotaryEmbedding()` (standard, theta=1e6) |
| lm_head | hidden → vocab logit projection (for generation) | `4096 → 128256`, bias=False | `4096 → 151665`, bias=False — **tensor missing from the checkpoint, to be removed** |

The class layout maps 1:1 (RMSNorm / RotaryEmbedding / Attention / MLP / DecoderLayer / Model).
Grepping the modeling sources says **the one structural difference is QK-norm** — `q_norm`/`k_norm`
appear 0 times in llama3, and 7 and 6 times in qwen3.

### 3. What the embedding side needs (HF metadata)

| Source | Value | Implication |
|---|---|---|
| `modules.json` | Transformer → Pooling → Normalize | Three-stage pipeline |
| `1_Pooling/config.json` | `pooling_mode_lasttoken: true`, dim 4096 | Last-token hidden state + L2 normalize |
| Checkpoint | `lm_head` tensor absent | No generation path needed |


### 4. Deriving the work items (against the 8-item checklist)

Comparing what we observed in Qwen3-Embedding-8B against the Llama template (`vllm_neuron/model/llama3/`).

| # | Item | Fields checked | Observation | Work it implies |
|---|---|---|---|---|
| 1 | Attention style | `num_attention_heads` 32=32, `num_key_value_heads` 8=8, `layer_types` all `full_attention`, `sliding_window` `None` / `use_sliding_window` `False` | Same GQA shape, no sliding window | **No change to the attention forward** — just swap the template's default `head_dim` (64) for 128 |
| 2 | Heterogeneous layers | `layer_types` array, no `per_layer_config` | All 36 layers the same type | **No per-layer branching, no branching in weight init** |
| 3 | Position encoding | `rope_theta` 1e6, `rope_scaling` `None`, no `partial_rotary_factor` | Llama uses a `rope_parameters` dict (`rope_type=default`, `theta=5e5`) plus piecewise scaling | Switch `config.py` RoPE representation to `rope_theta` + `rope_scaling`, drop the piecewise scaling from RotaryEmbedding. **Unfuse RoPE from `NF.qkv_proj` to open a seam for QK-norm** (the crux of Stage 1) |
| 4 | MLP / activation | `hidden_act` `silu`, `intermediate_size` 12288 | Llama is SiLU too, same structure | **`NF.mlp` works as-is** — only the config numbers change |
| 5 | Normalization | `rms_norm_eps` 1e-6, `q_norm`/`k_norm` in the modeling source (llama3 0× / qwen3 7× and 6×) | Same RMSNorm type and placement, **QK-norm is the only addition** | Add per-head RMSNorm over `head_dim` to attention (not hidden_size), wire gamma into the decode path |
| 6 | Config shape | Top-level fields only, no `text_config`/`vision_config` | No nesting | **No change to `from_configs()` parsing** — only drop the Llama-only fields (`draft_vocab_size`, `norm_before_fc`, `norm_before_residual`) and the nested Eagle3 block |
| 7 | Embedding | `tie_word_embeddings` `False`, `vocab_size` 151665 | No weight sharing, `lm_head` tensor missing from the checkpoint | Remove `lm_head`, route to last-token pooling + L2 normalize. **151665 doesn't divide by TP=4 (37916.25), so embed_tokens loader needs `pad_shard=True`** |
| 8 | Special features | No `final_logit_softcapping`, no `enable_moe_block`, no `vision_config` | Nothing extra architecturally | **Nothing.** What §3 asks for instead is `DispatchPooler` LAST + L2 |

## Stage 1: Implement

I copied `vllm_neuron/model/llama3/` to `qwen3_embedding/`, leaving out what an embedding model never touches (`eagle3_model.py` for speculative decoding, the FP8 files), then worked through the eight items above.

### The change that mattered most — unfusing RoPE

For speed, the Llama template **folds RoPE into the QKV kernel**. Qwen3 needs QK-norm to land
**between the QKV projection and RoPE**, and a fused kernel leaves no room to slip anything in there.

So I unfused it and rebuilt the order outside the kernel.

```python
qkv = NF.qkv_proj(hidden=..., qkv_weights=..., bias=None).squeeze(0)   # ① QKV projection (no RoPE)
q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
q = self.q_norm(q); k = self.k_norm(k)                                  # ② QK-norm
q, k = apply_rotary_pos_emb(q, k, cos, sin)                             # ③ RoPE
```

**What that costs** — the KV cache writes, chunked prefill, the FP8 path, and context parallelism
that used to happen inside the kernel. An embedding model only prefills, and the recipe states FP8
isn't supported, so nothing is actually lost. Without chunked prefill, though, **prefix caching
(reusing a shared prefix) is off the table.** Turn it on anyway and you get an explicit error
rather than quietly wrong numbers.

### Two traps, defused early

Both turned up in Step 0 and got handled during implementation. Either one takes the server down at startup.

1. **How transformers 5.x tidies up config** — the library moves your `rope_theta` value inside a
   slot called `rope_scaling`. It looks like scaling is configured, but if `rope_type` is `default`
   it's plain standard RoPE. The code recognizes that shape and pulls the value back out.
2. **151665 doesn't divide by 4** (151665 ÷ 4 = 37916.25) — split the embedding table across four
   cores and the last core comes up short. `pad_shard=True` fills the gap.
   **Copy the Llama template as-is and you hit this every time** — Llama's vocab of 128256 divides
   cleanly by 4, so the option never appears in the template.

### Reading the actual checkpoint

I opened all 398 tensors in the weight files.

- The `lm_head` tensor (the output layer for next-word prediction) is **simply not there**, which matches an embedding model
- Keys carry no `model.` prefix, so the name mapping accounts for that
- `q_norm`/`k_norm` are present, and being per-head, they must not be sharded across cores

## Stage 2: Register

The built-in implementation already owns `Qwen3ForCausalLM`, so I registered under a different name
(`Qwen3EmbeddingForEmbedding`) and pointed at it with `--hf-overrides` at launch. It goes in two
places: vLLM's registry and the plugin's own list.

## Stage 3: Compile & smoke test

```bash
export VLLM_NEURON_QWEN3_EMBEDDING=1     # turn on model registration
export NEURON_SKIP_EFA_AFFINITY=1        # no EFA on this instance
vllm serve Qwen/Qwen3-Embedding-8B --runner pooling \
  --hf-overrides '{"architectures": ["Qwen3EmbeddingForEmbedding"]}' \
  --tensor-parallel-size 4 --max-model-len 2048 \
  --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --no-enable-prefix-caching --port 8000
```

**Zero startup failures.** Up in about 3 minutes 30 seconds, compilation included.

### What the smoke test showed

| Check | Result | Meaning |
|---|---|---|
| Vector length | **4096** | Matches the recipe |
| Vector magnitude | ‖v‖ = 1.000001 | L2 normalization is applied |
| Related vs unrelated sentence | 0.6822 > 0.2867 | It distinguishes meaning |
| English ↔ Korean, same meaning | **0.8144** | It catches meaning across languages |
| Response time (4 sentences) | 127 ms | — |

**Passed.**

## Stage 4: Validate accuracy (the 3-level framework)

| Level | What it looks at | What it compares against | Pass condition |
|---|---|---|---|
| **3 module** | Individual pieces (RMSNorm, MLP, …) | The HF reference implementation | Tensors match |
| **2 prompt** | Output for a single sentence | Three-way: HF FP32 / HF BF16 / Neuron | Neuron error ≈ BF16 error |
| **1 task** | Whole-model performance | Real benchmark scores | A bar I set myself |

> **One adaptation, because this is an embedding model.** Level 2 in the original framework compares
> a generative model's logits (next-word probabilities). This model produces no text, so I put the
> **final embedding vector** in that slot and ran the same three-way comparison.

I went in the order the doc recommends: **Level 3 → 2 → 1**.

---

### Level 3: module level

Each ported piece runs on **the real checkpoint weights** and gets checked against the HF reference.
Everything on CPU in FP32, which takes hardware error out of the picture and leaves only the question
of whether the math is identical.

| Piece | What the layer does | Max error | Verdict |
|---|---|---|---|
| RMSNorm (hidden_size) | Normalizes layer input/output | 0.000e+00 | PASS |
| **QK-norm (head_dim)** | **The layer only Qwen3 has — the heart of this port** | 0.000e+00 | PASS |
| RoPE (rotate_half) | Injects token position | 0.000e+00 | PASS |
| MLP (SwiGLU) | Feed-forward block | 0.000e+00 | PASS |

**Zero error everywhere — bit-identical.** QK-norm especially, since it's the part with no Llama
template to copy from. Had it been wrong, every later stage would have been meaningless. The formula,
the eps, and the axis it applies over (per-head across head_dim) all line up.

**LEVEL 3: PASS**

### Level 2: prompt level (three-way)

The same sentence, embedded three ways.

| Name | What it is | Role |
|---|---|---|
| `expected` | HF FP32 (CPU, 32-bit) | **The ground truth** |
| `baseline` | HF BF16 (CPU, 16-bit) | **The yardstick for acceptable error** |
| `target` | Neuron (our server) | The thing under test |

Slipping BF16 in there is the whole point. Computing in 16 bits **produces some error to begin with**,
just from precision. So Neuron's error shouldn't be measured against zero — it should be measured
**against BF16's error**.

> The rule — **Neuron error ≈ BF16 error means healthy. Neuron error >> BF16 means a bug.**

| Sentence | cos(Neuron, FP32) | cos(BF16, FP32) | maxΔ Neuron | maxΔ BF16 | Verdict |
|---|---|---|---|---|---|
| "The cat sits on the mat" | 0.99990443 | 0.99989652 | 2.231e-03 | 2.630e-03 | PASS |
| "Stock markets fell sharply…" | 0.99993960 | 0.99990832 | 2.738e-03 | 1.585e-03 | PASS |
| "고양이가 매트 위에 앉아 있다" | 0.99993650 | 0.99990324 | 1.530e-03 | 2.024e-03 | PASS |
| "Photosynthesis converts light…" | 0.99990419 | 0.99991308 | 1.544e-03 | 1.609e-03 | PASS |

**Neuron's error sits in the same decade as BF16's** (1.5–2.7e-03 against 1.6–2.6e-03). On two of the
sentences Neuron is actually the smaller one. Cosine similarity to the FP32 truth is **0.9999** on all
four. What's left isn't an implementation mistake, it's **the error 16-bit arithmetic comes with**.

**LEVEL 2: PASS**

### Level 1: task level

MTEB, on the three tasks the challenge specified. The bar is the official recipe's numbers.

| Task | What it measures | Ours | Recipe | Ratio |
|---|---|---|---|---|
| STS12 | How similar are two sentences | **0.8139** | 0.8639 | 94.2% |
| SciFact | Find the evidence document for a scientific claim | **0.7906** | 0.7839 | **100.9%** |
| NFCorpus | Find documents matching a medical query | **0.4101** | 0.4143 | 99.0% |

SciFact came in over the bar and the other two land at 94–99%.

Conditions — MTEB 2.20.3, server exposed as OpenAI-compatible `/v1/embeddings`. Two things swing the
numbers: **retrieval tasks need title and body concatenated**, and **the instruction goes on the query
side only** (per the model card; symmetric tasks like sentence similarity get none). NFCorpus has
documents past the 2048-token input limit, so truncation happens server-side.

**LEVEL 1: PASS** (my own bar — within 90% of the recipe)

### Why STS12 comes up 5% short

Level 1 was the only one that didn't land cleanly, so I split the cause. I ran **the built-in official
implementation under the identical server config and the identical eval code** (only four cores, so
I swapped servers and measured one after the other).

```
ours     : 0.8139421814483995
official : 0.8139421814483995
```

**Identical to 16 decimal places.** If the official implementation scores exactly the same, the gap
lives in **the scoring conditions**, not the model. Likely candidates: per-task instructions (not in
the materials I was given, and I wasn't going to invent them), MTEB version, dataset revision.

With Levels 3 and 2 passing, that conclusion follows — if the pieces are exact and per-sentence output
sits inside BF16's error, there's nowhere in the implementation for the gap to come from.

### 3-level summary

| Level | Result | Evidence |
|---|---|---|
| 3 module | **PASS** | RMSNorm, QK-norm, RoPE, MLP all at zero error |
| 2 prompt | **PASS** | Neuron error ≈ BF16 error, cos 0.9999 |
| 1 task | **PASS** | SciFact 100.9%, NFCorpus 99.0%, STS12 94.2% |
| Cross-check | — | Matches the official implementation to 16 decimal places |

## Stage 5 — Performance

An embedding model **never emits tokens one at a time.** One request means reading the input once, and
that's the end of it. Per-token metrics from generative serving don't mean anything here, so I measured
per-request latency and throughput.

| Concurrency | Requests/sec | Mean latency | p99 latency |
|---|---|---|---|
| 1 | 29.50 | 33.8 ms | 35.4 ms |
| 2 | 35.61 | 55.5 ms | 61.2 ms |
| **4** | **36.05** | 107.5 ms | 118.9 ms |
| 8 | 35.28 | 209.9 ms | 227.7 ms |

Three things stand out.

1. **It saturates somewhere between 2 and 4.** Going 1→2 buys 21%; 2→4 buys 1%, and 4→8 gives some
   back. Past that, latency just scales with the queue. Longer line, same throughput.
2. **Batching sentences into one request buys nothing.** At 8 sentences per request, embeddings per
   second still stops at 35.5. Every embedding is a single read, so there's no idle window to overlap
   into.
3. **The slow tail behaves.** p99 stays under 1.15× the median at every concurrency, which makes it
   predictable.

Peak throughput is **36 requests/sec, roughly 4,650 tokens/sec** (4 cores, 8B model, BF16).

---

## Conclusion

**Phase 1 goal — "a model that behaves correctly" — met**

| Criterion | Result |
|---|---|
| Does the server come up | Zero startup failures |
| Do the embeddings carry meaning | 0.81 cross-lingual, clean separation of related vs unrelated |
| Does accuracy reach the bar | All three levels PASS — zero module error, inside BF16's error, SciFact 100.9% |
| Is the implementation correct | Matches the official one to 16 decimal places |
| Is performance there | 36 requests/sec, with no custom kernels |
