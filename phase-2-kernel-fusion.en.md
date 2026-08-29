# Challenge — vLLM Neuron Model Onboarding Report

> Phase 2: Adding a custom NKI kernel
> Cutting HBM round trips by turning on the existing kernel's QK-norm/RoPE fusion

Phase 1 was about a model that behaves correctly. Phase 2 is optional, and what it deals with is speed.

| Item | Value |
|---|---|
| Target | The `qwen3_embedding` implementation from Phase 1 |
| Hardware | trn2.3xlarge (4 NeuronCores) · TP=4 · BF16 |
| Plugin | vllm-project/vllm-neuron `release-0.24.0.1.1.0` |
| What I did | Turned on the QK-norm fusion an existing kernel already had |
| Result | +4–5% saturated throughput (37.18 req/s), accuracy unchanged |


## The order the official doc prescribes

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ 1. Search nkilib     2. Use what you find     3. Otherwise write @nki.jit     │
│    (existing kernels)   (where this work ended)  (not needed this time)       │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Mini glossary

| Term | What it means |
|---|---|
| **nkilib** | The NKI kernels the plugin already ships. Step 1 of Phase 2 is digging through it |
| **fusion** | Doing several operations inside one kernel, so intermediate results never leave it |
| **turning fusion on** | Not writing a new kernel, but passing the optional arguments it already has, so work done outside moves inside |
| **`@nki.jit`** | The decorator you put on a hand-written NKI kernel. Step 3 of Phase 2 |


## 1. Searching the kernel library: it was already there

In Phase 1, I unfused the kernel because I read it as having no seam for QK-norm.
Going back to check that call, I read every argument `NF.qkv_proj` accepts. The arguments I needed
were sitting right there.

```
vllm_neuron/functional/attention/qkv.py:700-707

qk_norm_pre_rope_q_norm / k_norm   (NormType)      # QK-norm between projection and RoPE
qk_norm_pre_rope_eps
qk_norm_pre_rope_q_gamma / k_gamma
```

`vllm_neuron/model/qwen3_vl/model_bf16.py:419-423` in the same repository calls the kernel with
exactly this pattern. The llama3 template never touches those arguments, which is how I missed them.

So no new kernel. Phase 2 for this model turns into recovering the fusion Phase 1 gave up.

## 2. What changes

| | Phase 1 (unfused) | Phase 2 (fused) |
|---|---|---|
| QKV projection | kernel | kernel |
| QK-norm | **PyTorch op** | inside the kernel |
| RoPE | **PyTorch op** | inside the kernel |
| Intermediate Q/K | written to HBM, read back | stays inside the kernel |

The last row is the whole story. The unfused path builds Q and K, writes them to HBM, reads them
back to normalize. Then writes and reads again to apply RoPE. Fusing deletes those round trips.

## 3. Implementation

The fused path goes into `forward_prefill`.

```python
qkv = NF.qkv_proj(
    hidden=hidden_states.unsqueeze(0),
    qkv_weights=self.qkv_proj_weight, bias=None,
    cos_cache=cos.unsqueeze(0), sin_cache=sin.unsqueeze(0),
    num_q_heads=self.num_attention_heads_per_rank,
    num_kv_heads=self.num_key_value_heads_per_rank,
    d_head=self.head_dim,
    qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
    qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
    qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
    qk_norm_pre_rope_q_gamma=self.q_norm.weight.unsqueeze(0),
    qk_norm_pre_rope_k_gamma=self.k_norm.weight.unsqueeze(0),
).squeeze(0)
```

Phase 1's unfused path stays in the tree behind an environment variable. Set
`VLLM_NEURON_QWEN3_EMBEDDING_UNFUSED=1` and you get the old path back. That way A/B measurements
come out of the same binary.

## 4. Did accuracy survive

Change the order of operations and floating-point results shift a little. So I reran Stage 4.

### Level 2 (three-way prompt comparison)

The rule is the same as Phase 1. Neuron's error in the same decade as BF16's means healthy.

| Sentence | cos(Neuron, FP32) | maxΔ Neuron | maxΔ BF16 | Verdict |
|---|---|---|---|---|
| The cat sits on the mat | 0.99985552 | 2.322e-03 | 2.630e-03 | PASS |
| Stock markets fell sharply… | 0.99992463 | 2.740e-03 | 1.585e-03 | PASS |
| 고양이가 매트 위에 앉아 있다 | 0.99990664 | 1.810e-03 | 2.024e-03 | PASS |
| Photosynthesis converts light… | 0.99988360 | 2.158e-03 | 1.609e-03 | PASS |

Still inside BF16's error.

**LEVEL 2: PASS**

### Level 1 (STS12 task score)

```
unfused : 0.8139421814483995
fused   : 0.8139851321256291      difference 4.3e-05
```

That difference is two orders of magnitude below BF16's error (around 2e-03). The same score, for
any practical purpose.

## 5. How much faster

Both paths, same hardware, same settings, measured back to back.

| Condition | Fused | Unfused | Change |
|---|---|---|---|
| 129 tokens, concurrency 1 | 28.59 req/s | 28.94 req/s | -1.2% (measurement noise) |
| 129 tokens, concurrency 2 | **37.18 req/s** | 35.59 req/s | **+4.5%** |
| 129 tokens, concurrency 4 | 36.70 req/s | 35.27 req/s | +4.1% |
| 1901 tokens, concurrency 1 | **7.08 req/s** (141.2 ms) | 6.76 req/s (147.7 ms) | **+4.7%** |
| 1901 tokens, concurrency 2 | 7.61 req/s | 7.43 req/s | +2.4% |

Saturated throughput went up 4–5%.

Two things stand out.

1. **Single-request latency barely moves.** At concurrency 1 the hardware is idling, so cutting
   memory traffic shows up as nothing. The win arrives once the chip is full. Bandwidth you didn't
   spend turns into throughput only then.
2. **Long inputs get 4.7% too.** Fifteen times the tokens, roughly the same ratio of gain, because
   the round trips and the arithmetic both scale with token count.

## Conclusion

| Item | Result |
|---|---|
| New NKI kernel | **Not needed.** `NF.qkv_proj` already supports it |
| Accuracy | Held (Level 2 PASS, STS12 differs by 4.3e-05) |
| Throughput | **+4–5%** (37.18 req/s, 4,796 tok/s) |
| Side effect | Recovered the fused path Phase 1 abandoned |
