# Challenge — vLLM Neuron 모델 온보딩 리포트

> Phase 2: 커스텀 NKI 커널 추가
> 기존 커널의 QK-norm/RoPE 융합 경로 활성화로 HBM 왕복 제거

Phase 1 의 목표가 동작이 올바른 모델이었다면, Phase 2 는 선택 단계이고 여기서는 성능을 다룬다.

| 항목 | 값 |
|---|---|
| 대상 | Phase 1 에서 올린 `qwen3_embedding` 구현 |
| 장비 | trn2.3xlarge (NeuronCore 4개) · TP=4 · BF16 |
| 플러그인 | vllm-project/vllm-neuron `release-0.24.0.1.1.0` |
| 한 일 | 기존 커널의 QK-norm 융합 기능을 켰다 |
| 결과 | 포화 처리량 +4~5% (37.18 req/s), 정확도 유지 |


## 공식 문서가 정한 순서 (3단계)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ 1. nkilib 검색      2. 있으면 그대로 활용      3. 없으면 @nki.jit 로 직접 작성 │
│    (기존 커널)         (이번 작업이 여기서 끝)     (이번엔 불필요)             │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 용어 미니 사전

| 용어 | 뜻 |
|---|---|
| **nkilib** | 플러그인에 이미 들어 있는 NKI 커널 모음. Phase 2 의 1단계는 여기를 뒤지는 일이다 |
| **융합 (fusion)** | 여러 연산을 커널 하나 안에서 처리하는 것. 중간 결과를 메모리에 내리지 않는다 |
| **융합 기능을 켠다** | 커널이 이미 가진 선택 인자를 넘겨 밖에서 하던 연산을 안으로 들이는 것 |
| **`@nki.jit`** | NKI 커널을 직접 짤 때 붙이는 데코레이터. Phase 2 의 3단계 |


## 1. 커널 검색: 이미 있었다

Phase 1 에서는 "융합 커널에 QK-norm 이 끼어들 자리가 없다"고 보고 융합을 풀었다.
그 판단을 다시 확인하려고 `NF.qkv_proj` 의 인자를 끝까지 훑었더니, 필요한 인자가 그대로 있었다.

```
vllm_neuron/functional/attention/qkv.py:700-707

qk_norm_pre_rope_q_norm / k_norm   (NormType)      # projection 과 RoPE 사이의 QK-norm
qk_norm_pre_rope_eps
qk_norm_pre_rope_q_gamma / k_gamma
```

같은 저장소의 `vllm_neuron/model/qwen3_vl/model_bf16.py:419-423` 이 정확히 이 패턴으로 커널을 부른다.
llama3 템플릿이 이 인자를 한 번도 쓰지 않아서 놓친 것이다.

그래서 새 커널을 짤 필요가 없고 이 모델의 Phase 2 는 Phase 1 에서 포기했던 융합을 되찾는 작업이 된다.

## 2. 무엇이 달라지나

| | Phase 1 (비융합) | Phase 2 (융합) |
|---|---|---|
| QKV 계산 | 커널 | 커널 |
| QK-norm | **PyTorch 연산** | 커널 안 |
| RoPE | **PyTorch 연산** | 커널 안 |
| 중간 결과 Q/K | HBM 에 내렸다 다시 올림 | 커널 내부에 머무름 |

핵심은 마지막 줄. 비융합 경로는 Q·K 를 만들어 HBM 에 쓰고 다시 읽어서 정규화한 뒤,
또 쓰고 또 읽어서 RoPE 를 건다. 융합하면 이 왕복이 통째로 사라진다.

## 3. 구현

`forward_prefill` 에 융합 경로를 넣었다.

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

Phase 1 의 비융합 경로도 그대로 남겨 두고 환경변수로 갈랐다.
`VLLM_NEURON_QWEN3_EMBEDDING_UNFUSED=1` 을 주면 옛 경로로 돌아간다.
같은 바이너리에서 A/B 를 재기 위한 장치.

## 4. 정확도가 깨지지 않았나

연산 순서가 바뀌면 부동소수점 결과가 미세하게 달라지므로 Stage 4 의 검증을 다시 돌렸다.

### Level 2 (프롬프트 3자 비교)

판정 기준은 Phase 1 과 같다. Neuron 오차가 BF16 오차와 같은 자릿수면 정상.

| 문장 | cos(Neuron, FP32) | maxΔ Neuron | maxΔ BF16 | 판정 |
|---|---|---|---|---|
| The cat sits on the mat | 0.99985552 | 2.322e-03 | 2.630e-03 | PASS |
| Stock markets fell sharply… | 0.99992463 | 2.740e-03 | 1.585e-03 | PASS |
| 고양이가 매트 위에 앉아 있다 | 0.99990664 | 1.810e-03 | 2.024e-03 | PASS |
| Photosynthesis converts light… | 0.99988360 | 2.158e-03 | 1.609e-03 | PASS |

여전히 BF16 오차 범위 안이다.

**LEVEL 2: PASS**

### Level 1 (STS12 태스크 점수)

```
비융합 : 0.8139421814483995
융합   : 0.8139851321256291      차이 4.3e-05
```

차이가 BF16 오차(약 2e-03)보다 두 자릿수 작다. 점수는 실질적으로 같은 셈이다.

## 5. 얼마나 빨라졌나

같은 장비, 같은 설정에서 두 경로를 번갈아 측정했다.

| 조건 | 융합 | 비융합 | 변화 |
|---|---|---|---|
| 129 토큰, 동시성 1 | 28.59 req/s | 28.94 req/s | -1.2% (측정 노이즈) |
| 129 토큰, 동시성 2 | **37.18 req/s** | 35.59 req/s | **+4.5%** |
| 129 토큰, 동시성 4 | 36.70 req/s | 35.27 req/s | +4.1% |
| 1901 토큰, 동시성 1 | **7.08 req/s** (141.2 ms) | 6.76 req/s (147.7 ms) | **+4.7%** |
| 1901 토큰, 동시성 2 | 7.61 req/s | 7.43 req/s | +2.4% |

포화 처리량이 4~5% 올랐다.

읽어낼 점 두 가지.

1. **단건 지연은 거의 그대로다.** 동시성 1 에서는 장비가 놀고 있어서 메모리 왕복을 줄여도
   티가 안 난다. 이득은 장비가 꽉 찬 다음부터 나오고 아낀 대역폭이 그때 처리량으로 바뀐다.
2. **긴 입력에서도 4.7% 다.** 토큰이 15배로 늘어도 이득 비율이 비슷하다. 왕복량과 연산량이
   둘 다 토큰 수에 비례하기 때문이다.

## 결론

| 항목 | 결과 |
|---|---|
| 새 NKI 커널 작성 | **불필요.** 기존 `NF.qkv_proj` 가 이미 지원한다 |
| 정확도 | 유지 (Level 2 PASS, STS12 차이 4.3e-05) |
| 처리량 | **+4~5%** (37.18 req/s, 4,796 tok/s) |
| 부수 효과 | Phase 1 에서 포기했던 커널 융합 경로를 되찾음 |
