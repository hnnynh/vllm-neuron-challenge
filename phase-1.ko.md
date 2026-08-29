# Challenge — vLLM Neuron 모델 온보딩 리포트

> Phase 1: NF 함수로 모델 "교체" 
> Qwen3-Embedding-8B를 vLLM Neuron에 올리기

플러그인이 제공하는 **NF 함수**로 모델을 조립한다.

| 항목 | 값 |
|---|---|
| 모델 | Qwen/Qwen3-Embedding-8B (80억 파라미터 임베딩 모델) |
| 장비 | trn2.3xlarge — AWS Trainium2 칩, NeuronCore 4개 |
| 기반 | vLLM Inference NeuronX **공식 DLC**(컨테이너 이미지) `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04` |
| 플러그인 | vllm-project/vllm-neuron `release-0.24.0.1.1.0` |
| dtype | BF16 |
| TP | 4 |


## 전체 흐름 (5단계)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ 1. Implement    2. Register      3. Compile &    4. Validate    5. Benchmark  │
│ (config/model/   (ModelRegistry)   Smoke Test     Accuracy       & Tune       │
│  factory/weights)                                                             │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 용어 미니 사전

| 용어 | 뜻 |
|---|---|
| **NF 함수** | 플러그인이 제공하는 조립 블록(`NF.mlp` 등). 내부적으로 Neuron 에 최적화된 커널을 호출한다 |
| **NKI** | Neuron Kernel Interface. 커널을 직접 짤 때 쓰는 저수준 API (Phase 2 영역) |
| **decode** | 토큰을 한 개씩 뱉는 단계. **임베딩은 prefill 만 있다** |
| **TP (tensor parallel)** | 모델을 여러 코어에 쪼개 얹는 방식. 여기선 4코어에 4등분 |
| **RoPE** | 토큰 위치 정보를 회전으로 넣는 기법 |
| **QK-norm** | 어텐션의 Q·K 를 정규화하는 층. Qwen3 에 있고 Llama 에는 없다 |
| **GQA** | 여러 Q 헤드가 K/V 헤드를 나눠 쓰는 어텐션 구조 |
| **pooling** | 토큰별 벡터를 문장 벡터 하나로 합치는 것. 이 모델은 마지막 토큰을 쓴다 |
| **컴파일 / NEFF** | Neuron은 실행 전 그래프를 미리 컴파일한다. 그 산출물이 NEFF |

## 실행 환경

| 항목 | 상태 |
|---|---|
| 인스턴스 | trn2.3xlarge ✓ (12 vCPU, 124GB RAM, `/dev/neuron0`) |
| 디스크 | 258GB 여유 ✓ (8B 모델 ~16GB + DLC 이미지 감당 가능) |
| Docker | 설치됨, 이미지 0개 — DLC 미확보 |
| NeuronCore | 유휴 ✓ |


## 환경 구축 — DLC

DLC는 플러그인이 비-editable로 설치된 상태로 오기 때문에 모델을 추가하려면 소스를 마운트해 교체한다.

```bash
docker run -d --name vllm-dev \
  --device /dev/neuron0 --network host --shm-size 8g \
  -v ~/vllm-neuron/repo:/workspace/src \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash <IMG> -c 'sleep infinity'

docker exec vllm-dev bash -c 'pip uninstall -y vllm-neuron && cd /workspace/src && pip install -e . --no-deps'
```

| 플래그 | 없으면 |
|---|---|
| `--no-deps` | pip 이 의존성을 재해석해 DLC 고정 조합(torch 2.11.0 / torch-xla 2.11.0 / libtorch-neuronx-lite / neuronx-cc 2.27.5334.0 / transformers 5.15.0)을 흔든다 |
| `--device /dev/neuron0` | 플러그인이 "No Neuron devices found" 로 등록을 건너뛴다 |


## Stage 0: 아키텍처 Diff 분석

`~/workspace/compare.py` (표준 라이브러리 AST 파싱 + json, 가중치 로드 없음 — 메모리 0)
실행 스크립트 출처: [AWS vLLM Neuron 모델 온보딩 가이드](https://awslabs.github.io/accelerated-compute-tutorials/aws-ai-chip/inference/vllm/model-onboarding/#_1)

### 1. config.json

| 필드 | 의미 | meta-llama/Llama-3.1-8B | Qwen/Qwen3-Embedding-8B |
|---|---|---|---|
| architectures | 모델 클래스 이름 — vLLM `ModelRegistry` 조회 키 | `['LlamaForCausalLM']` | `['Qwen3ForCausalLM']` |
| model_type | HF config 클래스 매핑 키 | `llama` | `qwen3` |
| bos_token_id | 문장 시작 토큰 ID | 128000 | 151643 |
| eos_token_id | 문장 끝 토큰 ID | 128001 | 151645 |
| vocab_size | 어휘 크기 = embed_tokens 행 수 | 128256 | 151665 |
| num_hidden_layers | 디코더 레이어 수 | 32 | 36 |
| intermediate_size | MLP 중간 차원 (gate/up 출력) | 14336 | 12288 |
| max_position_embeddings | 최대 컨텍스트 길이 | 131072 | 40960 |
| rms_norm_eps | RMSNorm 분모 안정화 상수 | 1e-05 | 1e-06 |
| rope_parameters | RoPE 설정 — 표현 방식 자체가 다름 | `{'factor': 8.0, 'low_freq_fa…}` (piecewise 스케일링) | `{'rope_theta': 1000000, 'rop…}` (표준, theta=1e6) |
| layer_types | 레이어별 attention 타입 배열 | — | `['full_attention', …]` (36개 전부 동일) |
| max_window_layers | 슬라이딩 윈도우 적용 레이어 수 | — | 36 |
| sliding_window | 슬라이딩 윈도우 폭 | — | `None` (미사용) |
| use_sliding_window | 슬라이딩 윈도우 on/off | — | `False` |
| mlp_bias | MLP Linear bias 유무 | `False` | — (필드 없음 = bias 없음) |
| pretraining_tp | 학습 시 TP 차수 (추론 무관) | 1 | — |

### 2. 모델 구조

| 모듈 | 의미 | LlamaForCausalLM | Qwen3ForCausalLM |
|---|---|---|---|
| embed_tokens | 토큰 ID → 벡터 룩업 테이블 | `Embedding(128256, 4096)` | `Embedding(151665, 4096)` |
| layers | 디코더 블록 스택 | `32 x LlamaDecoderLayer` | `36 x Qwen3DecoderLayer` |
| q_proj / o_proj | Q 생성 / attention 출력을 hidden 으로 복귀 | `4096 → 4096`, bias=False | `4096 → 4096`, bias=False |
| k_proj / v_proj | K·V 생성 — 1024 = 8 KV head × 128, **GQA (Q 32 head 가 4:1 공유)** | `4096 → 1024`, bias=False | `4096 → 1024`, bias=False |
| **q_norm / k_norm** | **Q·K 를 head_dim(128) 기준 per-head 정규화 — 내적 스케일 폭주 억제. projection 과 RoPE 사이에 위치** | **없음** | **`RMSNorm((128,), eps=1e-06)`** |
| gate_proj / up_proj | MLP 확장 — gate 는 게이팅 신호, up 은 값 | `4096 → 14336` | `4096 → 12288` |
| down_proj | MLP 축소 — hidden 차원 복귀 | `14336 → 4096` | `12288 → 4096` |
| act_fn | gate 에 적용하는 활성함수 (SwiGLU) | `SiLUActivation()` | `SiLUActivation()` |
| input / post_attention_layernorm | 블록 내 pre-norm 2개 — attention 앞, MLP 앞 | `RMSNorm((4096,), eps=1e-05)` | `RMSNorm((4096,), eps=1e-06)` |
| norm | 마지막 블록 뒤 최종 정규화 — **임베딩 출력이 여기서 나옴** | `RMSNorm((4096,), eps=1e-05)` | `RMSNorm((4096,), eps=1e-06)` |
| rotary_emb | RoPE cos/sin 캐시 생성 | `LlamaRotaryEmbedding()` (piecewise 스케일링) | `Qwen3RotaryEmbedding()` (표준, theta=1e6) |
| lm_head | hidden → vocab logit 투영 (생성용) | `4096 → 128256`, bias=False | `4096 → 151665`, bias=False — **체크포인트에 텐서 부재, 제거 대상** |

클래스 구성은 1:1 대응한다(RMSNorm / RotaryEmbedding / Attention / MLP / DecoderLayer / Model).
키워드 대조 결과 **구조적 핵심 차이는 QK-norm 하나**다 — llama3 에 `q_norm`/`k_norm` 이 0회,
qwen3 에 7회/6회 등장.

### 3. 임베딩 고유 요구사항 (HF 메타데이터)

| 출처 | 값 | 함의 |
|---|---|---|
| `modules.json` | Transformer → Pooling → Normalize | 파이프라인 3단 |
| `1_Pooling/config.json` | `pooling_mode_lasttoken: true`, 차원 4096 | 마지막 토큰 은닉상태 + L2 정규화 |
| 체크포인트 | `lm_head` 텐서 부재 | 생성 경로 불필요 |


### 4. 온보딩 작업 항목 도출 (8개 체크리스트 대조)

Qwen3-Embedding-8B의 관측값을 Llama 템플릿(`vllm_neuron/model/llama3/`)과 대조해 작업 항목 도출

| # | 비교 항목 | 확인한 필드 | 관측 | 도출된 작업 |
|---|---|---|---|---|
| 1 | Attention 방식 | `num_attention_heads` 32=32, `num_key_value_heads` 8=8, `layer_types` 전부 `full_attention`, `sliding_window` `None` / `use_sliding_window` `False` | GQA 형태 동일, 슬라이딩 윈도우 없음 | **attention forward 구조 변경 불필요** — 단 `head_dim` 128 로 템플릿 기본값(64) 교체 |
| 2 | Heterogeneous Layers | `layer_types` 배열, `per_layer_config` 부재 | 36개 레이어 전부 동일 타입 | **레이어별 분기·weight 초기화 분기 불필요** |
| 3 | Position Encoding | `rope_theta` 1e6, `rope_scaling` `None`, `partial_rotary_factor` 부재 | Llama 는 `rope_parameters` dict(`rope_type=default`, `theta=5e5`) + piecewise 스케일링 | `config.py` RoPE 표현을 `rope_theta` + `rope_scaling` 로 전환, RotaryEmbedding 의 piecewise 스케일링 삭제. **QK-norm seam 확보를 위해 `NF.qkv_proj` 융합 RoPE 해제** (Stage 1 핵심) |
| 4 | MLP / Activation | `hidden_act` `silu`, `intermediate_size` 12288 | Llama 도 SiLU, 구조 동일 | **`NF.mlp` 그대로 사용 가능** — config 값만 교체 |
| 5 | Normalization | `rms_norm_eps` 1e-6, modeling 소스에 `q_norm`/`k_norm` (llama3 0회 / qwen3 7회·6회) | RMSNorm 타입·위치는 동일, **QK-norm 만 추가** | Attention 에 `head_dim` 기준 per-head RMSNorm 추가(hidden_size 아님), decode 경로에 gamma 연결 |
| 6 | Config 구조 | top-level 필드만, `text_config`/`vision_config` 없음 | 중첩 없음 | **`from_configs()` 파싱 로직 변경 불필요** — Llama 전용 필드(`draft_vocab_size`, `norm_before_fc`, `norm_before_residual`)와 Eagle3 nested 블록만 제거 |
| 7 | Embedding | `tie_word_embeddings` `False`, `vocab_size` 151665 | weight sharing 없음, 체크포인트에 `lm_head` 텐서 부재 | `lm_head` 제거 → last-token pooling + L2 normalize 경로. **151665 는 TP=4 로 안 나눠떨어짐(37916.25) → embed_tokens 로더 `pad_shard=True`** |
| 8 | Special features | `final_logit_softcapping`·`enable_moe_block`·`vision_config` 전부 부재 | 추가 아키텍처 요소 없음 | **없음.** 대신 임베딩 요구(§3)가 `DispatchPooler` LAST + L2 를 추가 요구 |

## Stage 1: Implement (모델 구현)

임베딩에 필요 없는 파일(추측 디코딩용 `eagle3_model.py`, FP8 관련 파일들)을 제외하고 `vllm_neuron/model/llama3/` 를 `qwen3_embedding/` 으로 복사한 뒤 위 8개 항목대로 수정했다.

### 가장 중요한 변경 — RoPE 융합 해제

Llama 템플릿은 속도를 위해 **RoPE 를 QKV 계산 커널 안에 합쳐 놓았다**. 그런데 Qwen3 는
QK-norm 이 **QKV 계산과 RoPE 사이**에 들어가야 한다. 합쳐진 커널에는 끼어들 틈이 없다.

그래서 융합을 풀고 순서를 밖에서 만들었다.

```python
qkv = NF.qkv_proj(hidden=..., qkv_weights=..., bias=None).squeeze(0)   # ① QKV 계산 (RoPE 없이)
q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
q = self.q_norm(q); k = self.k_norm(k)                                  # ② QK-norm
q, k = apply_rotary_pos_emb(q, k, cos, sin)                             # ③ RoPE
```

**대신 포기한 것** — 커널 안에서 처리하던 KV 캐시 쓰기, 분할 prefill, FP8 경로, 컨텍스트 병렬.
임베딩은 prefill 만 쓰고 FP8 은 레시피가 지원하지 않는다고 명시했으므로 실질 손실은 없다.
다만 분할 prefill 을 안 쓰므로 **prefix caching(같은 앞부분 재사용) 은 지원하지 않는다.**
잘못 켜고 실행하면 조용히 틀리는 대신 명시적으로 에러를 낸다.

### 미리 막아둔 두 함정

둘 다 Step 0 에서 미리 발견해 구현 단계에서 처리했다. 안 했으면 서버 기동이 실패한다.

1. **transformers 5.x 의 config 정리 방식** — 이 라이브러리는 `rope_theta` 값을
   `rope_scaling` 이라는 칸 안으로 옮겨 담는다. 겉보기엔 "스케일링이 설정됨"처럼 보이지만
   `rope_type` 이 `default` 면 그냥 표준 RoPE 다. 이 형태를 알아보고 값만 꺼내 쓰게 했다.
2. **vocab 151665 가 4로 안 나눠떨어짐** (151665 ÷ 4 = 37916.25) — 임베딩 표를 4코어에 쪼갤 때
   마지막 코어 몫이 모자란다. `pad_shard=True` 로 빈 자리를 채우게 했다.
   **Llama 템플릿을 그대로 쓰면 반드시 밟는 함정이다** — Llama 의 vocab 128256 은 4로 딱
   나눠떨어져서 템플릿에는 이 옵션 자체가 안 쓰여 있다.

### 체크포인트 실물 확인

가중치 파일의 텐서 398개를 전부 열어 확인했다.

- `lm_head`(다음 단어 예측용 출력층) 텐서가 **아예 없다** → 임베딩 모델이라는 사실과 일치
- 키 이름에 `model.` 접두사가 없다 → 이름 매핑에 반영
- `q_norm`/`k_norm` 이 있다 → 헤드마다 따로라 코어 분할을 하면 안 된다

## Stage 2: Register (모델 등록)

모델의 값 `Qwen3ForCausalLM` 은 이미 내장 구현이 차지하고 있어서 다른 이름(`Qwen3EmbeddingForEmbedding`)으로 등록하고 실행할 때 `--hf-overrides` 로 지정했다. vLLM 쪽 레지스트리와 플러그인 쪽 목록 두 군데 모두 등록했다.

## Stage 3: Compile & Smoke Test

```bash
export VLLM_NEURON_QWEN3_EMBEDDING=1     # 모델 등록 켜기
export NEURON_SKIP_EFA_AFFINITY=1        # 이 장비엔 EFA가 없다
vllm serve Qwen/Qwen3-Embedding-8B --runner pooling \
  --hf-overrides '{"architectures": ["Qwen3EmbeddingForEmbedding"]}' \
  --tensor-parallel-size 4 --max-model-len 2048 \
  --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --no-enable-prefix-caching --port 8000
```

**결과 — 기동 실패 0건.** 컴파일 포함 약 3분 30초 만에 떴다.

### 동작 확인 결과

| 확인 항목 | 결과 | 의미 |
|---|---|---|
| 벡터 길이 | **4096** | 레시피가 명시한 값과 일치 |
| 벡터 크기 | ‖v‖ = 1.000001 | L2 정규화가 걸려 있다 |
| 비슷한 문장 vs 무관한 문장 | 0.6822 > 0.2867 | 의미를 구분한다 |
| 영어 ↔ 한국어 같은 뜻 | **0.8144** | 다국어 의미를 잡는다 |
| 응답 시간 (4문장) | 127 ms | — |

**통과.**

## Stage 4: Validate Accuracy (3-Level 프레임워크)

| Level | 무엇을 보나 | 무엇과 비교하나 | 통과 기준 |
|---|---|---|---|
| **3 모듈** | 부품 하나하나 (RMSNorm, MLP …) | HF 레퍼런스 구현 | 텐서가 일치 |
| **2 프롬프트** | 문장 하나의 출력값 | HF FP32 / HF BF16 / Neuron 3자 비교 | Neuron 오차 ≈ BF16 오차 |
| **1 태스크** | 모델 전체 성능 | 실제 벤치마크 점수 | 직접 정한 기준선 |

> **임베딩 모델이라 한 가지를 바꿔 적용했다.** 원 프레임워크의 Level 2 는 생성 모델의
> logit(다음 단어 확률)을 비교한다. 이 모델은 글자를 만들지 않으므로 그 자리에
> **최종 임베딩 벡터**를 놓고 같은 3자 비교를 했다.

문서가 권하는 순서대로 **Level 3 → 2 → 1** 으로 진행했다.

---

### Level 3: 모듈 단위 (부품 검사)

포팅한 각 부품을 **실제 체크포인트 가중치**로 계산해 HF 레퍼런스 구현과 맞춰봤다.
CPU FP32 로 돌려 하드웨어 오차를 배제하고 순수하게 수식이 같은지만 본다.

| 부품 | 무엇을 하는 층인가 | 최대 오차 | 판정 |
|---|---|---|---|
| RMSNorm (hidden_size) | 레이어 입출력 정규화 | 0.000e+00 | PASS |
| **QK-norm (head_dim)** | **Qwen3 에만 있는 층 — 이번 포팅의 핵심** | 0.000e+00 | PASS |
| RoPE (rotate_half) | 토큰 위치 정보 주입 | 0.000e+00 | PASS |
| MLP (SwiGLU) | 피드포워드 블록 | 0.000e+00 | PASS |

**전부 오차 0 — 비트 단위로 같다.** 특히 QK-norm 은 Llama 템플릿에 없어 새로 넣은 부분이라
여기서 어긋났다면 이후 단계가 전부 무의미해진다. 수식·eps·적용 축(head_dim 기준 per-head)이
모두 맞았다는 뜻이다.

**LEVEL 3: PASS**

### Level 2: 프롬프트 단위 (3자 비교)

같은 문장을 세 가지로 임베딩해 비교한다.

| 이름 | 무엇인가 | 역할 |
|---|---|---|
| `expected` | HF FP32 (CPU, 32비트) | **정답 기준** |
| `baseline` | HF BF16 (CPU, 16비트) | **허용 오차의 척도** |
| `target` | Neuron (우리 서버) | 검사 대상 |

핵심은 BF16 을 끼워 넣는 것. 16비트로 계산하면 정밀도 때문에 **원래 어느 정도 오차가 난다.**
따라서 Neuron 오차를 0 과 비교하면 안 되고 **BF16 오차와 비교**해야 한다.

> 판정 원리 — **Neuron 오차 ≈ BF16 오차 → 정상. Neuron 오차 >> BF16 → 버그.**

| 문장 | cos(Neuron, FP32) | cos(BF16, FP32) | maxΔ Neuron | maxΔ BF16 | 판정 |
|---|---|---|---|---|---|
| "The cat sits on the mat" | 0.99990443 | 0.99989652 | 2.231e-03 | 2.630e-03 | PASS |
| "Stock markets fell sharply…" | 0.99993960 | 0.99990832 | 2.738e-03 | 1.585e-03 | PASS |
| "고양이가 매트 위에 앉아 있다" | 0.99993650 | 0.99990324 | 1.530e-03 | 2.024e-03 | PASS |
| "Photosynthesis converts light…" | 0.99990419 | 0.99991308 | 1.544e-03 | 1.609e-03 | PASS |

**Neuron 오차가 BF16 오차와 같은 자릿수다** (1.5~2.7e-03 대 1.6~2.6e-03). 두 문장에서는
오히려 Neuron 쪽이 더 작고 FP32 정답과의 코사인 유사도는 네 문장 모두 **0.9999** 다.
즉 남은 차이는 구현 오류가 아니라 **16비트 연산이 원래 갖는 오차**다.

**LEVEL 2: PASS**

### Level 1: 태스크 단위 (실전 성능)

표준 벤치마크 MTEB 로 과제가 지정한 세 태스크를 측정했고 기준선은 공식 레시피 수치로 잡았다.

| 태스크 | 무엇을 재나 | 우리 결과 | 레시피 기준 | 달성률 |
|---|---|---|---|---|
| STS12 | 두 문장이 얼마나 비슷한지 맞히기 | **0.8139** | 0.8639 | 94.2% |
| SciFact | 과학 주장에 맞는 근거 문서 찾기 | **0.7906** | 0.7839 | **100.9%** |
| NFCorpus | 의학 질의에 맞는 문서 찾기 | **0.4101** | 0.4143 | 99.0% |

SciFact 는 기준을 넘었고 나머지도 94~99% 다.

측정 조건 — MTEB 2.20.3, 서버를 OpenAI 호환 `/v1/embeddings` 로 두고 붙였다. 두 가지가 결과를
좌우한다: **검색 태스크는 제목과 본문을 이어 붙여야** 하고, **지시문은 질의 쪽에만 붙여야** 한다
(모델 카드 규약. 문장 유사도 같은 대칭 태스크에는 붙이지 않는다). NFCorpus 에는 입력 한도
2048 토큰을 넘는 문서가 있어 서버측 절단을 요청했다.

**LEVEL 1: PASS** (사용자 정의 기준 — 레시피 대비 90% 이상)

### STS12 가 5% 모자란 이유

Level 1 만 완전히 붙지 않았으므로 원인을 갈랐다. **내장 공식 구현을 똑같은 서버 설정·똑같은 평가 코드로** 돌렸다 (코어가 4개뿐이라 서버를 바꿔 끼우며 차례로 측정).

```
자체 구현 : 0.8139421814483995
공식 구현 : 0.8139421814483995
```

**소수점 16자리까지 같다.** 공식 구현조차 같은 점수를 내므로 남은 차이는 **채점 조건 차이**다. 후보는 과제별 전용 지시문(주어진 자료에 없어 임의로 만들지 않았다), MTEB 버전, 데이터셋 리비전 정도.

Level 3 과 2 가 통과한 상태이므로 이 결론은 자연스럽다 — 부품도 맞고 문장 단위 출력도 BF16 오차 범위 안이라면 남은 격차가 구현에서 나올 여지가 없다.

### 3-Level 종합

| Level | 결과 | 근거 |
|---|---|---|
| 3 모듈 | **PASS** | RMSNorm·QK-norm·RoPE·MLP 오차 0 |
| 2 프롬프트 | **PASS** | Neuron 오차 ≈ BF16 오차, cos 0.9999 |
| 1 태스크 | **PASS** | SciFact 100.9%, NFCorpus 99.0%, STS12 94.2% |
| 교차 확인 | — | 공식 구현과 소수점 16자리 일치 |

## Stage 5 — 성능 측정

임베딩 모델은 **글자를 하나씩 뱉는 단계가 없다.** 요청 1건 = 입력을 한 번 읽는 것으로 끝이다.
그래서 생성 모델에서 쓰는 "토큰당 시간" 지표는 의미가 없고 요청 단위 지연과 처리량을 잰다.

| 동시 요청 수 | 초당 요청 | 평균 지연 | p99 지연 |
|---|---|---|---|
| 1 | 29.50 | 33.8 ms | 35.4 ms |
| 2 | 35.61 | 55.5 ms | 61.2 ms |
| **4** | **36.05** | 107.5 ms | 118.9 ms |
| 8 | 35.28 | 209.9 ms | 227.7 ms |

읽어낼 점 세 가지.

1. **동시 2~4에서 한계에 도달한다.** 1→2 는 21% 빨라지지만 2→4 는 1%, 4→8 은 오히려 준다.
   그 뒤로는 지연만 비례해서 늘어난다 — 줄만 길어지고 처리량은 그대로다.
2. **여러 문장을 한 요청에 묶어도 이득이 없다.** 요청당 8문장으로 묶어봐도 초당 임베딩 수가
   똑같이 35.5 에서 멈춘다. 임베딩은 전부 "한 번에 읽기"라, 묶어도 겹쳐서 처리할 빈 시간이 없기 때문이다.
3. **느린 쪽 꼬리가 안정적이다.** 모든 조건에서 p99 가 중앙값의 1.15배 미만이라 예측 가능하다.

최대 처리량은 **초당 36건, 약 4,650 토큰/초** (4코어, 8B 모델, BF16 기준).

---

## 결론

**Phase 1 목표 — "동작이 올바른 모델" 달성**

| 판정 기준 | 결과 |
|---|---|
| 서버가 뜨는가 | 기동 실패 0건 |
| 임베딩이 의미를 담는가 | 교차언어 0.81, 유사/무관 구분 명확 |
| 정확도가 기준에 닿는가 | 3-Level 전부 PASS — 모듈 오차 0, BF16 오차 범위 내, SciFact 100.9% |
| 구현이 정확한가 | 공식 구현과 소수점 16자리 일치 |
| 성능이 나오는가 | 초당 36건 (커스텀 커널 없이) |
