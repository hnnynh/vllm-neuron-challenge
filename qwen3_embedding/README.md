# qwen3_embedding — vLLM Neuron 플러그인 모델

`vllm_neuron/model/llama3/` 를 복사해 `Qwen/Qwen3-Embedding-8B` 로 포팅한 모듈.
trn2.3xlarge (NeuronCore 4개, TP=4, BF16) 기준.

| 파일 | 내용 |
|---|---|
| `config.py` | 모델 설정 (head_dim 128, rope_theta 1e6, vocab 151665, eps 1e-6) |
| `model.py` | 백본 — QK-norm, 표준 RoPE, prefill 경로(융합/비융합) |
| `model_embedding.py` | pooling 모델(LAST + L2), 가중치 로딩 |
| `factory.py` | 모델 선택 + 설정 검증 |

Llama 템플릿과의 핵심 차이는 **QK-norm**(projection 과 RoPE 사이)이다.
`vocab_size` 151665 가 TP=4 로 나누어떨어지지 않아 embed_tokens 로더에 `pad_shard=True` 가 필요하다.

## 설치

```bash
cp -r qwen3_embedding <repo>/vllm_neuron/model/    # 모듈 배치
git apply ../registration.patch                    # 두 레지스트리에 등록
pip install -e . --no-deps
```

## 실행

```bash
export VLLM_NEURON_QWEN3_EMBEDDING=1        # 모델 등록 활성화
export NEURON_SKIP_EFA_AFFINITY=1           # trn2.3xlarge 에 EFA 없음
vllm serve Qwen/Qwen3-Embedding-8B --runner pooling \
  --hf-overrides '{"architectures": ["Qwen3EmbeddingForEmbedding"]}' \
  --tensor-parallel-size 4 --max-model-len 2048 \
  --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --no-enable-prefix-caching --port 8000
```

`VLLM_NEURON_QWEN3_EMBEDDING_UNFUSED=1` 이면 QK-norm·RoPE 를 커널 융합 없이 실행한다 (A/B 대조용).

## 제약

prefix caching · FP8/MXFP4 · DCP(context parallel) 미지원.
