# Neuron Onboarding Challenge

- AWS Trainium 위 [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron) 플러그인에 모델 온보딩
- 참고 문서: [AWS 모델 온보딩 가이드](https://awslabs.github.io/accelerated-compute-tutorials/aws-ai-chip/inference/vllm/model-onboarding)
- 모델 하나당 Phase 하나

## Phase 1 — Qwen3-Embedding-8B

- 환경: `trn2.3xlarge` (NeuronCore 4개) · 플러그인 `release-0.24.0.1.1.0` · 공식 Neuron DLC
- 방식: Llama 템플릿 이식
- 정확도: 레시피 기준값과 동일
- 처리량: 동시 요청 2~4에서 포화 (36.05 req/s)
- 레포트
  - [phase-1-onboarding.ko.md](phase-1-onboarding.ko.md)
  - [phase-1-onboarding.en.md](phase-1-onboarding.en.md)

## Phase 2 — 커널 융합 (선택)

- 환경: Phase 1 과 동일
- 대상: Phase 1 의 `qwen3_embedding` 구현
- 확인: `NF.qkv_proj` 에 QK-norm 융합 인자 기존재 — NKI 커널 직접 작성 불필요
- 조치: 융합 활성화
- 결과: 포화 처리량 4~5% 상승 (37.18 req/s), 정확도 유지
- 레포트
  - [phase-2-kernel-fusion.ko.md](phase-2-kernel-fusion.ko.md)
  - [phase-2-kernel-fusion.en.md](phase-2-kernel-fusion.en.md)
 
