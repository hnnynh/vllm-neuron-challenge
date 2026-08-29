#!/usr/bin/env python3
"""
vLLM Neuron 모델 온보딩 — Step 0: 아키텍처 Diff 분석
임의의 두 모델 비교 가능 (MODEL_A, MODEL_B만 변경)

사용법: python3 arch_diff_analysis.py
필요: pip install transformers torch (weight 다운로드 없음, config만 fetch)
"""

import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_A = "meta-llama/Llama-3.1-8B"
MODEL_B = "google/gemma-4-31B-it"

print("=" * 80)
print("  1. config.json 비교")
print("=" * 80)

config_a = AutoConfig.from_pretrained(MODEL_A)
config_b = AutoConfig.from_pretrained(MODEL_B)

dict_a = config_a.to_dict()
dict_b = config_b.to_dict()

all_keys = sorted(set(list(dict_a.keys()) + list(dict_b.keys())))

print(f"\n{'Key':<40} {MODEL_A:<30} {MODEL_B:<30}")
print("-" * 100)

only_a = []
only_b = []
different = []

for key in all_keys:
    if key in ['_name_or_path', 'transformers_version', 'tokenizer_class', '_attn_implementation_autoset']:
        continue

    val_a = dict_a.get(key, "—")
    val_b = dict_b.get(key, "—")

    if val_a == "—":
        only_b.append((key, val_b))
    elif val_b == "—":
        only_a.append((key, val_a))
    elif val_a != val_b:
        different.append((key, val_a, val_b))

print("\n### 값이 다른 필드 ###")
for key, va, vb in different:
    print(f"  {key:<38} {str(va)[:28]:<30} {str(vb)[:28]}")

print(f"\n### {MODEL_B}에만 있는 필드 ({len(only_b)}개) ###")
for key, val in only_b[:20]:
    print(f"  {key:<38} = {str(val)[:50]}")

print(f"\n### {MODEL_A}에만 있는 필드 ({len(only_a)}개) ###")
for key, val in only_a[:20]:
    print(f"  {key:<38} = {str(val)[:50]}")


print("\n\n")
print("=" * 80)
print("  2. 모델 구조 출력 (weight 로드 없음, 메모리 0)")
print("=" * 80)

print(f"\n### {MODEL_A} 구조 ###\n")
try:
    with torch.device("meta"):
        model_a = AutoModelForCausalLM.from_config(config_a)
    model_str = str(model_a)
    lines = model_str.split('\n')
    for line in lines[:60]:
        print(f"  {line}")
    if len(lines) > 60:
        print(f"  ... ({len(lines) - 60} more lines)")
    del model_a
except Exception as e:
    print(f"  [ERROR] {e}")

print(f"\n### {MODEL_B} 구조 ###\n")
try:
    with torch.device("meta"):
        model_b = AutoModelForCausalLM.from_config(config_b)
    model_str = str(model_b)
    lines = model_str.split('\n')
    for line in lines[:60]:
        print(f"  {line}")
    if len(lines) > 60:
        print(f"  ... ({len(lines) - 60} more lines)")
    del model_b
except Exception as e:
    print(f"  [ERROR] {e}")
