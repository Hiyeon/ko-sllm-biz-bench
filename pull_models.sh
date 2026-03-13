#!/bin/bash
# Ollama 모델 일괄 다운로드 스크립트
# 실행: bash pull_models.sh

MODELS=(
  # --- 2B 이하 ---
  "qwen2.5:0.5b"       # Qwen 2.5 0.5B
  "gemma3:1b"          # Gemma 3 1B
  "llama3.2:1b"        # Llama 3.2 1B
  # "exaone4.0:1.2b"   # EXAONE 4.0 1.2B (Ollama 미지원 시 주석 유지)
  "qwen2.5:1.5b"       # Qwen 2.5 1.5B
  "gemma2:2b"          # Gemma 2 2B

  # --- 3B~4B ---
  "llama3.2:3b"        # Llama 3.2 3B
  "qwen2.5:3b"         # Qwen 2.5 3B
  "phi4-mini"          # Phi-4 Mini 3.8B
  "gemma3:4b"          # Gemma 3 4B

  # --- 7B~9B ---
  "qwen2.5:7b"         # Qwen 2.5 7B
  "deepseek-r1:7b"     # DeepSeek-R1 7B
  "exaone3.5:7.8b"     # EXAONE 3.5 7.8B (LG AI, 한국어 특화)
  "llama3.1:8b"        # Llama 3.1 8B
  "gemma2:9b"          # Gemma 2 9B
)

TOTAL=${#MODELS[@]}
echo "Total: ${TOTAL} models"
echo ""

PASS=0
FAIL=0

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  echo "[$((i+1))/$TOTAL] pulling $model ..."
  ollama pull "$model"
  if [ $? -eq 0 ]; then
    echo "OK: $model"
    ((PASS++))
  else
    echo "FAIL: $model"
    ((FAIL++))
  fi
  echo ""
done

echo "Done: ${PASS} success / ${FAIL} failed"
echo ""
echo "Installed models:"
ollama list
