#!/bin/bash

# Load .env file (if python-dotenv is used in client scripts)
# source .env

# --- КОНФИГУРАЦИЯ МОДЕЛИ ---
# Выберите одну из моделей:
# MODEL_NAME="facebook/opt-1.3b"
# MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct" # Требует HF токена
# MODEL_NAME="HuggingFaceH4/zephyr-7b-alpha"     # Требует HF токена

MODEL_NAME="facebook/opt-1.3b" # Пример: используем легкую модель для старта

HOST="0.0.0.0"
PORT="8000"
DTYPE="auto" # Используйте "half" или "bfloat16" для экономии памяти GPU, если поддерживается
GPU_MEM_UTILIZATION="0.75" # Уменьшено для решения проблемы с нехваткой памяти
MAX_MODEL_LEN="1024" # Уменьшено для экономии памяти

echo "Starting vLLM server for model: $MODEL_NAME"
python -m vllm.entrypoints.api_server \
    --model $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --dtype $DTYPE \
    --gpu-memory-utilization $GPU_MEM_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code # Может потребоваться для некоторых моделей, которые используют кастомный код
