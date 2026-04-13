#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# TaskMind-1.1B Fine-Tuning Script
# Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Task:  Team chat intent classification → structured JSON
# HW:    Apple M5 Pro Max (any M-series Mac works)
# Time:  ~10 minutes for 300 iterations
# ─────────────────────────────────────────────────────────────────────────────

set -e

VENV="./.venv-mlx"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR="data/task_mgmt_data/train"
ADAPTER_OUT="out/taskmind_v1"
HF_REPO="SatyamSinghal/taskmind-1.1b-chat-lora"

echo "▶ Step 1 — Verify environment"
$VENV/bin/python -c "import mlx_lm; print('  mlx_lm OK')"
$VENV/bin/python -c "import huggingface_hub; print('  huggingface_hub OK')"

echo "▶ Step 2 — Check training data"
echo "  Train examples: $(wc -l < $DATA_DIR/train.jsonl)"
echo "  Valid examples: $(wc -l < data/task_mgmt_data/valid/valid.jsonl)"

echo "▶ Step 3 — Fine-tune TaskMind-1.1B"
$VENV/bin/python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data "$DATA_DIR" \
  --batch-size 4 \
  --iters 300 \
  --learning-rate 1e-4 \
  --num-layers 16 \
  --max-seq-length 256 \
  --mask-prompt \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --save-every 100 \
  --adapter-path "$ADAPTER_OUT"

echo "▶ Step 4 — Quick inference test"
$VENV/bin/python - << 'PY'
from mlx_lm import load, generate
import json

SYSTEM = (
    "You are TaskMind, an intelligent task management AI that extracts structured "
    "task information from team chat messages. Output valid JSON only with fields: "
    "intent, assigneeName, project, title, deadline, priority, progressPercent. "
    "Intent must be one of: TASK_ASSIGN, TASK_DONE, TASK_UPDATE, PROGRESS_NOTE, GENERAL_MESSAGE."
)

model, tokenizer = load(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_path="out/taskmind_v1"
)

test_msgs = [
    "@Agrim fix the growstreams deck ASAP NO Delay",
    "done bhai, merged the PR",
    "login page 60% ho gaya",
    "getting 500 error on registration",
    "Sure sir ready for it 🔥",
]

print("\n── TaskMind Inference Test ──")
for msg in test_msgs:
    prompt = f"<|system|>\n{SYSTEM}\n<|user|>\n{msg}\n<|assistant|>\n"
    out = generate(model, tokenizer, prompt, max_tokens=128).strip()
    try:
        parsed = json.loads(out)
        intent = parsed.get("intent","?")
        print(f"  [{intent:<18}] {msg[:55]}")
    except:
        print(f"  [PARSE ERROR      ] {msg[:55]} → {out[:50]}")
PY

echo ""
echo "▶ Step 5 — Copy README to adapter folder"
cp README_TaskMind.md "$ADAPTER_OUT/"

echo "▶ Step 6 — Push to HuggingFace"
$VENV/bin/python - << 'PY'
import os
from huggingface_hub import HfApi, create_repo, upload_folder

token  = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    print("  ⚠ Set HUGGING_FACE_HUB_TOKEN env var to push to HuggingFace")
    exit(0)

repo_id = "SatyamSinghal/taskmind-1.1b-chat-lora"
api = HfApi(token=token)
create_repo(repo_id=repo_id, private=False, exist_ok=True, repo_type="model")
upload_folder(
    repo_id=repo_id,
    folder_path="out/taskmind_v1",
    repo_type="model",
    token=token
)
print(f"  ✓ Uploaded → https://huggingface.co/{repo_id}")
PY

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  TaskMind-1.1B training and push complete."
echo "  Model: https://huggingface.co/SatyamSinghal/taskmind-1.1b-chat-lora"
echo "══════════════════════════════════════════════════════════"
