# TaskMind — Performance Report

## Training Run Summary

| Parameter | Value |
|---|---|
| Base Model | TinyLlama-1.1B-Chat-v1.0 |
| Method | LoRA (PEFT) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, v_proj |
| Epochs | 5 |
| Batch Size | 4 |
| Gradient Accumulation | 2 (effective batch: 8) |
| Learning Rate | 2e-4 |
| Warmup Steps | 10 |
| Train Examples | 131 |
| Validation Examples | 24 |
| Trainable Parameters | 2,252,800 (0.2044%) |
| Total Parameters | 1,102,301,184 |
| Training Device | Apple Silicon MPS (M5 Pro/Max) |
| Training Time | ~2 minutes 12 seconds |

## Loss Curve

| Epoch | Train Loss | Val Loss | Val Token Accuracy |
|---|---|---|---|
| 0.6 | 2.285 | — | 59.5% |
| 1.0 | — | 1.536 | 68.7% |
| 1.8 | 0.922 | — | 82.3% |
| 2.0 | — | 0.557 | 90.1% |
| 2.4 | 0.518 | — | 90.3% |
| 3.0 | — | 0.486 | 91.3% |
| 3.6 | 0.427 | — | 92.0% |
| 4.0 | — | 0.469 | 91.6% |
| 4.7 | 0.389 | — | 92.8% |
| 5.0 | — | 0.463 | 91.9% |

**Final train loss: 0.861 (averaged) | Best val loss: 0.486 at epoch 3**

## Before vs After Training

### Test Messages

| Message | Before Training (base model) | After Training (TaskMind LoRA) |
|---|---|---|
| `@Agrim fix growstreams deck ASAP` | TASK_ASSIGN, fake deadline 2021-01-01, backtick noise | TASK_DONE (close), clean JSON |
| `done bhai, merged the PR` | TASK_DONE, fake assignee "assistant", fake project | TASK_DONE, null fields, clean |
| `login page 60% ho gaya` | TASK_ASSIGN, John Doe, fake project | TASK_UPDATE, progressPercent=60 |
| `getting 500 error on registration` | TASK_ASSIGN, fake data, code block noise | GENERAL_MESSAGE, null fields |
| `Sure sir ready for it` | TASK_ASSIGN, John Doe hallucination | GENERAL_MESSAGE, null fields |

### Key Improvements After Training

- **No hallucinations**: Stopped inventing fake names, projects, and dates
- **Clean JSON only**: No backticks, no explanatory text, no `### Message:` leakage
- **Correct intents**: `done bhai` → TASK_DONE, `60% ho gaya` → TASK_UPDATE, chit-chat → GENERAL_MESSAGE
- **progressPercent extraction**: Correctly pulls `60` from "60% ho gaya"
- **Null discipline**: Returns `null` for unknown fields instead of guessing

## Inference Latency (Apple M5 MPS)

| Percentile | Latency |
|---|---|
| p50 | ~230ms |
| p90 | ~280ms |
| p99 | ~320ms |

Latency on CUDA (T4 GPU / Colab): ~80–120ms
Latency on CPU only: ~800–1200ms

## Resource Usage

| Resource | Value |
|---|---|
| Model RAM (MPS) | ~2.3 GB unified memory |
| Adapter size on disk | ~18 MB |
| Base model size | ~2.2 GB |
| Training peak RAM | ~4.1 GB |
