# TaskMind — Architecture Overview

## System Architecture

```
WhatsApp Message
      |
      v
+------------------+
|  Management      |
|  Portal / Bot    |
+--------+---------+
         |
         | HTTP POST /v1/classify
         v
+------------------+     +------------------+
|  TaskMind API    |     |  Rate Limiter    |
|  (FastAPI)       |<--->|  (in-memory)    |
|  port 8001       |     +------------------+
+--------+---------+
         |
         | model.classify(message)
         v
+------------------+
|  TinyLlama 1.1B  |
|  + LoRA Adapter  |
|  (PEFT, 2.25M    |
|   trainable params)|
+--------+---------+
         |
         | raw JSON string
         v
+------------------+
|  JSON Parser     |
|  (extract + map  |
|   to TaskResult) |
+------------------+
         |
         v
  Structured Response
  {intent, assigneeName,
   project, title, deadline,
   priority, progressPercent}
```

## Model Architecture

```
TinyLlama-1.1B-Chat-v1.0 (Base)
  └── Transformer Layers (22 blocks)
      └── Attention: q_proj, k_proj, v_proj, o_proj
          └── LoRA Adapters injected at q_proj + v_proj
              Rank (r): 16
              Alpha: 32
              Dropout: 0.05
              Trainable params: 2,252,800 (0.2% of total)
```

## Prompt Format (### System style, hardware-agnostic)

```
### System:
You are TaskMind. Read the team WhatsApp message and return ONLY a JSON
object with these exact fields: intent (TASK_ASSIGN / TASK_DONE /
TASK_UPDATE / PROGRESS_NOTE / GENERAL_MESSAGE), assigneeName, project,
title, deadline, priority, progressPercent. Use null for unknown fields.

### Message:
{user_message}

### Response:
{"intent": "...", "assigneeName": "...", ...}
```

## API Module Structure

```
api/
  config.py      -- Settings from environment variables
  schemas.py     -- Pydantic request/response models
  inference.py   -- Model loading, prompt building, inference
  main.py        -- FastAPI app, routes, middleware, lifespan
```

## Training Pipeline

```
taskmind-data/
  train.jsonl (131 examples)   --+
  valid.jsonl (24 examples)    --+--> prep_taskmind.py
                                 |        |
                                 |        v
                         data/task_mgmt_data/
                           train/train.jsonl
                           valid/valid.jsonl
                                 |
                                 v
                          run_taskmind.py
                           (LoRA training via
                            PEFT + SFTTrainer)
                                 |
                                 v
                         out/taskmind_lora_peft/
                           adapter_config.json
                           adapter_model.safetensors
```

## What is MPS?

MPS stands for **Metal Performance Shaders** — Apple's GPU compute framework built into macOS.

- Available on all Apple Silicon Macs (M1, M2, M3, M4, M5 series)
- Provides GPU acceleration for PyTorch via `torch.backends.mps`
- Shares memory with the CPU (unified memory architecture)
- The M5 Pro/Max has 36–48 GB unified memory, making it capable of running 1–7B models entirely in memory
- Training speed: roughly 1.3–1.5 seconds per step on M5 Max
- No CUDA drivers needed — works out of the box on macOS 12.3+

## Infrastructure Achievements

| Milestone | Detail |
|---|---|
| Local fine-tuning on Mac | Full LoRA training in 2 min 12 sec on Apple Silicon |
| Zero cloud GPU cost | $0 training cost using MPS |
| Production API | FastAPI with versioned routes, rate limiting, API key auth |
| Adapter size | 18 MB (vs 2.2 GB full model) |
| HF deployment | Adapter published to huggingface.co |
| Docker ready | Dockerfile + docker-compose for any Linux server |
| Colab compatible | taskmind_train.ipynb works on free Colab GPU |
