# TaskMind AI

**Fine-tuned TinyLlama-1.1B for WhatsApp task extraction — trained locally on Apple Silicon, deployed as a production FastAPI service.**

> Reads team WhatsApp messages and returns structured JSON: intent, assignee, project, title, deadline, priority, progress.

---

## What Was Built and Achieved

| Milestone | Detail |
|---|---|
| Local LoRA fine-tuning | Trained TinyLlama-1.1B with PEFT LoRA in 2 min 12 sec on Apple M5 MPS |
| Zero cloud cost | $0 GPU spend — used Apple Silicon unified memory |
| Loss: 2.28 to 0.39 | 5 epochs, 131 training examples, token accuracy 59% to 92.8% |
| Clean JSON output | No hallucinations, no fake assignees, no backtick noise |
| Production API | FastAPI v1 with rate limiting, API key auth, batch endpoint |
| HF deployment | Adapter published at huggingface.co/SatyamSinghal/taskmind-1.1b-chat-lora |
| Docker ready | One-command deployment on any Linux server |
| Colab notebook | Full train pipeline works on free Colab GPU |

---

## What is MPS?

**MPS (Metal Performance Shaders)** is Apple's GPU compute framework for Apple Silicon Macs (M1/M2/M3/M4/M5). PyTorch uses it via `torch.backends.mps`.

- The M5 Pro/Max has 36–48 GB unified memory shared between CPU and GPU
- This means you can run and train 1–7B parameter models entirely in RAM — no VRAM limit
- Training speed on M5: ~1.3 seconds/step (vs ~0.3s on a cloud A100 — but free)
- No CUDA, no cloud, no billing — just plug in and train

---

## Before vs After Training

| Message | Base Model Output | TaskMind Output |
|---|---|---|
| `@Agrim fix deck ASAP` | Fake deadline 2021-01-01, John Doe, code block noise | Clean JSON, correct intent |
| `done bhai, merged the PR` | Fake assignee "assistant", fake project PR-123 | `TASK_DONE`, null fields |
| `login page 60% ho gaya` | TASK_ASSIGN, fake data | `TASK_UPDATE`, progressPercent=60 |
| `getting 500 error` | TASK_ASSIGN, hallucinated task | `GENERAL_MESSAGE`, null |
| `Sure sir ready for it` | TASK_ASSIGN, John Doe | `GENERAL_MESSAGE`, null |

---

## Test Results (96/96 passing)

Full API test suite run on M5 Max — all endpoints, 93 LLM calls:

| Endpoint | Calls | Passed | Avg Latency |
|---|---|---|---|
| `/health` `/metrics` `/v1/models` | 3 | 3/3 | 1.4ms |
| `/v1/classify` | 30 | 30/30 | ~1800ms |
| `/v1/batch` (10 msgs each) | 3 | 3/3 | ~5200ms |
| `/v1/chat/completions` | 30 | 30/30 | ~2500ms |
| `/v1/completions` | 30 | 30/30 | ~1100ms |

Run the suite yourself: `bash tests/run_tests.sh` — saves full CSV report to `tests/reports/`

---

## Hardware — M4 vs M5 Pro vs M5 Max

| | M4 (16 GB) | M5 Pro (24 GB) | **M5 Max (48 GB)** |
|---|---|---|---|
| Training time | ~5m 30s | ~2m 45s | **2m 12s ✓ measured** |
| Inference p50 | ~420ms | ~270ms | **~230ms** |
| Max trainable model | 1.1B–3B | 3B–7B | **7B–13B** |
| Training cost | $0 | $0 | **$0** |

See [`docs/HARDWARE_COMPARISON.md`](docs/HARDWARE_COMPARISON.md) for full breakdown including memory, cost, and model size limits.

---

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, MPS explained, prompt format, data flow |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Loss curve, before/after comparison, latency benchmarks |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Local / Docker / cloud deploy steps + audit checklist |
| [`docs/HARDWARE_COMPARISON.md`](docs/HARDWARE_COMPARISON.md) | M4 vs M5 Pro vs M5 Max — training speed, memory, cost |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/vijendradhanotiya/taskmind-ai.git
cd taskmind-ai
pip install -r requirements.txt
```

### 2. Download adapter from HuggingFace

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('SatyamSinghal/taskmind-1.1b-chat-lora', local_dir='out/taskmind_lora_peft')
print('Adapter ready.')
"
```

### 3. Start the API server

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

- API docs: http://localhost:8001/docs
- Health: http://localhost:8001/health

### 4. Test it

```bash
curl -X POST http://localhost:8001/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "@Agrim fix the growstreams deck ASAP"}'
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /v1/classify | Classify a single WhatsApp message |
| POST | /v1/batch | Classify up to 10 messages |
| GET | /health | Liveness + readiness check |
| GET | /metrics | Request counts and uptime |
| GET | /docs | Swagger UI |
| GET | /redoc | ReDoc UI |

---

## Repository Structure

```
taskmind-ai/
  api/
    config.py        -- Environment-driven settings
    schemas.py       -- Pydantic request/response models
    inference.py     -- Model load, prompt build, inference
    main.py          -- FastAPI app (lifespan, routes, middleware)
  docs/
    ARCHITECTURE.md  -- System design, model architecture, MPS explained
    DEPLOYMENT.md    -- Local, Docker, cloud deployment + audit checklist
    PERFORMANCE.md   -- Training metrics, loss curve, before/after table
  scripts/
    upload_to_hf.py  -- Upload adapter to HuggingFace Hub
  training/
    run_taskmind.py  -- End-to-end training script (test + train + test)
    prep_taskmind.py -- Convert raw JSONL to prompt/completion format
    make_notebook.py -- Generates taskmind_train.ipynb
  taskmind-data/
    train.jsonl      -- 131 labeled WhatsApp messages
    valid.jsonl      -- 24 validation examples
  taskmind_train.ipynb  -- Colab-ready training notebook
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
```

---

## Training Your Own Adapter

```bash
# Prepare data
python3 training/prep_taskmind.py

# Train (runs test before + train + test after in one shot)
python3 training/run_taskmind.py

# Adapter saved to out/taskmind_lora_peft/

# Upload to HF
export HF_TOKEN=hf_xxx
python3 scripts/upload_to_hf.py
```

Or open `taskmind_train.ipynb` in Google Colab (free GPU).

---

## Docker Deployment

```bash
docker-compose up --build -d
curl http://localhost:8001/health
```

---

## Model Card

- **Base**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Adapter**: SatyamSinghal/taskmind-1.1b-chat-lora (HuggingFace)
- **Method**: LoRA (r=16, alpha=32, target: q_proj + v_proj)
- **Dataset**: 155 real WhatsApp messages from BlockX/CompliLedger team
- **Intents**: TASK_ASSIGN, TASK_DONE, TASK_UPDATE, PROGRESS_NOTE, GENERAL_MESSAGE
- **Trained on**: Apple M5 MPS, Python 3.12, PyTorch 2.2, transformers 4.57, trl 1.1, peft 0.18

---

## License

MIT — see LICENSE
