# TaskMind — Deployment Guide

## Option 1: Run Locally (Mac / Linux)

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
git clone https://github.com/vijendradhanotiya/taskmind-ai.git
cd taskmind-ai

pip install -r requirements.txt

# Download the LoRA adapter from HuggingFace
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('SatyamSinghal/taskmind-1.1b-chat-lora', local_dir='out/taskmind_lora_peft')
print('Adapter ready.')
"

# Start the API server
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

Server is live at http://localhost:8001
Docs at http://localhost:8001/docs

---

## Option 2: Docker

```bash
git clone https://github.com/vijendradhanotiya/taskmind-ai.git
cd taskmind-ai

# Place the adapter in out/taskmind_lora_peft/ first (see Option 1 download step)

docker-compose up --build -d

# Check logs
docker-compose logs -f taskmind-api

# Health check
curl http://localhost:8001/health
```

---

## Option 3: Cloud VM (AWS EC2 / GCP / Hetzner)

```bash
# On a fresh Ubuntu 22.04 VM

# Install Python
sudo apt update && sudo apt install -y python3 python3-pip git

# Clone and install
git clone https://github.com/vijendradhanotiya/taskmind-ai.git
cd taskmind-ai
pip3 install -r requirements.txt

# Download adapter
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('SatyamSinghal/taskmind-1.1b-chat-lora', local_dir='out/taskmind_lora_peft')
"

# Run with systemd or screen
screen -S taskmind
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8001
# Ctrl+A, D to detach
```

For GPU servers (A10, T4), ensure CUDA is installed. The model auto-detects device.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BASE_MODEL` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | HF model ID |
| `ADAPTER_DIR` | out/taskmind_lora_peft | Path to LoRA adapter |
| `REQUIRE_AUTH` | false | Enable API key auth |
| `TASKMIND_API_KEY` | (empty) | API key when auth enabled |
| `RATE_LIMIT_PER_MINUTE` | 60 | Requests per client per minute |
| `MAX_BATCH_SIZE` | 10 | Max messages per batch request |
| `MAX_NEW_TOKENS` | 150 | Max tokens to generate |
| `PORT` | 8001 | Server port |
| `LOG_LEVEL` | INFO | INFO / DEBUG / WARNING |

---

## API Usage

### Single message classification

```bash
curl -X POST http://localhost:8001/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "@Agrim fix the growstreams deck ASAP"}'
```

Response:
```json
{
  "id": "req_abc123",
  "model": "taskmind-1.1b-lora-v1",
  "message": "@Agrim fix the growstreams deck ASAP",
  "result": {
    "intent": "TASK_ASSIGN",
    "assigneeName": "Agrim",
    "project": "Growstreams",
    "title": "Fix growstreams deck",
    "deadline": null,
    "priority": "high",
    "progressPercent": null
  },
  "raw_output": "{\"intent\": \"TASK_ASSIGN\", ...}",
  "parse_success": true,
  "latency_ms": 234.5,
  "timestamp": "2025-04-14T01:30:00Z"
}
```

### Batch classification (up to 10 messages)

```bash
curl -X POST http://localhost:8001/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "@Agrim fix the growstreams deck ASAP",
      "done bhai, merged the PR",
      "login page 60% ho gaya"
    ]
  }'
```

### Enable API key authentication

```bash
export REQUIRE_AUTH=true
export TASKMIND_API_KEY=my_secret_key_here

curl -X POST http://localhost:8001/v1/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my_secret_key_here" \
  -d '{"message": "done bhai"}'
```

### Health check

```bash
curl http://localhost:8001/health
```

---

## Audit Checklist

- [ ] Set `REQUIRE_AUTH=true` and rotate `TASKMIND_API_KEY` every 30 days
- [ ] Enable reverse proxy (nginx) with TLS in production
- [ ] Monitor `/health` endpoint from your uptime tool
- [ ] Store inference logs for audit trail
- [ ] Review `/metrics` for anomalous request volumes
- [ ] Pin model and adapter versions in deployment config
- [ ] Document adapter hash (sha256 of adapter_model.safetensors) for reproducibility
