---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
model_name: TaskMind — TinyLlama 1.1B Chat LoRA
tags:
  - lora
  - sft
  - peft
  - trl
  - transformers
  - text-classification
  - intent-detection
  - task-management
  - hinglish
  - base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
license: apache-2.0
pipeline_tag: text-generation
language:
  - en
  - hi
metrics:
  - token_accuracy
---

# TaskMind — TinyLlama 1.1B Chat LoRA

A LoRA adapter fine-tuned on [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) for **WhatsApp message intent classification and structured task extraction** in English and Hinglish (Hindi–English code-switch).

Trained entirely on **Apple Silicon MPS (M5 Max)** — no cloud GPU, no cost, 2 minutes 12 seconds.

> 📦 Full pipeline, production API server, test suite, and deployment docs →
> [github.com/vijendradhanotiya/taskmind-ai](https://github.com/vijendradhanotiya/taskmind-ai)

---

## What It Does

Given a raw WhatsApp team message, the model extracts structured intent as JSON — the model itself outputs valid JSON, no regex hacks needed.

**Input:**
```
@Neha the design review is pending from your end
```

**Output:**
```json
{
  "intent": "TASK_ASSIGN",
  "assigneeName": "Neha",
  "project": null,
  "title": "Design review",
  "deadline": null,
  "priority": "normal",
  "progressPercent": null
}
```

---

## Supported Intents

| Intent | Trigger Pattern | Example |
|---|---|---|
| `TASK_ASSIGN` | @mention + action | "@Rohan review the PR I just pushed" |
| `TASK_DONE` | completion language | "done bhai, merged the PR" |
| `TASK_UPDATE` | progress percentage | "login page 60% ho gaya" |
| `TASK_BLOCKED` | blocker / error | "CI/CD pipeline is broken again" |
| `PROGRESS_NOTE` | status update | "deployment failed on prod — rollback initiated" |
| `GENERAL_MESSAGE` | no task signal | "good morning team!", "okay noted" |

---

## Quick Start

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER    = "SatyamSinghal/taskmind-1.1b-chat-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model     = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
model     = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

SYSTEM_PROMPT = (
    "You are TaskMind, an AI that reads WhatsApp messages and extracts structured task data. "
    "Always respond with valid JSON only. No explanation. No markdown."
)

def classify(message: str) -> dict:
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": message},
    ]
    ids = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text, "parse_success": False}

print(classify("@Agrim fix the growstreams deck ASAP"))
```

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Method | LoRA (Low-Rank Adaptation) via SFT |
| LoRA rank | r = 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Trainable params | ~4.2M / 1.1B (0.38%) |
| Dataset size | 131 training + 20 validation examples |
| Epochs | 5 |
| Batch size | 4 |
| Max sequence length | 512 |
| Optimizer | AdamW (paged) |
| Learning rate | 2e-4 with cosine schedule |
| Hardware | Apple M5 Max — MPS backend |
| Training time | 2 minutes 12 seconds |
| Training cost | $0 |

---

## Performance

| Metric | Before Fine-tuning | After Fine-tuning |
|---|---|---|
| Eval loss | 2.28 | **0.39** |
| Token accuracy | 59% | **92.8%** |
| JSON parse success | ~30% | **~97%** |
| Correct intent | Often wrong | **Correct in tested cases** |

### Before vs After — Real Examples

| Message | Base Model | TaskMind |
|---|---|---|
| `@Agrim fix deck ASAP` | Fake deadline 2021-01-01, assignee "John Doe" | `TASK_ASSIGN`, correct title |
| `done bhai, merged the PR` | Fake project "PR-123", wrong intent | `TASK_DONE`, null fields |
| `login page 60% ho gaya` | `TASK_ASSIGN`, hallucinated data | `TASK_UPDATE`, progressPercent=60 |
| `getting 500 error` | Hallucinated task | `GENERAL_MESSAGE` |
| `Sure sir ready for it` | John Doe, fake task | `GENERAL_MESSAGE`, null |

---

## API Server

A production-ready FastAPI server wrapping this adapter is available in the companion repo.

```bash
git clone https://github.com/vijendradhanotiya/taskmind-ai
pip install -r requirements.txt
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8001
```

OpenAI-compatible endpoints included:

```bash
# Classify a WhatsApp message
curl -X POST http://localhost:8001/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "@Vijendra deploy karo production pe aaj raat tak, urgent hai!"}'

# Generic chat completion
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is LoRA?"}], "max_tokens": 150}'
```

---

## Framework Versions

| Library | Version |
|---|---|
| PEFT | 0.18.1 |
| TRL | 1.1.0 |
| Transformers | 4.57.0 |
| PyTorch | 2.2.2 |
| Datasets | 4.8.4 |
| Tokenizers | 0.22.1 |

---

## Contributors

| Name | Role | GitHub |
|---|---|---|
| **Satyam Singhal** | Model training, dataset curation, API development | [@SatyamSinghal](https://github.com/SatyamSinghal) |
| **Vijendra Dhanotiya** | Architecture, deployment, repo maintainer | [@vijendradhanotiya](https://github.com/vijendradhanotiya) |

> Full source, deployment guide, hardware benchmarks, and test suite:
> **[github.com/vijendradhanotiya/taskmind-ai](https://github.com/vijendradhanotiya/taskmind-ai)**

---

## Citation

If you use this model or the TaskMind pipeline in your work:

```bibtex
@misc{taskmind2025,
  title   = {TaskMind: WhatsApp Intent Classification via LoRA Fine-tuning on TinyLlama},
  author  = {Singhal, Satyam and Dhanotiya, Vijendra},
  year    = {2025},
  url     = {https://huggingface.co/SatyamSinghal/taskmind-1.1b-chat-lora},
  note    = {LoRA adapter for TinyLlama-1.1B-Chat-v1.0, trained on Apple Silicon MPS}
}
```

```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward
             and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif
             and Gallouedec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```

---

## License

Apache 2.0 — free to use, modify, and distribute with attribution.
