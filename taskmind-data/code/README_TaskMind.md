---
license: mit
language:
  - en
  - hi
tags:
  - task-management
  - intent-classification
  - nlp
  - team-productivity
  - hinglish
  - whatsapp
  - json-extraction
  - slot-filling
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
pipeline_tag: text-generation
library_name: mlx-lm
datasets:
  - SatyamSinghal/team-chat-intent-v1
---

# TaskMind-1.1B — Team Chat Intent Classifier

**TaskMind** is a LoRA-fine-tuned language model that extracts structured task information from team chat messages in real time. It is trained on real WhatsApp/Slack conversations from a blockchain startup team, making it uniquely capable of handling Indian English, Hinglish, informal language, emojis, and the ambiguous short-form messages real teams actually send.

---

## What It Does

Given any team chat message, TaskMind outputs a structured JSON object:

```json
{
  "intent":          "TASK_ASSIGN",
  "assigneeName":    "Agrim",
  "project":         "Growstreams",
  "title":           "Fix pitch deck",
  "deadline":        "ASAP",
  "priority":        "urgent",
  "progressPercent": null
}
```

### The 5 Intent Classes

| Intent | Description | Example |
|--------|-------------|---------|
| `TASK_ASSIGN` | Someone is giving work to someone | "@Agrim fix the deck ASAP" |
| `TASK_DONE` | Work is complete | "done bhai, merged the PR" |
| `TASK_UPDATE` | Blocked, delayed, or needs attention | "getting 500 error on registration" |
| `PROGRESS_NOTE` | In-progress status with optional % | "login page 60% ho gaya" |
| `GENERAL_MESSAGE` | No actionable content | "Sure sir", "Okayy 🙌", "🔥🔥🔥" |

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Fine-tune method | LoRA (rank 16, alpha 16) |
| Framework | Apple MLX (`mlx-lm`) |
| Training hardware | Apple M5 Pro Max (36 GB unified memory) |
| Training examples | 188 real team chat messages |
| Train / Valid split | 159 / 29 |
| Iterations | 300 |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Max sequence length | 256 |
| Trainable parameters | ~4.6M / 1.1B (0.42%) |
| Adapter size | ~18 MB |

---

## Unique Capabilities

Unlike models trained on clean synthetic data, TaskMind handles:

- **Hinglish**: "done bhai", "yaar login wala karo", "ho gaya 60%"
- **Urgency markers**: ASAP, NO Delay, aaj tak, immediately
- **URL-as-completion**: A shared deployment link = TASK_DONE
- **False-positive filtering**: "Sure sir ready for it" → GENERAL_MESSAGE (not TASK_ASSIGN)
- **Noise filtering**: emoji-only, deleted messages, image omitted → GENERAL_MESSAGE
- **Question-form assignment**: "Can you do api testing?" → TASK_ASSIGN
- **Short completions**: "Done.", "Deployed.", "Fixed and pushed." → TASK_DONE

---

## Usage

### Option A — MLX (fastest on Apple Silicon)

```python
from mlx_lm import load, generate
import json

model, tokenizer = load(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_path="SatyamSinghal/taskmind-1.1b-chat-lora"
)

def classify(message: str) -> dict:
    system = (
        "You are TaskMind, an intelligent task management AI that extracts structured task information "
        "from team chat messages. Output valid JSON only with fields: intent, assigneeName, project, "
        "title, deadline, priority, progressPercent. "
        "Intent must be one of: TASK_ASSIGN, TASK_DONE, TASK_UPDATE, PROGRESS_NOTE, GENERAL_MESSAGE."
    )
    prompt = f"<|system|>\n{system}\n<|user|>\n{message}\n<|assistant|>\n"
    response = generate(model, tokenizer, prompt, max_tokens=128)
    return json.loads(response.strip())

# Examples
print(classify("@Agrim fix the deck ASAP NO Delay"))
print(classify("done bhai, merged the PR"))
print(classify("login page 60% ho gaya"))
print(classify("getting 500 error on registration"))
print(classify("Sure sir ready for it 🔥"))
```

### Option B — Ollama (after GGUF export)

```bash
ollama run taskmind "classify: @Sarthak finish the canton demo video by friday"
```

### Option C — FastAPI endpoint

```python
from fastapi import FastAPI
from mlx_lm import load, generate
import json

app = FastAPI(title="TaskMind API")
model, tokenizer = load(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_path="SatyamSinghal/taskmind-1.1b-chat-lora"
)
SYSTEM = "You are TaskMind... (system prompt)"

@app.post("/classify")
async def classify(message: str):
    prompt = f"<|system|>\n{SYSTEM}\n<|user|>\n{message}\n<|assistant|>\n"
    raw = generate(model, tokenizer, prompt, max_tokens=128)
    return json.loads(raw.strip())
```

---

## Training Your Own Version

```bash
# 1. Clone and set up
git clone https://github.com/SatyamSinghal/taskmind
cd taskmind && pip install mlx mlx-lm datasets huggingface_hub

# 2. Prepare data (already in this repo under data/)
# train.jsonl — 159 examples
# valid.jsonl — 29 examples

# 3. Fine-tune
python -m mlx_lm lora \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train \
  --data data/train \
  --batch-size 4 \
  --iters 300 \
  --learning-rate 1e-4 \
  --num-layers 16 \
  --max-seq-length 256 \
  --mask-prompt \
  --adapter-path out/taskmind_v1

# 4. Test
python -m mlx_lm generate \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path out/taskmind_v1 \
  --prompt "<|system|>\nYou are TaskMind...\n<|user|>\ndone bhai\n<|assistant|>\n" \
  --max-tokens 100
```

---

## Integration with Management Portal

This model is the AI core of a WhatsApp → Task Management Portal pipeline:

```
WhatsApp message arrives
       ↓
Webhook (FastAPI) receives message
       ↓
TaskMind classifies → JSON
       ↓
If intent == TASK_ASSIGN: create task card in portal
If intent == TASK_DONE:   mark task complete
If intent == TASK_UPDATE: flag as blocked
If intent == PROGRESS_NOTE: update progress bar
If intent == GENERAL_MESSAGE: ignore
       ↓
Management portal updates in real time
```

No forms. No manual entry. Team talks normally, portal updates automatically.

---

## Limitations

- Context-dependent "Done." (no task reference) outputs null title — app must infer from thread context
- Multi-task numbered lists ("priorities: 1. X 2. Y") extracts only the first task — pre-process before calling
- Best accuracy on English and Hinglish; pure Hindi messages may misclassify
- 188 training examples — production use should augment to 500+ for >90% accuracy

---

## Dataset

Training data: `SatyamSinghal/team-chat-intent-v1`

Real WhatsApp group messages from a blockchain startup team (Compliledger / BlockX AI), manually labelled for task intent, with team member names anonymised in the public release.

---

## Citation

```bibtex
@misc{taskmind2026,
  title   = {TaskMind: Real-Team Chat Intent Classification via LoRA Fine-Tuning},
  author  = {Satyam Singhal},
  year    = {2026},
  url     = {https://huggingface.co/SatyamSinghal/taskmind-1.1b-chat-lora}
}
```

---

## License

MIT. Free to use, fine-tune, and deploy.

Trained on Apple M5 Pro Max using Apple MLX. Part of the Ginie AI / Compliledger open-source ML stack.
