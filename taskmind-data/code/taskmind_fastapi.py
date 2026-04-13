"""
TaskMind FastAPI Server
Endpoint: POST /classify  →  structured JSON intent from any team chat message
Run:      uvicorn taskmind_fastapi:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json, os
from mlx_lm import load, generate

# ── System prompt (same as training) ─────────────────────────────────────────
SYSTEM = (
    "You are TaskMind, an intelligent task management AI that extracts structured "
    "task information from team chat messages. Output valid JSON only with fields: "
    "intent, assigneeName, project, title, deadline, priority, progressPercent. "
    "Intent must be one of: TASK_ASSIGN, TASK_DONE, TASK_UPDATE, PROGRESS_NOTE, GENERAL_MESSAGE."
)

# ── Load model once on startup ────────────────────────────────────────────────
ADAPTER = os.getenv("TASKMIND_ADAPTER", "out/taskmind_v1")
BASE    = os.getenv("TASKMIND_BASE",    "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print(f"Loading TaskMind from {BASE} + {ADAPTER}...")
model, tokenizer = load(BASE, adapter_path=ADAPTER)
print("Model ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="TaskMind API",
    description="Real-time team chat → structured task JSON. Built on TinyLlama-1.1B + LoRA.",
    version="1.0.0"
)

class ClassifyRequest(BaseModel):
    message: str
    sender: str | None = None      # optional: who sent it
    project_hint: str | None = None # optional: which project context

class TaskOutput(BaseModel):
    intent: str
    assigneeName: str | list | None
    project: str | None
    title: str | None
    deadline: str | None
    priority: str | None
    progressPercent: int | None
    sender: str | None
    raw_message: str

@app.post("/classify", response_model=TaskOutput)
async def classify(req: ClassifyRequest):
    """
    Classify a team chat message into a structured task object.

    - **TASK_ASSIGN** — someone is assigning work
    - **TASK_DONE** — work is complete
    - **TASK_UPDATE** — blocked or needs attention
    - **PROGRESS_NOTE** — in-progress status
    - **GENERAL_MESSAGE** — no actionable content
    """
    # Build prompt with optional context
    user_content = req.message
    if req.project_hint:
        user_content = f"[Project context: {req.project_hint}]\n{req.message}"

    prompt = f"<|system|>\n{SYSTEM}\n<|user|>\n{user_content}\n<|assistant|>\n"

    raw = generate(model, tokenizer, prompt, max_tokens=128).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON from partial output
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end])
            except:
                raise HTTPException(status_code=422, detail=f"Model output not valid JSON: {raw}")
        else:
            raise HTTPException(status_code=422, detail=f"Model output not valid JSON: {raw}")

    return TaskOutput(
        intent          = parsed.get("intent", "GENERAL_MESSAGE"),
        assigneeName    = parsed.get("assigneeName"),
        project         = parsed.get("project"),
        title           = parsed.get("title"),
        deadline        = parsed.get("deadline"),
        priority        = parsed.get("priority"),
        progressPercent = parsed.get("progressPercent"),
        sender          = req.sender,
        raw_message     = req.message
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": BASE, "adapter": ADAPTER}

@app.get("/")
async def root():
    return {
        "service": "TaskMind API",
        "version": "1.0.0",
        "endpoint": "POST /classify",
        "docs": "/docs"
    }
