import json
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(title="TaskMind API", version="1.0")

SYSTEM_MSG = (
    "You are TaskMind. Read the team WhatsApp message and return ONLY a JSON "
    "object with these exact fields: intent (TASK_ASSIGN / TASK_DONE / "
    "TASK_UPDATE / PROGRESS_NOTE / GENERAL_MESSAGE), assigneeName, project, "
    "title, deadline, priority, progressPercent. Use null for unknown fields."
)

BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "out/taskmind_lora_peft")

tokenizer = None
model     = None


def load_model():
    global tokenizer, model
    if model is not None:
        return

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if os.path.isdir(ADAPTER_DIR):
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        print(f"Loaded LoRA adapter from {ADAPTER_DIR}")
    else:
        model = base
        print(f"No adapter found at {ADAPTER_DIR}, using base model")

    if device != "cuda":
        model = model.to(device)
    model.eval()


@app.on_event("startup")
async def startup_event():
    load_model()


class MessageRequest(BaseModel):
    message: str


class TaskResponse(BaseModel):
    raw_output: str
    parsed: dict | None = None
    success: bool


@app.post("/classify", response_model=TaskResponse)
def classify(req: MessageRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    prompt = (
        "### System:\n" + SYSTEM_MSG
        + "\n\n### Message:\n" + req.message
        + "\n\n### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()

    parsed = None
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        pass

    return TaskResponse(raw_output=raw, parsed=parsed, success=parsed is not None)


@app.get("/health")
def health():
    return {"status": "ok", "model": BASE_MODEL, "adapter": ADAPTER_DIR}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("taskmind_fastapi:app", host="0.0.0.0", port=8001, reload=False)
