import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "http://localhost:8001"
REPORT_DIR = Path("tests/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = REPORT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

CYAN  = "\033[96m"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

CLASSIFY_PROMPTS = [
    "@Agrim fix the growstreams deck ASAP NO Delay",
    "done bhai, merged the PR",
    "login page 60% ho gaya",
    "getting 500 error on registration",
    "Sure sir ready for it",
    "@Arpit please complete the dashboard by Friday",
    "yaar aaj server down tha, fixed kar diya",
    "Payment gateway integration 80% complete",
    "Great work team, keep it up!",
    "@Neha the design review is pending from your end",
    "backend API for user auth is done",
    "database migration complete, tested on staging",
    "@Rohan review the PR I just pushed",
    "koi nahi bhai, ho jayega",
    "CI/CD pipeline is broken again",
    "sprint planning at 3pm today",
    "@Agrim user reported bug in checkout flow, P0",
    "75% of the onboarding flow is ready",
    "good morning team!",
    "@Priya can you update the wireframes by EOD",
    "hotfix deployed to production",
    "testing done, no issues found",
    "deployment failed on prod - rollback initiated",
    "@Shiv send me the API docs",
    "okay noted",
    "@Dev team please review architecture doc before tomorrow",
    "load testing done, handles 500 rps comfortably",
    "feature branch merged, ready for QA",
    "@Arpit standup in 10 mins",
    "LGTM, approved",
]

CHAT_PROMPTS = [
    [{"role": "user", "content": "What is LoRA in simple terms?"}],
    [{"role": "user", "content": "Explain what a fine-tuned model is like I'm 10"}],
    [{"role": "system", "content": "You are a helpful ML assistant."}, {"role": "user", "content": "What is MPS on Apple Silicon?"}],
    [{"role": "user", "content": "What is the difference between training and inference?"}],
    [{"role": "user", "content": "What does PEFT stand for?"}],
    [{"role": "user", "content": "Why is TinyLlama good for fine-tuning?"}],
    [{"role": "system", "content": "Answer in one sentence."}, {"role": "user", "content": "What is a transformer model?"}],
    [{"role": "user", "content": "What is overfitting in machine learning?"}],
    [{"role": "user", "content": "How does attention mechanism work?"}],
    [{"role": "user", "content": "What is the purpose of a tokenizer?"}],
    [{"role": "user", "content": "Explain gradient descent simply"}],
    [{"role": "user", "content": "What is a learning rate in training?"}],
    [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "What is SFT training?"}],
    [{"role": "user", "content": "What is a loss function?"}],
    [{"role": "user", "content": "Why do we use adapters instead of full fine-tuning?"}],
    [{"role": "user", "content": "What does batch size mean in training?"}],
    [{"role": "user", "content": "What is the HuggingFace hub used for?"}],
    [{"role": "user", "content": "Explain what epochs are in model training"}],
    [{"role": "user", "content": "What is a CUDA GPU?"}],
    [{"role": "user", "content": "What is the difference between CPU and GPU training?"}],
    [{"role": "user", "content": "What is FastAPI used for?"}],
    [{"role": "user", "content": "How does REST API work?"}],
    [{"role": "user", "content": "What is Docker and why use it?"}],
    [{"role": "system", "content": "You are a senior engineer."}, {"role": "user", "content": "What is model quantization?"}],
    [{"role": "user", "content": "What is the purpose of warmup steps in training?"}],
    [{"role": "user", "content": "What does eval_loss mean?"}],
    [{"role": "user", "content": "What is a checkpoint in ML training?"}],
    [{"role": "user", "content": "How does temperature affect model output?"}],
    [{"role": "user", "content": "What is top_p sampling?"}],
    [{"role": "user", "content": "What is token accuracy in training metrics?"}],
]

COMPLETION_PROMPTS = [
    "The capital of France is",
    "Machine learning is a branch of",
    "LoRA stands for",
    "Fine-tuning a model means",
    "The purpose of a neural network is",
    "Python is a programming language that",
    "Apple Silicon uses MPS which stands for",
    "The most important metric during training is",
    "A transformer model works by",
    "Gradient descent is an algorithm that",
    "The HuggingFace library allows developers to",
    "An API endpoint accepts",
    "FastAPI is a Python framework for",
    "Docker containers help in",
    "A training epoch is",
    "The loss function measures",
    "Overfitting occurs when",
    "A batch size of 4 means",
    "The tokenizer converts",
    "Inference is the process of",
    "A LoRA adapter adds",
    "The learning rate controls",
    "Warmup steps help the optimizer",
    "The eval dataset is used to",
    "A checkpoint saves",
    "Token accuracy measures",
    "The base model contains",
    "Model quantization reduces",
    "An attention mechanism helps the model",
    "The adapter weights are stored in",
]

results = []
totals = {"passed": 0, "failed": 0}


def log(color, symbol, label, message, latency_ms=None):
    lat = f" {DIM}({latency_ms:.0f}ms){RESET}" if latency_ms is not None else ""
    print(f"  {color}{symbol}{RESET} {BOLD}{label}{RESET}  {message}{lat}")


def call(method, path, body=None, label=""):
    url = BASE_URL + path
    t0 = time.perf_counter()
    try:
        if method == "GET":
            r = requests.get(url, timeout=60)
        else:
            r = requests.post(url, json=body, timeout=60)
        latency_ms = (time.perf_counter() - t0) * 1000
        ok = r.status_code < 400
        return r.status_code, r.json(), latency_ms, ok, None
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return 0, {}, latency_ms, False, str(e)


def record(endpoint, input_text, status, latency_ms, raw_response, ok, extra=None):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "input": str(input_text)[:300],
        "status_code": status,
        "latency_ms": round(latency_ms, 1),
        "ok": ok,
        "response_summary": json.dumps(raw_response)[:400],
        "intent": raw_response.get("result", {}).get("intent", "") if isinstance(raw_response, dict) else "",
        "parse_success": raw_response.get("parse_success", "") if isinstance(raw_response, dict) else "",
        "error": extra or "",
    }
    results.append(row)
    if ok:
        totals["passed"] += 1
    else:
        totals["failed"] += 1
    return row


def section(title):
    print(f"\n{BOLD}{CYAN}{'='*58}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*58}{RESET}")


def run_classify():
    section("POST /v1/classify  (30 prompts)")
    for i, msg in enumerate(CLASSIFY_PROMPTS):
        status, data, latency, ok, err = call("POST", "/v1/classify", {"message": msg})
        row = record("/v1/classify", msg, status, latency, data, ok, err)
        intent = data.get("result", {}).get("intent", "?") if ok and data.get("result") else "NO_PARSE"
        symbol = "✓" if ok else "✗"
        color = GREEN if ok else RED
        log(color, symbol, f"[{i+1:02d}]", f"{msg[:45]:<45} → {intent}", latency)


def run_batch():
    section("POST /v1/batch  (3 batches of 10)")
    batches = [CLASSIFY_PROMPTS[:10], CLASSIFY_PROMPTS[10:20], CLASSIFY_PROMPTS[20:30]]
    for i, batch in enumerate(batches):
        status, data, latency, ok, err = call("POST", "/v1/batch", {"messages": batch})
        successful = data.get("successful", 0) if ok else 0
        total = data.get("total", len(batch))
        record("/v1/batch", f"batch_{i+1}[{len(batch)} msgs]", status, latency, data, ok, err)
        symbol = "✓" if ok else "✗"
        color = GREEN if ok else RED
        log(color, symbol, f"batch[{i+1}]", f"{len(batch)} messages → {successful}/{total} parsed  HTTP {status}", latency)


def run_chat():
    section("POST /v1/chat/completions  (30 messages)")
    for i, messages in enumerate(CHAT_PROMPTS):
        user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
        status, data, latency, ok, err = call("POST", "/v1/chat/completions", {
            "messages": messages, "max_tokens": 80, "temperature": 0.7
        })
        answer = ""
        if ok and data.get("choices"):
            answer = data["choices"][0]["message"]["content"][:60]
        record("/v1/chat/completions", user_text, status, latency, data, ok, err)
        symbol = "✓" if ok else "✗"
        color = GREEN if ok else RED
        log(color, symbol, f"[{i+1:02d}]", f"{user_text[:40]:<40} → {answer[:55]}", latency)


def run_completions():
    section("POST /v1/completions  (30 prompts)")
    for i, prompt in enumerate(COMPLETION_PROMPTS):
        status, data, latency, ok, err = call("POST", "/v1/completions", {
            "prompt": prompt, "max_tokens": 40, "temperature": 0.7
        })
        answer = ""
        if ok and data.get("choices"):
            answer = data["choices"][0]["text"][:55]
        record("/v1/completions", prompt, status, latency, data, ok, err)
        symbol = "✓" if ok else "✗"
        color = GREEN if ok else RED
        log(color, symbol, f"[{i+1:02d}]", f"{prompt[:40]:<40} → {answer}", latency)


def run_ops():
    section("Ops Endpoints  (/health  /metrics  /v1/models)")
    for method, path in [("GET", "/health"), ("GET", "/metrics"), ("GET", "/v1/models")]:
        status, data, latency, ok, err = call(method, path)
        record(path, path, status, latency, data, ok, err)
        symbol = "✓" if ok else "✗"
        color = GREEN if ok else RED
        summary = json.dumps(data)[:80]
        log(color, symbol, path, f"HTTP {status}  {summary}", latency)


def save_report():
    fieldnames = ["timestamp", "endpoint", "input", "status_code", "latency_ms",
                  "ok", "response_summary", "intent", "parse_success", "error"]
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n{BOLD}{GREEN}  Report saved → {REPORT_FILE}{RESET}")


def print_summary():
    total = totals["passed"] + totals["failed"]
    avg_lat = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    p_lat = sorted(r["latency_ms"] for r in results)
    p95 = p_lat[int(len(p_lat) * 0.95)] if p_lat else 0
    print(f"\n{BOLD}{CYAN}{'='*58}")
    print(f"  SUMMARY")
    print(f"{'='*58}{RESET}")
    print(f"  {GREEN}Passed : {totals['passed']}/{total}{RESET}")
    if totals["failed"]:
        print(f"  {RED}Failed : {totals['failed']}/{total}{RESET}")
    print(f"  {YELLOW}Avg latency : {avg_lat:.0f}ms{RESET}")
    print(f"  {YELLOW}p95 latency : {p95:.0f}ms{RESET}")
    print(f"  {DIM}Report      : {REPORT_FILE}{RESET}\n")


if __name__ == "__main__":
    print(f"\n{BOLD}{CYAN}")
    print("  ████████╗ █████╗ ███████╗██╗  ██╗███╗   ███╗██╗███╗   ██╗██████╗ ")
    print("     ██╔══╝██╔══██╗██╔════╝██║ ██╔╝████╗ ████║██║████╗  ██║██╔══██╗")
    print("     ██║   ███████║███████╗█████╔╝ ██╔████╔██║██║██╔██╗ ██║██║  ██║")
    print("     ██║   ██╔══██║╚════██║██╔═██╗ ██║╚██╔╝██║██║██║╚██╗██║██║  ██║")
    print("     ██║   ██║  ██║███████║██║  ██╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝")
    print("     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ ")
    print(f"  API Test Suite  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")

    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            raise ConnectionError
        print(f"  {GREEN}Server is up at {BASE_URL}{RESET}")
    except Exception:
        print(f"  {RED}Cannot reach server at {BASE_URL}")
        print(f"  Start it first: python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8001{RESET}\n")
        sys.exit(1)

    run_ops()
    run_classify()
    run_batch()
    run_chat()
    run_completions()
    save_report()
    print_summary()
