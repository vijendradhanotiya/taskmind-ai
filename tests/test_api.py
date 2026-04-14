import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL   = "http://localhost:8001"
DATASET    = Path(__file__).parent / "datasets" / "prompts_1000.jsonl"
REPORT_DIR = Path(__file__).parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = REPORT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

CYAN   = "\033[96m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

results = []
totals  = {"passed": 0, "failed": 0}
domain_stats: dict = defaultdict(lambda: {"total": 0, "passed": 0,
                                          "intent_correct": 0, "intent_total": 0,
                                          "latencies": []})


def load_dataset(sample: int | None) -> dict:
    if not DATASET.exists():
        print(f"{RED}Dataset not found: {DATASET}")
        print(f"Run: python3 tests/datasets/generate_dataset.py{RESET}")
        sys.exit(1)
    rows = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    by_ep: dict = defaultdict(list)
    for r in rows:
        by_ep[r["endpoint"]].append(r)
    if sample:
        sampled: dict = defaultdict(list)
        for ep, items in by_ep.items():
            random.shuffle(items)
            sampled[ep] = items[:sample]
        return sampled
    return by_ep


def log(color, symbol, label, message, latency_ms=None):
    lat = f" {DIM}({latency_ms:.0f}ms){RESET}" if latency_ms is not None else ""
    print(f"  {color}{symbol}{RESET} {BOLD}{label}{RESET}  {message}{lat}")


def call(method, path, body=None):
    url = BASE_URL + path
    t0  = time.perf_counter()
    try:
        r = requests.get(url, timeout=90) if method == "GET" \
            else requests.post(url, json=body, timeout=90)
        ms = (time.perf_counter() - t0) * 1000
        return r.status_code, r.json(), ms, r.status_code < 400, None
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return 0, {}, ms, False, str(e)


def record(endpoint, domain, input_text, status, latency_ms, raw, ok,
           expected_intent=None, actual_intent=None, err=None):
    intent_correct = ""
    if expected_intent and actual_intent:
        intent_correct = expected_intent.upper() == actual_intent.upper()

    row = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "endpoint":        endpoint,
        "domain":          domain,
        "input":           str(input_text)[:300],
        "status_code":     status,
        "latency_ms":      round(latency_ms, 1),
        "ok":              ok,
        "expected_intent": expected_intent or "",
        "actual_intent":   actual_intent  or "",
        "intent_correct":  intent_correct,
        "parse_success":   raw.get("parse_success", "") if isinstance(raw, dict) else "",
        "response_summary": json.dumps(raw)[:350],
        "error":           err or "",
    }
    results.append(row)
    ds = domain_stats[domain]
    ds["total"]    += 1
    ds["latencies"].append(latency_ms)
    if ok:
        totals["passed"] += 1
        ds["passed"] += 1
    else:
        totals["failed"] += 1
    if expected_intent and actual_intent:
        ds["intent_total"] += 1
        if intent_correct:
            ds["intent_correct"] += 1
    return row


def section(title):
    print(f"\n{BOLD}{CYAN}{'='*62}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*62}{RESET}")


def run_classify(items: list):
    section(f"POST /v1/classify  ({len(items)} prompts from dataset)")
    for i, item in enumerate(items):
        msg = item["input"]
        exp = item.get("expected_intent")
        status, data, ms, ok, err = call("POST", "/v1/classify", {"message": msg})
        actual = data.get("result", {}).get("intent") if ok and data.get("result") else None
        correct = (exp and actual and exp.upper() == actual.upper())
        row = record("/v1/classify", item["domain"], msg, status, ms, data, ok, exp, actual, err)
        display_intent = actual or "NO_PARSE"
        match_sym = f"{GREEN}✓{RESET}" if correct else (f"{RED}✗{RESET}" if exp else "")
        symbol = "✓" if ok else "✗"
        color  = GREEN if ok else RED
        log(color, symbol, f"[{i+1:03d}]",
            f"{msg[:40]:<40} → {display_intent:<18} {match_sym}", ms)


def run_batch(classify_items: list):
    batch_msgs = [i["input"] for i in classify_items[:30]]
    section("POST /v1/batch  (3 batches of 10)")
    for idx in range(3):
        batch = batch_msgs[idx*10:(idx+1)*10]
        status, data, ms, ok, err = call("POST", "/v1/batch", {"messages": batch})
        successful = data.get("successful", 0) if ok else 0
        total = data.get("total", len(batch))
        record("/v1/batch", "batch", f"batch_{idx+1}[{len(batch)} msgs]",
               status, ms, data, ok, err=err)
        symbol = "✓" if ok else "✗"
        color  = GREEN if ok else RED
        log(color, symbol, f"batch[{idx+1}]",
            f"{len(batch)} messages → {successful}/{total} parsed  HTTP {status}", ms)


def run_chat(items: list):
    section(f"POST /v1/chat/completions  ({len(items)} messages from dataset)")
    for i, item in enumerate(items):
        q = item["input"]
        messages = [{"role": "user", "content": q}]
        status, data, ms, ok, err = call("POST", "/v1/chat/completions",
                                         {"messages": messages, "max_tokens": 80, "temperature": 0.7})
        answer = ""
        if ok and data.get("choices"):
            answer = data["choices"][0]["message"]["content"][:55]
        record("/v1/chat/completions", item["domain"], q, status, ms, data, ok, err=err)
        symbol = "✓" if ok else "✗"
        color  = GREEN if ok else RED
        log(color, symbol, f"[{i+1:03d}]", f"{q[:38]:<38} → {answer}", ms)


def run_completions(items: list):
    section(f"POST /v1/completions  ({len(items)} prompts from dataset)")
    for i, item in enumerate(items):
        prompt = item["input"]
        status, data, ms, ok, err = call("POST", "/v1/completions",
                                         {"prompt": prompt, "max_tokens": 50, "temperature": 0.7})
        answer = ""
        if ok and data.get("choices"):
            answer = data["choices"][0]["text"][:55]
        record("/v1/completions", item["domain"], prompt, status, ms, data, ok, err=err)
        symbol = "✓" if ok else "✗"
        color  = GREEN if ok else RED
        log(color, symbol, f"[{i+1:03d}]", f"{prompt[:38]:<38} → {answer}", ms)


def run_ops():
    section("Ops Endpoints  (/health  /metrics  /v1/models)")
    for method, path in [("GET", "/health"), ("GET", "/metrics"), ("GET", "/v1/models")]:
        status, data, ms, ok, err = call(method, path)
        record(path, "ops", path, status, ms, data, ok, err=err)
        symbol = "✓" if ok else "✗"
        color  = GREEN if ok else RED
        log(color, symbol, path, f"HTTP {status}  {json.dumps(data)[:80]}", ms)


def save_report():
    fieldnames = ["timestamp","endpoint","domain","input","status_code","latency_ms",
                  "ok","expected_intent","actual_intent","intent_correct",
                  "parse_success","response_summary","error"]
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames).writerows(results)
    print(f"\n{BOLD}{GREEN}  Report saved → {REPORT_FILE}{RESET}")


def print_summary():
    total    = totals["passed"] + totals["failed"]
    lats     = sorted(r["latency_ms"] for r in results)
    avg_lat  = sum(lats) / len(lats) if lats else 0
    p95      = lats[int(len(lats) * 0.95)] if lats else 0

    classify_rows = [r for r in results if r["endpoint"] == "/v1/classify"
                     and r["expected_intent"]]
    intent_total   = len(classify_rows)
    intent_correct = sum(1 for r in classify_rows if r["intent_correct"] is True)
    intent_acc     = (intent_correct / intent_total * 100) if intent_total else 0

    print(f"\n{BOLD}{CYAN}{'='*62}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*62}{RESET}")
    print(f"  {GREEN}HTTP pass   : {totals['passed']}/{total}{RESET}")
    if totals["failed"]:
        print(f"  {RED}HTTP fail   : {totals['failed']}/{total}{RESET}")
    print(f"  {YELLOW}Avg latency : {avg_lat:.0f}ms{RESET}")
    print(f"  {YELLOW}p95 latency : {p95:.0f}ms{RESET}")
    if intent_total:
        acc_color = GREEN if intent_acc >= 80 else (YELLOW if intent_acc >= 60 else RED)
        print(f"  {acc_color}Intent accuracy : {intent_correct}/{intent_total} "
              f"= {intent_acc:.1f}%{RESET}")

    print(f"\n{BOLD}  Per-Domain Accuracy (classify only){RESET}")
    print(f"  {'Domain':<22} {'Calls':>6} {'HTTP%':>6} {'Intent%':>9} {'AvgMs':>7}")
    print(f"  {'-'*54}")
    for domain, ds in sorted(domain_stats.items()):
        if ds["intent_total"] == 0:
            continue
        http_pct   = ds["passed"] / ds["intent_total"] * 100
        intent_pct = ds["intent_correct"] / ds["intent_total"] * 100
        avg_ms     = sum(ds["latencies"]) / len(ds["latencies"])
        col = GREEN if intent_pct >= 80 else (YELLOW if intent_pct >= 60 else RED)
        print(f"  {domain:<22} {ds['intent_total']:>6} {http_pct:>5.0f}% "
              f"{col}{intent_pct:>8.1f}%{RESET} {avg_ms:>7.0f}ms")

    print(f"\n  {DIM}Report → {REPORT_FILE}{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TaskMind API Test Suite")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--sample", type=int, default=30, metavar="N",
                     help="Prompts per endpoint to sample (default: 30)")
    grp.add_argument("--all", action="store_true",
                     help="Run all 1000 prompts (slow — ~30 min)")
    args = parser.parse_args()
    sample = None if args.all else args.sample

    print(f"\n{BOLD}{CYAN}")
    print("  ████████╗ █████╗ ███████╗██╗  ██╗███╗   ███╗██╗███╗   ██╗██████╗ ")
    print("     ██╔══╝██╔══██╗██╔════╝██║ ██╔╝████╗ ████║██║████╗  ██║██╔══██╗")
    print("     ██║   ███████║███████╗█████╔╝ ██╔████╔██║██║██╔██╗ ██║██║  ██║")
    print("     ██║   ██╔══██║╚════██║██╔═██╗ ██║╚██╔╝██║██║██║╚██╗██║██║  ██║")
    print("     ██║   ██║  ██║███████║██║  ██╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝")
    print("     ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ ")
    mode_label = "ALL 1000 prompts" if args.all else f"sample={sample} per endpoint"
    print(f"  API Test Suite  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  [{mode_label}]{RESET}\n")

    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            raise ConnectionError
        print(f"  {GREEN}Server is up at {BASE_URL}{RESET}")
    except Exception:
        print(f"  {RED}Cannot reach server at {BASE_URL}")
        print(f"  Start: python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8001{RESET}\n")
        sys.exit(1)

    dataset = load_dataset(sample)
    classify_items    = dataset.get("/v1/classify", [])
    chat_items        = dataset.get("/v1/chat/completions", [])
    completion_items  = dataset.get("/v1/completions", [])

    print(f"  {DIM}Loaded {sum(len(v) for v in dataset.values())} prompts from dataset{RESET}")
    print(f"  {DIM}classify={len(classify_items)}  chat={len(chat_items)}  "
          f"completions={len(completion_items)}{RESET}\n")

    run_ops()
    run_classify(classify_items)
    run_batch(classify_items)
    run_chat(chat_items)
    run_completions(completion_items)
    save_report()
    print_summary()
