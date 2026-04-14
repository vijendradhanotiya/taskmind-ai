"""
mine_failures.py  --  Read the latest test CSV report, extract every wrong
classification, and write them as corrective training examples.

Output: data/round3/train_corrections.jsonl

Usage:
    python3 scripts/mine_failures.py                          # latest report
    python3 scripts/mine_failures.py tests/reports/foo.csv   # specific file
"""

import csv
import json
import sys
import glob
from pathlib import Path
from collections import Counter, defaultdict

SYSTEM = (
    "You are TaskMind. Read the team WhatsApp message and return ONLY a JSON "
    "object with these exact fields: intent (TASK_ASSIGN / TASK_DONE / "
    "TASK_UPDATE / PROGRESS_NOTE / GENERAL_MESSAGE), assigneeName, project, "
    "title, deadline, priority, progressPercent. Use null for unknown fields.\n\n"
    "Critical distinctions:\n"
    "  TASK_ASSIGN    = giving a task TO someone (@name + imperative verb, future action)\n"
    "  TASK_DONE      = work already finished (merged, shipped, deployed, completed, tested)\n"
    "  TASK_UPDATE    = partial progress or blocker (X% done/ho gaya, stuck, waiting, can't start)\n"
    "  PROGRESS_NOTE  = team broadcast: standup, EOD update, weekly/sprint report, load test,\n"
    "                   A/B test result, postmortem, retro highlights, release schedule\n"
    "  GENERAL_MESSAGE = social, greetings, reactions, logistics questions, acknowledgements"
)


def make_entry(message, intent):
    prompt = "### System:\n" + SYSTEM + "\n\n### Message:\n" + message + "\n\n### Response:\n"
    completion = json.dumps({
        "intent": intent, "assigneeName": None, "project": None,
        "title": None, "deadline": None, "priority": None, "progressPercent": None
    })
    return {"prompt": prompt, "completion": completion, "text": prompt + completion}


def main():
    reports = sorted(glob.glob("tests/reports/*.csv"))
    if not reports:
        print("No reports found in tests/reports/")
        sys.exit(1)

    csv_path = sys.argv[1] if len(sys.argv) > 1 else reports[-1]
    print(f"Mining failures from: {csv_path}")

    rows = list(csv.DictReader(Path(csv_path).read_text().splitlines()))
    classify = [r for r in rows if r["endpoint"] == "/v1/classify"]
    correct = [r for r in classify if r["intent_correct"] == "True"]
    wrong   = [r for r in classify if r["intent_correct"] == "False"]

    total = len(classify)
    print(f"Accuracy : {len(correct)}/{total} = {len(correct)/total*100:.1f}%")
    print(f"Failures : {len(wrong)} examples to mine as training data")

    confusion = defaultdict(Counter)
    for r in wrong:
        confusion[r["expected_intent"]][r["actual_intent"]] += 1

    print("\nConfusion matrix (expected -> wrongly predicted):")
    for exp in sorted(confusion):
        n_wrong = sum(confusion[exp].values())
        n_total = sum(1 for r in classify if r["expected_intent"] == exp)
        acc = (n_total - n_wrong) / n_total * 100
        print(f"  {exp} ({n_total} total, {acc:.0f}% correct, {n_wrong} wrong):")
        for pred, cnt in confusion[exp].most_common():
            print(f"    -> {pred:<22} {cnt}x")

    seen, entries = set(), []
    for r in wrong:
        msg = r["input"].strip()
        exp = r["expected_intent"].strip()
        if msg and exp and msg not in seen:
            seen.add(msg)
            entries.append(make_entry(msg, exp))

    out = Path("data/round3/train_corrections.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    dist = Counter(json.loads(e["completion"])["intent"] for e in entries)
    print(f"\nWrote {len(entries)} corrective examples -> {out}")
    print("Intent breakdown:")
    for k, v in sorted(dist.items()):
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()

