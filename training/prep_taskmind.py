import json
from pathlib import Path


TRAIN_IN  = Path("taskmind-data/train.jsonl")
VALID_IN  = Path("taskmind-data/valid.jsonl")
TRAIN_OUT = Path("data/task_mgmt_data/train/train.jsonl")
VALID_OUT = Path("data/task_mgmt_data/valid/valid.jsonl")

SYSTEM_MSG = (
    "You are TaskMind. Read the team WhatsApp message and return ONLY a JSON "
    "object with these exact fields: intent (TASK_ASSIGN / TASK_DONE / "
    "TASK_UPDATE / PROGRESS_NOTE / GENERAL_MESSAGE), assigneeName, project, "
    "title, deadline, priority, progressPercent. Use null for unknown fields."
)

TEMPLATE = "### System:\n{sys}\n\n### Message:\n{msg}\n\n### Response:\n{resp}"


def build_text(row):
    output_str = json.dumps(row["output"], ensure_ascii=False)
    return {
        "prompt": "### System:\n" + SYSTEM_MSG + "\n\n### Message:\n" + row["input"],
        "completion": "\n\n### Response:\n" + output_str,
        "text": TEMPLATE.format(sys=SYSTEM_MSG, msg=row["input"], resp=output_str),
    }


def convert(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(build_text(r), ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {dst}")


def main():
    convert(TRAIN_IN, TRAIN_OUT)
    convert(VALID_IN, VALID_OUT)
    print("Data ready for MLX and PEFT training.")


if __name__ == "__main__":
    main()
