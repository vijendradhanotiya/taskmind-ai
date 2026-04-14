import platform
import torch

if not hasattr(torch.backends.mps, "is_macos_or_newer"):
    def _is_macos_or_newer(major, minor=0):
        ver = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
        return ver >= (major, minor)
    torch.backends.mps.is_macos_or_newer = _is_macos_or_newer

import json
from pathlib import Path
from collections import Counter
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

ORIG_TRAIN  = Path("data/task_mgmt_data/train/train.jsonl")
ORIG_VALID  = Path("data/task_mgmt_data/valid/valid.jsonl")
HARD_NEG    = Path("data/round2/train_hard.jsonl")
OUT_DIR     = "out/taskmind_lora_r2"
MODEL_ID    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── Round 2 improvements over Round 1 ──────────────────────────────────────
# Round 1: r=16, alpha=32, q_proj+v_proj only, 5 epochs, lr=2e-4, 131 examples
# Round 2: r=32, alpha=64, all attention+FFN proj, 8 epochs, lr=8e-5, ~311 examples
# Key: lower lr for careful disambiguation, wider module coverage, hard negatives
LORA_R       = 32
LORA_ALPHA   = 64
TARGET_MODS  = ["q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
EPOCHS       = 8
LR           = 8e-5
BATCH        = 2
GRAD_ACC     = 4


def detect_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def build_combined_dataset():
    orig = load_jsonl(ORIG_TRAIN)
    hard = load_jsonl(HARD_NEG)
    combined = orig + hard

    counts = Counter(
        json.loads(r.get("completion", "{}").replace("\n\n### Response:\n", "") or "{}")
        .get("intent", "?")
        for r in combined
    )
    print(f"\nCombined train set: {len(combined)} examples")
    print("Intent distribution:")
    for k, v in sorted(counts.items()):
        print(f"  {k:<20} {v}")
    print()
    return Dataset.from_list([{"text": r["text"]} for r in combined])


def build_test_messages():
    return [
        ("@Arpit please complete the dashboard by Friday",       "TASK_ASSIGN"),
        ("@Neha please complete the onboarding flow this sprint","TASK_ASSIGN"),
        ("done bhai, merged the PR",                             "TASK_DONE"),
        ("notification service 60% done",                       "TASK_UPDATE"),
        ("DB migration 75% done",                               "TASK_UPDATE"),
        ("good morning team!",                                   "GENERAL_MESSAGE"),
        ("have a good weekend!",                                 "GENERAL_MESSAGE"),
        ("bhai kaafi solid kaam kiya tune",                      "GENERAL_MESSAGE"),
        ("sprint status: 60% tasks done, on track for Friday",  "PROGRESS_NOTE"),
        ("load test: stable at 800 rps",                        "PROGRESS_NOTE"),
    ]


SYSTEM = (
    "You are TaskMind. Read the team WhatsApp message and return ONLY a JSON "
    "object with these exact fields: intent (TASK_ASSIGN / TASK_DONE / "
    "TASK_UPDATE / PROGRESS_NOTE / GENERAL_MESSAGE), assigneeName, project, "
    "title, deadline, priority, progressPercent. Use null for unknown fields."
)


def run_inference(model, tokenizer, message, max_new_tokens=120):
    prompt = f"### System:\n{SYSTEM}\n\n### Message:\n{message}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def eval_accuracy(model, tokenizer, label):
    tests = build_test_messages()
    correct = 0
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for msg, expected in tests:
        raw = run_inference(model, tokenizer, msg)
        try:
            intent = json.loads(raw.split("\n")[0]).get("intent", "?")
        except Exception:
            intent = "PARSE_ERROR"
        match = "✓" if intent == expected else "✗"
        print(f"  {match} [{expected:<16}] {msg[:50]}")
        if intent == expected:
            correct += 1
    print(f"\n  Accuracy: {correct}/{len(tests)} = {correct/len(tests)*100:.0f}%")
    return correct, len(tests)


def main():
    device, dtype = detect_device()
    print(f"\n{'='*60}")
    print(f"  TaskMind LoRA — Round 2 Training")
    print(f"  Device : {device.upper()}")
    print(f"  LoRA r : {LORA_R}  alpha : {LORA_ALPHA}")
    print(f"  Modules: {', '.join(TARGET_MODS)}")
    print(f"  Epochs : {EPOCHS}  LR : {LR}")
    print(f"{'='*60}")

    print(f"\nLoading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    print("Base model loaded.\n")

    train_dataset = build_combined_dataset()
    valid_rows = load_jsonl(ORIG_VALID)
    valid_dataset = Dataset.from_list([{"text": r["text"]} for r in valid_rows])
    print(f"Valid: {len(valid_dataset)} examples\n")

    before_correct, total = eval_accuracy(model, tokenizer, "BASELINE (before Round 2)")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODS,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        warmup_steps=20,
        learning_rate=LR,
        fp16=(device == "cuda"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataloader_pin_memory=False,
        dataset_text_field="text",
        max_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    print("\nStarting Round 2 LoRA fine-tuning ...\n")
    trainer.train()
    print("\nTraining complete.")

    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nAdapter saved → {OUT_DIR}/")

    after_correct, _ = eval_accuracy(model, tokenizer, "AFTER Round 2 Training")

    print(f"\n{'='*60}")
    print(f"  ROUND 2 RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline accuracy : {before_correct}/{total} = {before_correct/total*100:.0f}%")
    print(f"  After Round 2     : {after_correct}/{total}  = {after_correct/total*100:.0f}%")
    gain = after_correct - before_correct
    print(f"  Improvement       : {'+' if gain >= 0 else ''}{gain} examples correct")
    print(f"\n  To use this adapter in the API:")
    print(f"  Update api/config.py → ADAPTER_PATH = '{OUT_DIR}'")
    print(f"  Then restart: python3 -m uvicorn api.main:app --port 8001\n")


if __name__ == "__main__":
    main()
