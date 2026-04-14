"""
train_round3.py  --  Round 3 LoRA fine-tuning.

Strategy: load base model + merge R2 adapter into its weights, then train a
fresh R3 adapter on top. This means R3 builds directly on everything R2 learned.
The final adapter at out/taskmind_lora_r3 encodes R2 + R3 combined knowledge.

Training data (combined):
  - Original round 1 data         (131 examples)
  - Round 2 hard negatives        (180 examples)
  - Round 3 mined failures        (~248 examples from test CSV)
  - Round 3 generated targeted    (~700 new examples)

Key improvements over R2:
  - PROGRESS_NOTE training share: ~4 examples -> ~200 examples
  - TASK_ASSIGN "please complete" disambiguation: 45 -> 250+ examples
  - Hindi TASK_UPDATE patterns added
  - Improved system prompt with explicit critical distinctions
  - r=32, alpha=64, 10 epochs, lr=6e-5
"""

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
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

ORIG_TRAIN  = Path("data/task_mgmt_data/train/train.jsonl")
ORIG_VALID  = Path("data/task_mgmt_data/valid/valid.jsonl")
HARD_NEG    = Path("data/round2/train_hard.jsonl")
CORRECTIONS = Path("data/round3/train_corrections.jsonl")
GENERATED   = Path("data/round3/train_generated.jsonl")
R2_ADAPTER  = "out/taskmind_lora_r2"
OUT_DIR     = "out/taskmind_lora_r3"
MODEL_ID    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

LORA_R      = 32
LORA_ALPHA  = 64
TARGET_MODS = ["q_proj", "v_proj", "k_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
EPOCHS      = 10
LR          = 6e-5
BATCH       = 2
GRAD_ACC    = 4

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


def detect_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_jsonl(path):
    if not Path(path).exists():
        print(f"  Skipping missing file: {path}")
        return []
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    print(f"  Loaded {len(rows):>4} examples from {path}")
    return rows


def build_combined_dataset():
    print("\nLoading all training data:")
    orig        = load_jsonl(ORIG_TRAIN)
    hard_neg    = load_jsonl(HARD_NEG)
    corrections = load_jsonl(CORRECTIONS)
    generated   = load_jsonl(GENERATED)

    combined = orig + hard_neg + corrections + generated

    def get_intent(r):
        raw = r.get("completion", "") or r.get("text", "")
        for part in raw.split("### Response:\n"):
            try:
                return json.loads(part.strip()).get("intent", "?")
            except Exception:
                pass
        return "?"

    dist = Counter(get_intent(r) for r in combined)
    print(f"\nCombined train set: {len(combined)} examples")
    print("Intent distribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k:<22} {v}")
    print()

    return Dataset.from_list([{"text": r["text"]} for r in combined])


def build_valid_dataset():
    rows = load_jsonl(ORIG_VALID)
    return Dataset.from_list([{"text": r["text"]} for r in rows])


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


def evaluate(model, tokenizer, label):
    test_cases = [
        ("@Arpit please complete the dashboard by Friday",         "TASK_ASSIGN"),
        ("@Neha please complete the onboarding flow ASAP",         "TASK_ASSIGN"),
        ("@Karan please complete the admin panel this sprint",     "TASK_ASSIGN"),
        ("done bhai, merged the PR",                               "TASK_DONE"),
        ("billing module hotfix deployed to prod",                 "TASK_DONE"),
        ("notification service 60% done",                         "TASK_UPDATE"),
        ("auth service 65% ho gaya",                               "TASK_UPDATE"),
        ("can't start login page without product specs",           "TASK_UPDATE"),
        ("good morning team!",                                     "GENERAL_MESSAGE"),
        ("have a good weekend!",                                   "GENERAL_MESSAGE"),
        ("zoom link?",                                             "GENERAL_MESSAGE"),
        ("standup: worked on dashboard yesterday, 70% done",       "PROGRESS_NOTE"),
        ("EOD update: auth service done, starting billing module", "PROGRESS_NOTE"),
        ("load test: payment module stable at 800 rps",            "PROGRESS_NOTE"),
        ("A/B test result on search feature: variant B wins",      "PROGRESS_NOTE"),
        ("retro highlights: need better staging coverage",         "PROGRESS_NOTE"),
    ]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    correct = 0
    for msg, expected in test_cases:
        raw = run_inference(model, tokenizer, msg)
        try:
            predicted = json.loads(raw).get("intent", "?")
        except Exception:
            predicted = raw[:30]
        tick = "+" if predicted == expected else "-"
        print(f"  [{tick}] [{expected:<15}] {msg[:52]}")
        if predicted == expected:
            correct += 1

    print(f"\n  Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.0f}%")
    return correct


def main():
    device, dtype = detect_device()

    print("=" * 60)
    print("  TaskMind LoRA -- Round 3 Training")
    print(f"  Device : {device.upper()}")
    print(f"  R2 adapter merged into base before R3 training")
    print(f"  LoRA r : {LORA_R}  alpha : {LORA_ALPHA}")
    print(f"  Epochs : {EPOCHS}  LR : {LR}")
    print("=" * 60)

    print(f"\nLoading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map={"": device}
    )
    print("Base model loaded.")

    print(f"\nMerging R2 adapter ({R2_ADAPTER}) into base weights ...")
    model = PeftModel.from_pretrained(base, R2_ADAPTER)
    model = model.merge_and_unload()
    print("R2 merged. Training R3 adapter on top of merged model.")

    evaluate(model, tokenizer, "BASELINE (base + R2 merged, before R3)")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=TARGET_MODS,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = build_combined_dataset()
    valid_ds = build_valid_dataset()
    print(f"Valid: {len(valid_ds)} examples\n")

    training_args = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        fp16=False,
        bf16=False,
        dataset_text_field="text",
        max_seq_length=512,
        report_to="none",
    )

    print("Starting Round 3 LoRA fine-tuning ...\n")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nAdapter saved -> {OUT_DIR}/")

    final_correct = evaluate(model, tokenizer, "AFTER Round 3 Training")

    print("\n" + "=" * 60)
    print("  ROUND 3 COMPLETE")
    print("=" * 60)
    print(f"  Probe accuracy: {final_correct}/16")
    print(f"\n  Next steps:")
    print(f"  1. Update api/config.py -> ADAPTER_DIR = '{OUT_DIR}'")
    print(f"  2. Restart: python3 -m uvicorn api.main:app --port 8001")
    print(f"  3. Test:    python3 tests/test_api.py --all")
    print(f"  4. Upload:  HF_TOKEN=xxx HF_REPO_ID=SatyamSinghal/taskmind-1.1b-chat-lora-r3 \\")
    print(f"              ADAPTER_DIR={OUT_DIR} python3 scripts/upload_to_hf.py")


if __name__ == "__main__":
    main()
