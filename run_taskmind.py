import platform
import torch

if not hasattr(torch.backends.mps, "is_macos_or_newer"):
    def _is_macos_or_newer(major, minor=0):
        ver = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
        return ver >= (major, minor)
    torch.backends.mps.is_macos_or_newer = _is_macos_or_newer

import json
import os
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

TRAIN_PATH = Path("taskmind-data/train.jsonl")
VALID_PATH  = Path("taskmind-data/valid.jsonl")
OUT_DIR     = "out/taskmind_lora_peft"
MODEL_ID    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_MSG = (
    "You are TaskMind. Read the team WhatsApp message and return ONLY a JSON "
    "object with these exact fields: intent (TASK_ASSIGN / TASK_DONE / "
    "TASK_UPDATE / PROGRESS_NOTE / GENERAL_MESSAGE), assigneeName, project, "
    "title, deadline, priority, progressPercent. Use null for unknown fields."
)

TEST_MESSAGES = [
    "@Agrim fix the growstreams deck ASAP NO Delay",
    "done bhai, merged the PR",
    "login page 60% ho gaya",
    "getting 500 error on registration",
    "Sure sir ready for it",
]


def detect_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_model_and_tokenizer(device, dtype):
    print(f"\nLoading {MODEL_ID} on {device} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        mdl = mdl.to(device)
    print("Model loaded.\n")
    return mdl, tok


def build_prompt(message):
    return (
        "### System:\n" + SYSTEM_MSG
        + "\n\n### Message:\n" + message
        + "\n\n### Response:\n"
    )


def run_inference(mdl, tok, message, max_new_tokens=150):
    prompt = build_prompt(message)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    mdl.eval()
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def test_model(mdl, tok, label):
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)
    results = []
    for msg in TEST_MESSAGES:
        result = run_inference(mdl, tok, msg)
        results.append(result)
        print(f"Input : {msg}")
        print(f"Output: {result[:250]}")
        print()
    return results


def load_dataset_from_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_row(ex):
    output_str = json.dumps(ex["output"], ensure_ascii=False)
    return {
        "text": (
            "### System:\n" + SYSTEM_MSG
            + "\n\n### Message:\n" + ex["input"]
            + "\n\n### Response:\n" + output_str
        )
    }


def main():
    device, dtype = detect_device()
    print(f"Running on: {device.upper()}")

    model, tokenizer = load_model_and_tokenizer(device, dtype)

    before_results = test_model(model, tokenizer, "TEST — BEFORE TRAINING (base model)")

    print("Preparing dataset ...")
    train_rows = [format_row(r) for r in load_dataset_from_jsonl(TRAIN_PATH)]
    valid_rows  = [format_row(r) for r in load_dataset_from_jsonl(VALID_PATH)]
    train_dataset = Dataset.from_list(train_rows)
    valid_dataset  = Dataset.from_list(valid_rows)
    print(f"Train: {len(train_dataset)}  Valid: {len(valid_dataset)}\n")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        learning_rate=2e-4,
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

    print("\nStarting LoRA fine-tuning ...\n")
    trainer.train()
    print("\nTraining complete.")

    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Adapter saved -> {OUT_DIR}/\n")

    after_results = test_model(model, tokenizer, "TEST — AFTER TRAINING (TaskMind LoRA)")

    print("=" * 60)
    print("  BEFORE vs AFTER COMPARISON")
    print("=" * 60)
    for msg, before, after in zip(TEST_MESSAGES, before_results, after_results):
        print(f"Msg    : {msg}")
        print(f"Before : {before[:130]}")
        print(f"After  : {after[:130]}")
        print()

    print("All done. Run the API server with:")
    print("  python3 taskmind_fastapi.py\n")


if __name__ == "__main__":
    main()
