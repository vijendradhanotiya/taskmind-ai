import os
import sys
from huggingface_hub import HfApi, create_repo, upload_folder

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID   = os.environ.get("HF_REPO_ID", "SatyamSinghal/taskmind-1.1b-chat-lora")
ADAPTER   = os.environ.get("ADAPTER_DIR", "out/taskmind_lora_peft")

if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN environment variable.")
    sys.exit(1)

if not os.path.isdir(ADAPTER):
    print(f"ERROR: Adapter directory not found: {ADAPTER}")
    sys.exit(1)

print(f"Creating/verifying repo: {REPO_ID}")
api = HfApi(token=HF_TOKEN)
create_repo(repo_id=REPO_ID, private=False, exist_ok=True, repo_type="model")

print(f"Uploading adapter from {ADAPTER} ...")
upload_folder(
    repo_id=REPO_ID,
    folder_path=ADAPTER,
    repo_type="model",
    token=HF_TOKEN,
    commit_message="Add TaskMind LoRA adapter — trained on WhatsApp task extraction dataset",
)

print(f"\nDone. Live at: https://huggingface.co/{REPO_ID}")
