import platform
import torch

if not hasattr(torch.backends.mps, "is_macos_or_newer"):
    def _is_macos_or_newer(major, minor=0):
        ver = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
        return ver >= (major, minor)
    torch.backends.mps.is_macos_or_newer = _is_macos_or_newer

import json
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from api.config import settings

logger = logging.getLogger(__name__)

SYSTEM_MSG = (
    "You are TaskMind. Read the team WhatsApp message and return ONLY a JSON "
    "object with these exact fields: intent (TASK_ASSIGN / TASK_DONE / "
    "TASK_UPDATE / PROGRESS_NOTE / GENERAL_MESSAGE), assigneeName, project, "
    "title, deadline, priority, progressPercent. Use null for unknown fields."
)


def _detect_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


class TaskMindModel:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None

    def load(self):
        if self._model is not None:
            return

        self._device, self._dtype = _detect_device()
        logger.info("Loading base model %s on %s", settings.BASE_MODEL, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            settings.BASE_MODEL,
            dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
        )

        if os.path.isdir(settings.ADAPTER_DIR):
            self._model = PeftModel.from_pretrained(base, settings.ADAPTER_DIR)
            logger.info("Loaded LoRA adapter from %s", settings.ADAPTER_DIR)
        else:
            self._model = base
            logger.warning("No adapter at %s — using base model only", settings.ADAPTER_DIR)

        if self._device != "cuda":
            self._model = self._model.to(self._device)

        self._model.eval()
        logger.info("Model ready on %s", self._device)

    @property
    def is_loaded(self):
        return self._model is not None

    @property
    def device(self):
        return self._device or "not loaded"

    def _build_prompt(self, message: str) -> str:
        return (
            "### System:\n" + SYSTEM_MSG
            + "\n\n### Message:\n" + message
            + "\n\n### Response:\n"
        )

    def _parse_json(self, raw: str):
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end]), True
        except (json.JSONDecodeError, ValueError):
            pass
        return None, False

    def classify(self, message: str) -> tuple:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        prompt = self._build_prompt(message)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        parsed, success = self._parse_json(raw)
        return raw, parsed, success


model_instance = TaskMindModel()
