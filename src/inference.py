"""Run sentiment inference with a Hugging Face model. """

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "distilbert-base-uncased"
ID2LABEL_PATH = Path("id2label.json")


def load_id2label(path: Path = ID2LABEL_PATH) -> dict[int, str]:
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {int(key): value for key, value in mapping.items()}


def authenticate_hf() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)


def predict(text: str, model_name: str | None = None) -> dict:
    model_name = model_name or os.environ.get("HF_MODEL_NAME") or DEFAULT_MODEL
    id2label = load_id2label()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())

    return {
        "text": text,
        "label": id2label.get(predicted_id, str(predicted_id)),
        "label_id": predicted_id,
        "confidence": float(probabilities[predicted_id].item()),
        "model": model_name,
    }


def main() -> None:
    input_text = os.environ.get("INPUT_TEXT", "").strip()
    if not input_text:
        print("ERROR: INPUT_TEXT environment variable is required.", file=sys.stderr)
        sys.exit(1)

    authenticate_hf()
    result = predict(input_text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
