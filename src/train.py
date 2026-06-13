"""Fine-tune DistilBERT on IMDB sentiment classification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import wandb
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ID2LABEL_PATH = Path("id2label.json")


def load_id2label(path: Path = ID2LABEL_PATH) -> dict[int, str]:
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {int(key): value for key, value in mapping.items()}


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_processed_splits(data_dir: Path) -> tuple[Dataset, Dataset]:
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized.remove_columns(["text"])


def build_compute_metrics():
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    return compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model on IMDB.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file (for example configs/train_v1.yaml).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the best checkpoint to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Target Hugging Face repository id, e.g. username/imdb-distilbert-v2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    id2label = load_id2label()
    label2id = {label: index for index, label in id2label.items()}

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset, eval_dataset = load_processed_splits(Path(config["data_dir"]))
    train_dataset = tokenize_dataset(train_dataset, tokenizer, config["max_length"])
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, config["max_length"])

    wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config={
            "model": model_name,
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "version": config["version"],
            "platform": os.environ.get("PLATFORM", "local"),
            "max_length": config["max_length"],
        },
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb",
        run_name=config["run_name"],
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()
    metrics = trainer.evaluate()
    wandb.log({f"eval_{key}": value for key, value in metrics.items()})

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required when --push-to-hub is set.")
        trainer.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{args.hub_model_id}"

    wandb.finish()


if __name__ == "__main__":
    main()
