"""Prepare the Stanford IMDB dataset for fine-tuning  . """

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ID2LABEL = {"0": "neg", "1": "pos"}
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Normalize review text for DistilBERT tokenization."""
    if not isinstance(text, str):
        return ""

    text = HTML_TAG_PATTERN.sub(" ", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_raw_dataset():
    """Load IMDB train and test splits from Hugging Face."""
    dataset = load_dataset("stanfordnlp/imdb")
    return dataset["train"].to_pandas(), dataset["test"].to_pandas()


def prepare_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean text, drop invalid rows, and remove duplicate reviews."""
    prepared = frame.copy()
    prepared["text"] = prepared["text"].map(clean_text)
    prepared = prepared[prepared["text"].str.len() > 0]
    prepared = prepared.dropna(subset=["text", "label"])
    prepared["label"] = prepared["label"].astype(int)
    prepared = prepared.drop_duplicates(subset=["text"], keep="first")
    prepared = prepared.reset_index(drop=True)
    return prepared[["text", "label"]]


def save_label_mapping(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(ID2LABEL, handle, indent=2)
        handle.write("\n")


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)


def print_dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("Dataset: stanfordnlp/imdb")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print("Train label distribution:")
    print(train_df["label"].value_counts().sort_index().rename(ID2LABEL).to_string())
    print("Test label distribution:")
    print(test_df["label"].value_counts().sort_index().rename(ID2LABEL).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and export the IMDB dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed parquet files.",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=Path("id2label.json"),
        help="Path to the id2label mapping file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_raw, test_raw = load_raw_dataset()
    train_df = prepare_split(train_raw)
    test_df = prepare_split(test_raw)

    save_label_mapping(args.label_file)
    save_processed_data(train_df, test_df, args.output_dir)
    print_dataset_summary(train_df, test_df)
    print(f"Saved processed data to {args.output_dir}")
    print(f"Saved label mapping to {args.label_file}")


if __name__ == "__main__":
    main()
