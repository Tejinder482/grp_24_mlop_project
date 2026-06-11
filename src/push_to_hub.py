"""Push the best local checkpoint to Hugging Face without retraining.  """

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_checkpoint_dir(output_dir: Path) -> Path:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", maxsplit=1)[-1]),
    )
    if checkpoints:
        return checkpoints[-1]
    if output_dir.exists():
        return output_dir
    raise FileNotFoundError(f"No checkpoints found under {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push a trained checkpoint to Hugging Face Hub.")
    parser.add_argument("--config", type=Path, required=True, help="Training config used for the run.")
    parser.add_argument("--hub-model-id", type=str, required=True, help="Target repo, e.g. user/imdb-distilbert-best.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_dir = resolve_checkpoint_dir(Path(config["output_dir"]))

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model.push_to_hub(args.hub_model_id)
    tokenizer.push_to_hub(args.hub_model_id)

    wandb.init(
        project=config["wandb_project"],
        name=f"push-{config['version']}",
        job_type="push-to-hub",
    )
    wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{args.hub_model_id}"
    wandb.finish()
    print(f"Pushed {checkpoint_dir} to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
