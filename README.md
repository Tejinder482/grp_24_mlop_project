# MLOps Group Project — IMDB Sentiment Classification

End-to-end MLOps pipeline for binary movie-review sentiment classification using the [Stanford IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) and [DistilBERT](https://huggingface.co/distilbert-base-uncased).

## Project overview

| Component | Choice | Why |
|-----------|--------|-----|
| Dataset | `stanfordnlp/imdb` | Classic binary sentiment task, 25k train / 25k test, fits Kaggle GPU limits |
| Model | `distilbert-base-uncased` | ~66M parameters, fast to fine-tune, strong baseline for text classification |
| Training | Kaggle Notebooks (GPU) | Free GPU for fine-tuning |
| Tracking | Weights & Biases | Experiment comparison across v1 and v2 |
| Inference | Docker + GitHub Actions | Reproducible deployment and CI |

### Experiment versions

| Version | Epochs | Batch size | Learning rate |
|---------|--------|------------|---------------|
| v1 | 3 | 16 | 3e-5 |
| v2 | 5 | 32 | 2e-5 |

## Repository structure

```text
mlops-group-project/
├── src/
│   ├── preprocess.py      # Clean IMDB data and write id2label.json
│   ├── train.py           # Fine-tune DistilBERT with W&B logging
│   ├── push_to_hub.py     # Push best checkpoint to Hugging Face Hub
│   └── inference.py       # Run predictions from CLI / Docker / GitHub Actions
├── configs/
│   ├── train_v1.yaml
│   └── train_v2.yaml
├── notebooks/
│   ├── kaggle_train_v1.ipynb
│   └── kaggle_train_v2.ipynb
├── data/                  # Generated locally (not committed)
├── .github/workflows/
│   ├── ci.yml
│   └── inference.yml
├── Dockerfile
├── id2label.json
├── requirements.txt
└── requirements-inference.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Preprocess data

```bash
python src/preprocess.py
```

This script:

- loads IMDB from Hugging Face
- strips HTML tags, lowercases text, removes empty/duplicate reviews
- saves `data/processed/train.parquet` and `data/processed/test.parquet`
- writes `id2label.json` (`0 -> neg`, `1 -> pos`)

Only `id2label.json` is committed. Processed parquet files stay local.

### 2. Train on Kaggle

1. Push the latest `develop` branch to GitHub (notebooks clone `Tejinder482/MLOPS_group_project`).
2. On [Kaggle](https://www.kaggle.com/code), click **New Notebook → Import Notebook** and upload:
   - `notebooks/kaggle_train_v1.ipynb` → save as **imdb-train-v1**
   - `notebooks/kaggle_train_v2.ipynb` → save as **imdb-train-v2**
3. In each notebook: **Settings → Accelerator → GPU T4 x2**.
4. **Add-ons → Secrets** (account-level):
   - `WANDB_API_KEY` — from https://wandb.ai/authorize
   - `HF_TOKEN` — from https://huggingface.co/settings/tokens (read + write)
5. Run all cells in **v1** first (~30–45 min), then run **v2**.
6. Open the W&B project and compare `run-v1` vs `run-v2` (accuracy, F1, loss).
7. In the winning notebook, run the optional **push to Hugging Face** cell (creates `tejinder482/imdb-distilbert-best`).
8. Set both Kaggle notebooks to **Public** (Share) and paste URLs below.

### 3. Local training (optional smoke test)

```bash
python src/preprocess.py
python src/train.py --config configs/train_v1.yaml
```

### 4. Inference

```bash
export HF_TOKEN=your_token
export HF_MODEL_NAME=tejinder482/imdb-distilbert-best
export INPUT_TEXT="This movie was amazing!"
python src/inference.py
```

### 5. Docker

```bash
docker build --build-arg HF_MODEL_NAME=tejinder482/imdb-distilbert-best -t mlops-a3-inference:latest .
docker run --rm -e HF_TOKEN=<token> -e INPUT_TEXT="Great film!" mlops-a3-inference:latest
docker tag mlops-a3-inference:latest tejinder482/mlops-a3-inference:latest
docker push tejinder482/mlops-a3-inference:latest
```

## GitHub configuration

### Branches

- `main` — protected, requires PR review
- `develop` — CI runs on push

### Secrets (Settings → Secrets and variables → Actions)

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | Pull private models / authenticate inference |
| `WANDB_API_KEY` | Optional for Actions if you log inference |

### Variables

| Variable | Example | Purpose |
|----------|---------|---------|
| `HF_MODEL_NAME` | `tejinder482/imdb-distilbert-best` | Model used by inference workflow |

### Workflows

- **CI** (`ci.yml`) — runs `flake8` on push to `develop` and PRs to `main`
- **Inference** (`inference.yml`) — manual dispatch with review text input

## Submission links

| Item | Link |
|------|------|
| GitHub Repository | https://github.com/Tejinder482/MLOPS_group_project |
| Kaggle Notebook v1 | https://www.kaggle.com/code/tejinder482/mlops-grp24-v1 |
| Kaggle Notebook v2 | https://www.kaggle.com/code/tejinder482/mlops-grp24-v2 |
| Hugging Face Model | https://huggingface.co/tejinder482/imdb-distilbert-best |
| Docker Image | https://hub.docker.com/r/tejinder482/mlops-a3-inference |
| W&B Dashboard | https://wandb.ai/tejindersingh2202-indian-institute-of-technology/mlops-assignment3 |
| W&B Report | https://wandb.ai/tejindersingh2202-indian-institute-of-technology/mlops-assignment3/reports/MLOPS-Group-24---VmlldzoxNzE5OTYzNg |

## Model selection rationale (for report)

We chose `distilbert-base-uncased` because it is a compact distilled version of BERT (~66M parameters, under the 200 MB assignment guideline). It is pretrained on English text and widely used for sentiment classification. Compared with larger models, it trains quickly on Kaggle's free GPU quota while still reaching strong accuracy on IMDB. The uncased tokenizer matches our lowercase preprocessing, and the Hugging Face model card documents proven performance on classification benchmarks.
