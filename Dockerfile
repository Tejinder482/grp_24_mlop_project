FROM python:3.11-slim

ARG HF_MODEL_NAME=distilbert-base-uncased
ENV HF_MODEL_NAME=${HF_MODEL_NAME}
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY src/inference.py src/inference.py
COPY id2label.json id2label.json

CMD ["python", "src/inference.py"]
