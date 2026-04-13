FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY out/taskmind_lora_peft/ ./out/taskmind_lora_peft/

ENV BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
ENV ADAPTER_DIR=out/taskmind_lora_peft
ENV HOST=0.0.0.0
ENV PORT=8001
ENV LOG_LEVEL=INFO

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
