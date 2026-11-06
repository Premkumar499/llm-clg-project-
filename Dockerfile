FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight CPU-only PyTorch first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./
# Install requirements excluding bitsandbytes for CPU deployment
RUN sed -i '/bitsandbytes/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY model/ ./model/
COPY prem_finetune_dataset.jsonl ./
COPY rag_pdf/ ./rag_pdf/

ENV PYTHONPATH=/app/backend
WORKDIR /app/backend

# Pre-build RAG FAISS index during build
RUN python rag.py

EXPOSE 8000 10000

# Render provides $PORT dynamically
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
