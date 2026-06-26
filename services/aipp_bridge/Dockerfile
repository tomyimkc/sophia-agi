FROM python:3.12-slim

WORKDIR /app
COPY requirements-rag.txt requirements-rag.txt
RUN pip install --no-cache-dir -r requirements-rag.txt

COPY agent agent
COPY data data
COPY docs docs
COPY benchmark/reference benchmark/reference
COPY training/examples training/examples
COPY rag/index rag/index
COPY services services

ENV PYTHONPATH=/app
ENV SOPHIA_RAG_INDEX_DIR=/app/rag/index
ENV SOPHIA_RAG_BACKEND=gemini
ENV PORT=8081
# SOPHIA_AIPP_TOKEN must be supplied at run time — the bridge fails closed without it.

EXPOSE 8081
CMD ["uvicorn", "services.aipp_bridge.main:app", "--host", "0.0.0.0", "--port", "8081"]
