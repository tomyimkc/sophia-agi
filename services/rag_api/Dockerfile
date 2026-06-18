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
ENV PORT=8080

EXPOSE 8080
CMD ["uvicorn", "services.rag_api.main:app", "--host", "0.0.0.0", "--port", "8080"]