FROM python:3.12-slim

WORKDIR /app

# Pure-stdlib exporter: no third-party deps to install.
COPY agent agent
COPY services services
COPY config config

ENV PYTHONPATH=/app
ENV SOPHIA_CLUSTER_SOURCE=mock
ENV PORT=9881

EXPOSE 9881
CMD ["python3", "-m", "services.cluster_exporter.main", "--host", "0.0.0.0", "--port", "9881"]
