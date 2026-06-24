# OPTIONAL — not required. This Space runs as a Gradio SDK Space (see README front
# matter: `sdk: gradio`), and the gate is imported in-process from sophia_mcp.tools_impl
# with NO `mcp` package needed. Use this Dockerfile only if you later want to run the
# full FastMCP server alongside the UI.
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r huggingface/space-v2/requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860
CMD ["python", "huggingface/space-v2/app.py"]
