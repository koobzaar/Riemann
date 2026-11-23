FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RIEMANN_DB_PATH=/data/riemann.sqlite \
    RIEMANN_GRAPH_PATH=/app/data/relation_graph.json

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data ./data
COPY tests/fixtures ./tests/fixtures

RUN pip install --no-cache-dir .

RUN mkdir -p /data

EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=10 \
    CMD python -c "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); assert r.status == 200"

CMD ["uvicorn", "riemann.app:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
