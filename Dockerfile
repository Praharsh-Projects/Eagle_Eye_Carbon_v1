FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app /app/app
COPY config /app/config
COPY demo_data /app/demo_data
COPY src /app/src
COPY README.md /app/README.md
COPY run_api.sh /app/run_api.sh

RUN mkdir -p /app/data/processed /app/data/chroma && chmod +x /app/run_api.sh

EXPOSE 8000

CMD ["sh", "-lc", "/app/run_api.sh"]
