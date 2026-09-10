# F-Intel DataServer - Dockerfile
# Build:  docker build -t fintel .
# Run:    docker run --env-file .env -p 8082:8082 fintel
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libhdf5-dev libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data logs

EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8082/health || exit 1

CMD ["python", "DataServer.py"]
