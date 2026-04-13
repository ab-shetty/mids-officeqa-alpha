FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python deps — kept in a separate layer so source edits don't reinstall them
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Agent source
COPY src/ ./src/

# Cache directory mounted at runtime (corpus downloads here on first request)
RUN mkdir -p /data/corpus
ENV CORPUS_CACHE_DIR=/data/corpus
ENV PYTHONUNBUFFERED=1

EXPOSE 9019

ENTRYPOINT ["python", "-m", "src.server"]
CMD ["--host", "0.0.0.0", "--port", "9019"]
