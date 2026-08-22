FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies for MDAnalysis/MDTraj/Kaleido
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

EXPOSE 8000

# PaaS-friendly: bind $PORT when set (Fly/Render), default 8000 for compose.
CMD sh -c "uvicorn md_platform.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"
