FROM python:3.10-slim

# Install system dependencies for MDAnalysis/MDTraj/Kaleido
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for caching
COPY pyproject.toml ./
# Create a dummy src/ to allow pip install . to succeed
RUN mkdir -p src/md_platform && touch src/md_platform/__init__.py
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ /app/src/

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "md_platform.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
