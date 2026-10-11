FROM python:3.11-slim

# System deps (curl for uv; build tools keep some scientific wheels happy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock requirements.txt ./

# Copy the source code BEFORE syncing (uv sync installs the project)
COPY . .

# Sync project deps, then add tooling used by each layer
RUN uv sync --frozen && \
    uv pip install kaggle && \
    # CPU PyTorch for Autoencoder track (keeps image smaller than CUDA wheels)
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Runtime directories (overridden by compose volume mounts in practice)
RUN mkdir -p data/raw data/processed artifacts/experiments artifacts/segmentation

RUN chmod +x scripts/run_data_layer.sh scripts/run_segmentation_layer.sh

# Headless matplotlib inside containers
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

# Default remains the data layer for backward compatibility with README examples
CMD ["./scripts/run_data_layer.sh"]
