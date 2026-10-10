FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock requirements.txt ./

# Copy the source code BEFORE syncing, as uv sync installs the project itself (needs README and src/)
COPY . .

# Sync dependencies using uv (creates a virtual environment automatically in .venv)
RUN uv sync --frozen && \
    uv pip install kaggle

# Set up the data directories
RUN mkdir -p data/raw data/processed

# Make the run script executable
RUN chmod +x scripts/run_data_layer.sh

# Default command
CMD ["./scripts/run_data_layer.sh"]
