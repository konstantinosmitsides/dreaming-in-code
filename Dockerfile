# Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV TZ=Europe/London \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH="/workspace/src:$PYTHONPATH" \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml requirements.txt README.md /workspace/
COPY src /workspace/src
COPY conf /workspace/conf
COPY experiments /workspace/experiments

# Install dependencies
# Using --system to install into the container's global python environment
RUN uv pip install --system "jax[cuda12]"
RUN uv pip install --system -e .[dev,evaluation]

# Pre-generate Craftax textures to save runtime overhead
ENV JAX_PLATFORMS=cpu
RUN python3 -c "from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG; print('Textures generated successfully')" && \
    python3 -c "from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_IMG; print('Classic textures generated')"

# Default entry point
ENTRYPOINT ["python3", "experiments/training/run_dicode.py"]
