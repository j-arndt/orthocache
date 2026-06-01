# syntax=docker/dockerfile:1

# ============================================================================
# OrthoCache — Reproducible Validation Build
# ============================================================================
# Multi-stage Dockerfile for building and testing the OrthoCache library.
#
# Usage:
#   docker build -t orthocache:latest .
#   docker run --rm orthocache:latest           # runs pytest (default)
#   docker run --rm orthocache:latest python -c "import orthocache; print(orthocache.__version__)"
#
# Stages:
#   1. deps    — install Python dependencies into a venv
#   2. test    — copy source, install package, run pytest
#   3. runtime — minimal image for library use
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/orthocache

# Create venv and install deps first (maximises layer caching)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
# Install build deps + runtime deps (CPU-only JAX for validation)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir "jax[cpu]>=0.4.25" "numpy>=1.24.0" "pytest>=8.0.0"

# ---------------------------------------------------------------------------
# Stage 2: Test Runner
# ---------------------------------------------------------------------------
FROM deps AS test

WORKDIR /opt/orthocache

# Copy source and tests
COPY src/ src/
COPY tests/ tests/
COPY pyproject.toml README.md ./

# Install orthocache in editable mode
RUN pip install --no-cache-dir -e ".[dev]"

# Verify import works
RUN python -c "from orthocache import fwht_512, compute_spectral_bands, compute_spectral_decay_ratio; print('OrthoCache imports OK')"

# Default: run full test suite
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# ---------------------------------------------------------------------------
# Stage 3: Runtime (minimal, no test deps)
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /opt/orthocache

# Copy venv from deps stage
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only source code (no tests)
COPY src/ src/
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir -e .

# Verify
RUN python -c "import orthocache; print(f'OrthoCache v{orthocache.__version__} ready')"

ENTRYPOINT ["python"]
