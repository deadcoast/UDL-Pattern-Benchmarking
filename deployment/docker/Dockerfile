# Multi-stage Docker build for UDL Rating Framework API
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV for fast Python package management
RUN pip install uv

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY deployment/api/requirements.txt .
COPY pyproject.toml .
COPY requirements.txt ./requirements-main.txt

# Install Python dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -r requirements.txt
RUN uv pip install -r requirements-main.txt

# Development stage
FROM base as development

# Install development dependencies
RUN uv pip install pytest pytest-cov black flake8 mypy

# Copy source code
COPY . .

# Install package in development mode
RUN uv pip install -e .

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY udl_rating_framework/ ./udl_rating_framework/
COPY deployment/api/ ./deployment/api/
COPY pyproject.toml .
COPY README.md .

# Install package
RUN uv pip install .

# Copy models directory if it exists
COPY models/ ./models/ 2>/dev/null || true

# Create directories for runtime
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Minimal production stage for smaller image
FROM python:3.11-alpine as minimal

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    curl \
    libffi-dev

# Install UV
RUN pip install uv

# Create app user
RUN addgroup -S appuser && adduser -S appuser -G appuser

WORKDIR /app

# Copy requirements and install dependencies
COPY deployment/api/requirements.txt .
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -r requirements.txt

# Copy application code
COPY udl_rating_framework/ ./udl_rating_framework/
COPY deployment/api/ ./deployment/api/
COPY pyproject.toml .

# Install package
RUN uv pip install .

RUN chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]