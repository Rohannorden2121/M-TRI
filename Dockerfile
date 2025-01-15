# Multi-stage Docker build for M-TRI application

# Stage 1: Build environment
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production environment  
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash mtri

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/mtri/.local

# Copy application code
COPY src/ ./src/
COPY data/sample/ ./data/sample/
COPY models/ ./models/
COPY configs/ ./configs/

# Copy configuration files
COPY .env.example ./.env

# Set ownership and switch to non-root user
RUN chown -R mtri:mtri /app
USER mtri

# Add local Python packages to PATH
ENV PATH=/home/mtri/.local/bin:$PATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Multi-service support via docker-compose
LABEL org.opencontainers.image.title="M-TRI API"
LABEL org.opencontainers.image.description="Microbial Toxin-Risk Index API service"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="M-TRI Team"