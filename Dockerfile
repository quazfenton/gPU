# =============================================================================
# Notebook ML Orchestrator - Production Dockerfile
# =============================================================================
# Multi-stage build for optimal image size and security
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and build
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /app
COPY requirements.txt requirements-orchestrator.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -r requirements-orchestrator.txt

# Install security dependencies
RUN pip install cryptography PyJWT bcrypt requests pyotp

# Copy source code
COPY notebook_ml_orchestrator/ ./notebook_ml_orchestrator/
COPY gui/ ./gui/
COPY templates/ ./templates/
COPY setup.py ./

# Install the package
RUN pip install -e .

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

# Labels
LABEL maintainer="Notebook ML Orchestrator Team" \
      version="1.0.0" \
      description="ML orchestration platform with multi-backend support"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONFAULTHANDLER=1 \
    # Application settings
    APP_HOME=/app \
    APP_USER=orchestrator \
    APP_GROUP=orchestrator \
    # Security defaults
    GUI_HOST=0.0.0.0 \
    GUI_PORT=7860 \
    GUI_ENABLE_AUTH=false \
    GUI_ENABLE_RATE_LIMITING=true \
    # Paths
    UPLOAD_DIR=/app/uploads \
    LOG_DIR=/app/logs \
    DB_DIR=/app/data

# Create non-root user for security
RUN groupadd --gid 1000 ${APP_GROUP} && \
    useradd --uid 1000 --gid ${APP_GROUP} --shell /bin/bash --create-home ${APP_USER}

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
WORKDIR ${APP_HOME}
COPY --from=builder /app/notebook_ml_orchestrator ./notebook_ml_orchestrator
COPY --from=builder /app/gui ./gui
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/setup.py ./

# Create directories with proper permissions
RUN mkdir -p ${UPLOAD_DIR} ${LOG_DIR} ${DB_DIR} && \
    chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME} && \
    chmod -R 755 ${APP_HOME}

# Copy and set up entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && \
    chown ${APP_USER}:${APP_GROUP} /usr/local/bin/entrypoint.sh

# Switch to non-root user
USER ${APP_USER}

# Expose ports
EXPOSE 7860 7861

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["python", "-m", "gui.main"]

# -----------------------------------------------------------------------------
# Stage 3: Development - With debug tools
# -----------------------------------------------------------------------------
FROM production as development

# Switch back to root for installing dev dependencies
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install dev Python packages
RUN pip install \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    hypothesis

# Copy development configuration
COPY .flake8 ./
COPY mypy.ini ./

# Switch back to non-root user
USER ${APP_USER}

# Enable debug mode
ENV DEBUG=true \
    VERBOSE_LOGGING=true \
    HOT_RELOAD=true

# Expose debug port
EXPOSE 5678

# Use the same entrypoint as production so initialization still runs; only override the default command
CMD ["python", "-m", "gui.main", "--debug", "--host", "0.0.0.0", "--port", "7860"]

# =============================================================================
# Build Instructions:
# =============================================================================
# Production build:
#   docker build -t notebook-ml-orchestrator:latest --target production .
#
# Development build:
#   docker build -t notebook-ml-orchestrator:dev --target development .
#
# Run production:
#   docker run -p 7860:7860 -p 7861:7861 \
#     -e MASTER_KEY=your-master-key \
#     -e JWT_SECRET=your-jwt-secret \
#     notebook-ml-orchestrator:latest
#
# Run development:
#   docker run -p 7860:7860 -p 7861:7861 -v $(pwd):/app \
#     notebook-ml-orchestrator:dev
# =============================================================================
