FROM astral/uv:python3.11-bookworm-slim AS runtime
WORKDIR /app

LABEL maintainer="fadel.seydou@delcaux.com"
LABEL version="0.0.1"
LABEL description="deepfix-server"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user for security
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash deepfix

# Copy workspace structure (needed for uv to resolve workspace dependencies)
COPY pyproject.toml ./pyproject.toml
COPY packages/deepfix-core ./packages/deepfix-core
COPY packages/deepfix-server ./packages/deepfix-server
COPY packages/deepfix-kb ./packages/deepfix-kb

# Install Python deps into a project-local venv
RUN uv venv .venv
RUN uv pip install -e /app/packages/deepfix-core
RUN uv pip install -e /app/packages/deepfix-kb
RUN uv pip install -e /app/packages/deepfix-server
RUN uv pip install opentelemetry-distro opentelemetry-exporter-otlp
RUN . .venv/bin/activate && opentelemetry-bootstrap -a requirements > /tmp/otel-reqs.txt && uv pip install -r /tmp/otel-reqs.txt

# Create .deepfix and .cache directories with proper permissions for the deepfix user
RUN mkdir -p /app/.deepfix && chown -R deepfix:deepfix /app/.deepfix

USER deepfix
ENV PATH="/app/.venv/bin:${PATH}"
ENV DEEPFIX_HOME=/app/.deepfix
ENV DSPY_CACHEDIR=/app/.deepfix/.dspy_cache
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV DEEPFIX_SERVER_DATABASE_URL=sqlite:////app/.deepfix/deepfix_server.db

# Expose ports
EXPOSE 8844

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8844/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Launch the server
CMD ["deepfix-server", "launch", "-port", "8844", "-host", "0.0.0.0", "-workers", "1"]
