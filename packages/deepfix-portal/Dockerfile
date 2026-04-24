# Stage 1: Build the React frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /app

# In the root context, we copy from the deepfix-portal package directory
COPY packages/deepfix-portal/package-lock.json packages/deepfix-portal/package.json ./
COPY packages/deepfix-portal/tsconfig.json \
     packages/deepfix-portal/vite.config.ts \
     packages/deepfix-portal/postcss.config.js \
     packages/deepfix-portal/components.json ./

# Copy source code needed for the build
COPY packages/deepfix-portal/client ./client
COPY packages/deepfix-portal/shared ./shared
COPY packages/deepfix-portal/script ./script

RUN npm install
RUN npm run build

# Stage 2: Python Backend Runtime
FROM astral/uv:python3.11-bookworm-slim AS runtime
WORKDIR /app

LABEL maintainer="fadel.seydou@delcaux.com"
LABEL version="0.0.1"
LABEL description="deepfix-portal-unified"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STATIC_DIR=/app/packages/deepfix-portal/dist/public

# System deps for compiling wheels like bcrypt/psycopg2, plus create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --no-create-home --gecos "" app

# Copy workspace structure (needed for uv to resolve workspace dependencies)
COPY pyproject.toml ./pyproject.toml
COPY packages/deepfix-core ./packages/deepfix-core
COPY packages/deepfix-portal ./packages/deepfix-portal
COPY packages/deepfix-server ./packages/deepfix-server
COPY packages/deepfix-kb ./packages/deepfix-kb

# Install Python deps into a project-local venv
RUN uv venv .venv
RUN uv pip install /app/packages/deepfix-core
RUN uv pip install /app/packages/deepfix-server
RUN uv pip install /app/packages/deepfix-portal
RUN uv pip install /app/packages/deepfix-kb

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/dist/public ${STATIC_DIR}

WORKDIR /app/packages/deepfix-portal

# Create directories with proper permissions for the app user
RUN mkdir -p .deepfix .cache .mplconfig .dspy-cache && chown -R app:app .deepfix .cache .mplconfig .dspy-cache

USER app
ENV PATH="/app/.venv/bin:${PATH}"
ENV DEEPFIX_HOME=/app/packages/deepfix-portal/.deepfix
ENV DSPY_CACHEDIR=/app/packages/deepfix-portal/.dspy-cache
ENV MPLCONFIGDIR=/app/packages/deepfix-portal/.mplconfig

EXPOSE 5041

# Start the FastAPI app
CMD ["uvicorn", "deepfix_portal.api.main:app", "--host", "0.0.0.0", "--port", "5041"]
