"""
FastAPI main application entry point
"""

import os

from deepfix_core.models import DatabaseBase  # Base for RequestLog table
from deepfix_server.logging import setup_dspy_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .routes import analysis, api_keys, auth, request_logs, users

# Create database tables
Base.metadata.create_all(bind=engine)
# Also create tables from deepfix_core (request_logs table)
DatabaseBase.metadata.create_all(bind=engine)

app = FastAPI(
    title="DeepFix Portal Backend",
    description="Backend for DeepFix Portal",
    version="1.0.0",
)

# CORS configuration
# Defaults for local development
default_origins = ["http://localhost:5173", "http://localhost:8844"]

# Allowed origins from environment variable (comma-separated list)
cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
allowed_origins = (
    [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
    if cors_origins_env
    else []
)

# Add FRONTEND_URL if present
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url and frontend_url not in allowed_origins:
    allowed_origins.append(frontend_url)

# Use defaults if no origins specified via environment
if not allowed_origins:
    allowed_origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(api_keys.router, prefix="/api/api-keys", tags=["api-keys"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(
    request_logs.router, prefix="/api/request-logs", tags=["request-logs"]
)
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "DeepFix Portal Backend is running"}


if os.getenv("MLFLOW_EXP_NAME") and os.getenv("MLFLOW_TRACKING_URI"):
    setup_dspy_logging(
        experiment_name=os.getenv("MLFLOW_EXP_NAME"),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5041)
