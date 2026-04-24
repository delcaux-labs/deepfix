## deepfix-portal

Backend service for the DeepFix Portal, providing user management, authentication, and API key issuance for the DeepFix platform.

### Overview

`deepfix-portal` is a FastAPI-based backend that:

- **Authenticates users** with email/password and JWTs.
- **Manages users** (profiles, admin/users separation).
- **Issues and manages API keys** for accessing DeepFix services.
- **Validates API keys** for server-to-server communication (used by `deepfix-server`).
- **Serves the Portal SPA** in production when a built frontend is available.

The main application is exposed as `deepfix_portal.api.main:app`.

### Features

- **FastAPI backend**: typed request/response models via Pydantic.
- **JWT authentication**:
  - Login, signup, logout, and "who am I" (`/api/auth/me`) endpoints.
  - Configurable token lifetime.
- **User management**:
  - Admin-only endpoint to list all users.
  - Per-user access control on profile reads.
- **API key management**:
  - Create, list, and revoke per-user API keys.
  - Cached validation endpoint for high-throughput server-to-server checks.
- **Database-agnostic**:
  - Uses SQLAlchemy; default is SQLite, but Postgres (or others) are supported via `DATABASE_URL`.
- **Production-ready basics**:
  - CORS configuration for local frontends.
  - SPA/static file serving when running in production mode.

### Directory structure

The core backend lives under `deepfix_portal.api`:

- **`main.py`**: FastAPI app factory and entrypoint.
- **`database.py`**: SQLAlchemy engine/session and `Base` configuration.
- **`models.py`**: ORM models (`User`, `APIKey`, plus `RequestLog` re-exported from `deepfix-core`).
- **`schemas.py`**: Pydantic request/response schemas.
- **`routes/`**:
  - `auth.py` – login, signup, logout, and `/me`.
  - `users.py` – admin/user profile endpoints.
  - `api_keys.py` – CRUD and validation for API keys.
- **`dependencies.py`**: FastAPI dependency helpers (current user, admin user, service-token verification).
- **`utils.py`**: Password hashing and JWT helpers.
- **`migrations/`**: Alembic configuration and migration scripts.

### Installation

From the monorepo root (with `uv`):

```bash
uv sync
```

Then, to run commands in the `deepfix-portal` environment:

```bash
uv pip install -e deepfix-portal
```

Or, if you are working only with this package:

```bash
cd deepfix-portal
pip install -e .
```

### Configuration

`deepfix-portal` is configured primarily through environment variables:

- **`DATABASE_URL`** (optional): SQLAlchemy-style database URL.
  - Default: `sqlite:///./deepfix.db`
  - Example for Postgres: `postgresql+psycopg2://user:password@localhost:5432/deepfix`
- **`DEEPFIX_PORTAL_SECRET_KEY`** (required): Secret key for signing JWTs.
- **`DEEPFIX_PORTAL_ALGORITHM`** (required): JWT signing algorithm (for example `HS256`).
- **`ACCESS_TOKEN_EXPIRE_MINUTES`** (optional): JWT access token lifetime in minutes (default: `30`).
- **`DEEPFIX_PORTAL_SERVICE_TOKEN`** (required for server-to-server validation):
  - Shared secret used by other DeepFix services (such as `deepfix-server`) when calling the `/api/api-keys/validate` endpoint via the `X-Service-Token` header.

You can set these in your shell, `.env` file, or container/orchestration configuration.

### Running the backend locally

From the monorepo root (recommended with `uv`):

```bash
uv run uvicorn deepfix_portal.api.main:app --reload --host 0.0.0.0 --port 5041
```

Or with a regular Python environment:

```bash
uvicorn deepfix_portal.api.main:app --reload --host 0.0.0.0 --port 5041
```

By default, CORS is enabled for:

- `http://localhost:8844`
- `http://localhost:5173`

Adjust these in `deepfix_portal/api/main.py` if your frontend runs on different origins.

### API overview

All backend routes are prefixed with `/api`.

- **Health**
  - `GET /api/health` – simple health check (`{"status": "ok", ...}`).

- **Authentication (`/api/auth`)**
  - `POST /api/auth/login` – email/password login, returns `AuthResponse` with `access_token`.
  - `POST /api/auth/signup` – create a new user and return an access token.
  - `POST /api/auth/logout` – logical logout (JWT should be removed client‑side).
  - `GET /api/auth/me` – return the authenticated user (`UserResponse`).

- **Users (`/api/users`)**
  - `GET /api/users/` – list all users (admin only).
  - `GET /api/users/{user_id}` – get a user; non-admin users can only see their own profile.

- **API keys (`/api/api-keys`)**
  - `GET /api/api-keys/` – list active API keys for the current user.
  - `POST /api/api-keys/` – create a new API key (`df_live_...`) for the current user.
  - `GET /api/api-keys/{key_id}` – get details about a specific key.
  - `DELETE /api/api-keys/{key_id}` – revoke (deactivate) an API key.
  - `POST /api/api-keys/validate` – validate an API key for server-to-server access.
    - Requires `X-Service-Token: <DEEPFIX_PORTAL_SERVICE_TOKEN>` header.
    - Returns `APIKeyValidationResponse` with both key and user metadata.

### Database and migrations

The default setup uses a local SQLite file (`deepfix.db`). For production deployments, point `DATABASE_URL` to your managed Postgres (or other supported) database.

Alembic migration scripts live under `deepfix_portal/api/migrations`. A typical upgrade command looks like:

```bash
uv run alembic -c deepfix_portal/api/alembic.ini upgrade head
```

Ensure that your `DATABASE_URL` is set before running migrations.

### Development

- **Code style**: The project uses `ruff` for linting and formatting (see `pyproject.toml`).
- **Python version**: `>=3.11`.
- **Workspace**: `deepfix-portal` is part of the broader DeepFix `uv` workspace alongside `deepfix-core`, `deepfix-server`, and `deepfix-sdk`.

When contributing, try to:

- Keep models, schemas, and routes small and focused.
- Add or update Pydantic schemas alongside any changes to the SQLAlchemy models.
- Extend this README with any new configuration or endpoints you introduce.

## Database Migrations

This directory contains Alembic database migrations.

### Initial Setup

If this is a fresh setup, you may need to initialize Alembic:

```bash
cd deepfix-portal/src/deepfix_portal
alembic init migrations
```

Then update `migrations/env.py` to import your models (already done if using the provided setup).

### Creating Migrations

After making changes to models in `api/models.py`:

```bash
alembic revision --autogenerate -m "Description of changes"
```

### Applying Migrations

```bash
alembic upgrade head
```

### Rolling Back

```bash
alembic downgrade -1  # Roll back one migration
alembic downgrade base  # Roll back all migrations
```
