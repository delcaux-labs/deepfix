from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized settings for the DeepFix Portal application.
    Loads values from environment variables or a .env file.
    """

    # General & Infrastructure
    STATIC_DIR: Optional[str] = None # directory where the built frontend is served from. Set in Dockerfile for production
    FRONTEND_URL: str = "http://localhost:5173"

    # Database
    DATABASE_URL: str = "sqlite:///./deepfix.db"
    DB_POOL_RECYCLE: int = 3600

    # Authentication & Security
    DEEPFIX_PORTAL_SECRET_KEY: str = "placeholder_secret_key"  # Change in production
    DEEPFIX_PORTAL_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DEEPFIX_PORTAL_SERVICE_TOKEN: str = "placeholder_service_token"

    # DeepFix Analysis Proxy
    DEEPFIX_SERVER_URL: str = "http://localhost:8844"
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None

    # Email Service (SMTP)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: Optional[str] = None
    SMTP_FROM_NAME: str = "DeepFix"
    SMTP_USE_TLS: bool = True

    # Configuration for Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def get_smtp_config(self):
        """
        Get SMTP configuration from centralized settings
        """
        return {
            "host": self.SMTP_HOST,
            "port": self.SMTP_PORT,
            "username": self.SMTP_USER,
            "password": self.SMTP_PASSWORD,
            "from_email": self.SMTP_FROM_EMAIL
            or self.SMTP_USER
            or "noreply@deepfix.com",
            "from_name": self.SMTP_FROM_NAME,
            "use_tls": self.SMTP_USE_TLS,
        }
    
    def get_frontend_url(self) -> str:
        """
        Get frontend URL from centralized settings and validate it.
        """
        # Normalize the URL (remove trailing slashes)
        return self.FRONTEND_URL.rstrip("/")


# Instantiate the settings object
settings = Settings()
