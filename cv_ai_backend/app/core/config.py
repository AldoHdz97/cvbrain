"""
CV-AI Backend Configuration
Clean, secure, and production-ready settings
"""

import os
import secrets
from pathlib import Path
from typing import List, Optional
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

class Settings(BaseSettings):
    """Application settings with security best practices"""
    
    model_config = {
        "env_prefix": "CV_AI_",
        "env_file": ".env",
        "case_sensitive": False
    }

    # App Info
    app_name: str = "CV-AI Backend"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", pattern="^(development|staging|production)$")
    debug: bool = Field(default=True)

    # Server
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1000, le=65535)
    
    # Security
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1"])
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = Field(default=800, ge=100, le=2000)
    openai_temperature: float = Field(default=0.4, ge=0.0, le=1.0)
    openai_timeout: int = Field(default=30, ge=5, le=60)

    # ChromaDB
    chroma_persist_dir: str = "../embeddings/chroma"
    chroma_collection_name: str = "cv_chunks"
    embedding_model: str = "text-embedding-3-small"

    # Query Processing
    default_query_chunks: int = Field(default=3, ge=1, le=8)
    query_timeout_seconds: int = Field(default=30, ge=5, le=60)

    # CORS - Secure defaults
    enable_cors: bool = True
    cors_origins: List[str] = Field(default=[
        "http://localhost:3000",
        "http://localhost:8080",
        "https://your-frontend-domain.com"  # Replace with actual domain
    ])
    cors_credentials: bool = False  # Secure default
    cors_methods: List[str] = Field(default=["GET", "POST"])
    cors_headers: List[str] = Field(default=["Content-Type", "Authorization"])

    # Caching
    enable_caching: bool = True
    cache_ttl: int = Field(default=1800, ge=300, le=3600)  # 30 minutes
    redis_url: Optional[str] = None

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Validate OpenAI API key format"""
        if not v or not v.startswith(("sk-", "sk-proj-")):
            raise ValueError("Invalid OpenAI API key format")
        if len(v) < 40:  # Proper minimum length
            raise ValueError("OpenAI API key too short")
        return v

    @field_validator("chroma_persist_dir")
    @classmethod
    def validate_chroma_dir(cls, v: str) -> str:
        """Ensure ChromaDB directory exists"""
        path = Path(v).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"

    def get_cors_config(self) -> dict:
        """Get CORS configuration"""
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": self.cors_credentials,
            "allow_methods": self.cors_methods,
            "allow_headers": self.cors_headers,
        }

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Export settings instance
settings = get_settings()
