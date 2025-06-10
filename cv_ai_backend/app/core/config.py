"""
CV-AI Backend Ultimate Configuration Management v3.0
Combining comprehensive features with clean security architecture

FEATURES:
- Environment-specific configurations with production hardening
- Security-first API key handling (never logged)
- Advanced performance tuning parameters
- Comprehensive monitoring and observability settings
- Memory leak prevention with automatic limits
- Latest FastAPI 0.115+ and Pydantic v2 optimizations
"""

import os
import sys
import secrets
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum
from functools import lru_cache
import logging

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, computed_field, ConfigDict

from pydantic.networks import AnyHttpUrl

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class CacheBackend(str, Enum):
    """Cache backend options"""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"

class UltimateSettings(BaseSettings):
    """
    Ultimate Configuration Management v3.0

    FEATURES:
    - Security-first API key validation (CRITICAL FIX from cvbrain2)
    - Comprehensive monitoring settings (from cvbrain)
    - Advanced performance tuning (hybrid approach)
    - Environment-specific defaults (enhanced)
    - Memory protection limits (CRITICAL)
    - Latest 2024 FastAPI/Pydantic patterns
    """

    model_config = ConfigDict(
        # Latest Pydantic v2 optimizations
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        case_sensitive=False,
        env_prefix="CV_AI_",
        env_file=[".env", ".env.local"],
        env_file_encoding="utf-8"
    )

    # ========================================================================
    # Application Metadata
    # ========================================================================
    app_name: str = Field(default="CV-AI Backend Ultimate", description="Application name")
    app_version: str = Field(default="3.0.0", description="Application version")
    app_description: str = Field(
        default="Ultimate Enterprise AI-powered CV query system with full observability",
        description="Application description"
    )

    # ========================================================================
    # Environment Configuration (Enhanced)
    # ========================================================================
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Current deployment environment"
    )
    debug: bool = Field(default=True, description="Enable debug mode")
    testing: bool = Field(default=False, description="Testing mode flag")

    # ========================================================================
    # Server Configuration (FastAPI 0.115+ Optimized)
    # ========================================================================
    api_host: str = Field(default="127.0.0.1", description="API server host")
    api_port: int = Field(default=8000, ge=1000, le=65535, description="API server port")
    api_reload: bool = Field(default=True, description="Enable auto-reload (development)")
    api_workers: int = Field(default=1, ge=1, le=16, description="Number of worker processes")

    # Advanced performance settings (Latest patterns)
    max_concurrent_requests: int = Field(default=100, ge=1, le=10000)
    request_timeout_seconds: int = Field(default=30, ge=5, le=300)
    keepalive_timeout: int = Field(default=5, ge=1, le=60)
    max_request_size: int = Field(default=1048576, description="Max request size in bytes")

    # WebSocket support (FastAPI 0.115+ feature)
    enable_websockets: bool = Field(default=False, description="Enable WebSocket support")
    websocket_ping_interval: int = Field(default=20, ge=5, le=300)

    # ========================================================================
    # Security Configuration (CRITICAL - cvbrain2 approach)
    # ========================================================================
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for signing tokens"
    )
    allowed_hosts: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed host headers"
    )

    # Rate limiting (Memory leak prevention)
    rate_limit_requests: int = Field(default=100, ge=1, le=10000)
    rate_limit_burst: int = Field(default=50, ge=1, le=1000)
    rate_limit_window_minutes: int = Field(default=1, ge=1, le=60)
    rate_limit_max_clients: int = Field(default=10000, ge=100, le=100000)  # CRITICAL

    # Security headers (Production hardening)
    enable_security_headers: bool = Field(default=True)
    enable_hsts: bool = Field(default=False)  # Enable in production
    content_security_policy: Optional[str] = Field(default=None)

    # ========================================================================
    # OpenAI Configuration (Enhanced Security)
    # ========================================================================
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    openai_organization: Optional[str] = Field(default=None, description="OpenAI organization ID")
    openai_model: str = Field(default="gpt-4o-mini", description="Default OpenAI model")
    openai_max_tokens: int = Field(default=800, ge=100, le=4000)
    openai_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    openai_timeout: int = Field(default=30, ge=5, le=120)
    openai_max_retries: int = Field(default=3, ge=0, le=10)

    # Connection pooling (Production optimization)
    openai_max_connections: int = Field(default=10, ge=1, le=100)
    openai_keepalive_connections: int = Field(default=5, ge=1, le=50)
    openai_connection_timeout: float = Field(default=10.0, ge=1.0, le=60.0)

    # Latest OpenAI features (December 2024)
    openai_enable_streaming: bool = Field(default=False, description="Enable response streaming")
    openai_enable_function_calling: bool = Field(default=False, description="Enable function calling")

    # ========================================================================
    # Vector Database Configuration (ChromaDB 1.0.11+)
    # ========================================================================
    chroma_persist_dir: str = Field(
        default="../embeddings/chroma",
        description="ChromaDB persistence directory"
    )
    chroma_collection_name: str = Field(
        default="cv_chunks",
        description="ChromaDB collection name"
    )

    # ChromaDB 1.0.11+ specific settings
    chroma_allow_reset: bool = Field(default=False, description="Allow database reset")
    chroma_anonymized_telemetry: bool = Field(default=False, description="Send anonymous telemetry")
    chroma_auth_enabled: bool = Field(default=False, description="Enable authentication")

    # Embedding settings (Performance optimized)
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    embedding_cache_size: int = Field(default=1000, ge=100, le=10000)
    embedding_cache_ttl: int = Field(default=3600, ge=300, le=86400)
    embedding_batch_size: int = Field(default=10, ge=1, le=100)

    # ========================================================================
    # Query Processing (Enhanced)
    # ========================================================================
    default_query_chunks: int = Field(default=3, ge=1, le=10)
    max_query_chunks: int = Field(default=8, ge=1, le=20)
    query_timeout_seconds: int = Field(default=30, ge=5, le=120)

    # Advanced query features
    enable_query_caching: bool = Field(default=True)
    query_cache_size: int = Field(default=500, ge=50, le=5000)
    query_cache_ttl: int = Field(default=1800, ge=300, le=3600)
    enable_query_classification: bool = Field(default=True)
    enable_confidence_scoring: bool = Field(default=True)

    # ========================================================================
    # Caching Configuration (Hybrid Approach)
    # ========================================================================
    cache_backend: CacheBackend = Field(default=CacheBackend.MEMORY)

    # Memory cache settings
    memory_cache_size: int = Field(default=1000, ge=100, le=10000)
    memory_cache_ttl: int = Field(default=3600, ge=300, le=86400)

    # Redis cache settings (optional)
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    redis_timeout: int = Field(default=5, ge=1, le=30)
    redis_max_connections: int = Field(default=10, ge=1, le=100)
    redis_retry_on_timeout: bool = Field(default=True)

    # ========================================================================
    # CORS Configuration (Enhanced)
    # ========================================================================
    enable_cors: bool = Field(default=True)
    cors_origins: List[Union[str, AnyHttpUrl]] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:3001",
            "http://127.0.0.1:3001"
        ]
    )
    cors_credentials: bool = Field(default=True)
    cors_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]
    )
    cors_headers: List[str] = Field(
        default_factory=lambda: ["*"]
    )
    cors_max_age: int = Field(default=600)

    # ========================================================================
    # Logging Configuration (Structured)
    # ========================================================================
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_json_format: bool = Field(default=False)
    log_file_path: Optional[Path] = Field(default=None)
    log_correlation_id: bool = Field(default=True)
    log_request_body: bool = Field(default=False)  # Security consideration
    log_response_body: bool = Field(default=False)  # Security consideration

    # ========================================================================
    # Monitoring and Metrics (Production)
    # ========================================================================
    enable_metrics: bool = Field(default=False)
    metrics_port: int = Field(default=8001, ge=1000, le=65535)
    metrics_path: str = Field(default="/metrics")
    enable_health_checks: bool = Field(default=True)
    health_check_interval: int = Field(default=60, ge=10, le=300)

    # Performance monitoring
    enable_performance_monitoring: bool = Field(default=False)
    performance_collection_interval: int = Field(default=60, ge=10, le=300)
    enable_memory_profiling: bool = Field(default=False)

    # Error tracking
    enable_error_tracking: bool = Field(default=True)
    error_retention_days: int = Field(default=7, ge=1, le=30)

    # ========================================================================
    # Background Tasks (Enhanced)
    # ========================================================================
    enable_background_tasks: bool = Field(default=True)
    background_task_interval: int = Field(default=300, ge=60, le=3600)  # 5 minutes
    cleanup_task_interval: int = Field(default=3600, ge=300, le=86400)  # 1 hour

    # ========================================================================
    # VALIDATORS (CRITICAL SECURITY)
    # ========================================================================

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """CRITICAL: Validate API key without logging sensitive data"""
        if not v or not v.startswith(("sk-", "sk-proj-")):
            raise ValueError("OpenAI API key must start with 'sk-' or 'sk-proj-'")
        if len(v) < 20:
            raise ValueError("OpenAI API key appears invalid (too short)")
        # SECURITY: Never log the actual key - this is the critical fix
        return v

    @field_validator("chroma_persist_dir")
    @classmethod
    def validate_chroma_dir(cls, v: str) -> str:
        """Enhanced directory validation with proper error handling"""
        path = Path(v).expanduser().resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return str(path)
        except Exception as e:
            raise ValueError(f"Cannot create or write to ChromaDB directory {v}: {e}")

    @field_validator("redis_url", mode="before")
    @classmethod
    def validate_redis_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate Redis URL format"""
        if v is None:
            return v
        if not v.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError("Redis URL must start with redis://, rediss://, or unix://")
        return v

    @field_validator("log_file_path", mode="before")
    @classmethod
    def validate_log_path(cls, v: Union[str, Path, None]) -> Optional[Path]:
        """Validate log file path and create directory if needed"""
        if v is None:
            return v
        path = Path(v).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ========================================================================
    # COMPUTED PROPERTIES & METHODS
    # ========================================================================

    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION

    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT

    def get_masked_api_key(self) -> str:
        """SECURITY: Get masked API key for safe logging"""
        if not self.openai_api_key:
            return "NOT_SET"
        return f"{self.openai_api_key[:8]}...{self.openai_api_key[-4:]}"

    def get_openai_client_config(self) -> Dict[str, Any]:
        """Get OpenAI client configuration with latest settings"""
        config = {
            "api_key": self.openai_api_key,
            "timeout": self.openai_timeout,
            "max_retries": self.openai_max_retries,
        }

        # Add organization if provided
        if self.openai_organization:
            config["organization"] = self.openai_organization

        # HTTP client configuration
        config["http_client_config"] = {
            "limits": {
                "max_connections": self.openai_max_connections,
                "max_keepalive_connections": self.openai_keepalive_connections
            },
            "timeout": {
                "connect": self.openai_connection_timeout,
                "read": self.openai_timeout,
                "write": 10.0,
                "pool": 5.0
            }
        }

        return config

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration based on backend"""
        config = {
            "backend": self.cache_backend.value,
            "memory": {
                "size": self.memory_cache_size,
                "ttl": self.memory_cache_ttl
            }
        }

        if self.cache_backend in [CacheBackend.REDIS, CacheBackend.HYBRID] and self.redis_url:
            config["redis"] = {
                "url": self.redis_url,
                "timeout": self.redis_timeout,
                "max_connections": self.redis_max_connections,
                "retry_on_timeout": self.redis_retry_on_timeout
            }

        return config

    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers configuration"""
        headers = {}

        if self.enable_security_headers:
            headers.update({
                "X-Frame-Options": "DENY",
                "X-Content-Type-Options": "nosniff",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin"
            })

            if self.enable_hsts and self.is_production:
                headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

            if self.content_security_policy:
                headers["Content-Security-Policy"] = self.content_security_policy

        return headers

    # ========================================================================
    # ENVIRONMENT-SPECIFIC CONFIGURATION
    # ========================================================================

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization configuration based on environment"""  # ✅ Properly indented

        if self.environment == Environment.PRODUCTION:
            # Production hardening - Use object.__setattr__ to bypass validation
            object.__setattr__(self, 'debug', False)
            object.__setattr__(self, 'api_reload', False)
            object.__setattr__(self, 'log_json_format', True)
            object.__setattr__(self, 'enable_metrics', True)
            object.__setattr__(self, 'enable_performance_monitoring', True)
            object.__setattr__(self, 'enable_hsts', True)
            object.__setattr__(self, 'log_request_body', False)
            object.__setattr__(self, 'log_response_body', False)

            # Stricter rate limiting in production
            if self.rate_limit_requests > 100:
                object.__setattr__(self, 'rate_limit_requests', 60)

        elif self.environment == Environment.DEVELOPMENT:
            # Development convenience
            object.__setattr__(self, 'debug', True)
            object.__setattr__(self, 'api_reload', True)
            object.__setattr__(self, 'log_level', LogLevel.DEBUG)
            object.__setattr__(self, 'enable_metrics', False)
            object.__setattr__(self, 'enable_performance_monitoring', False)

            # More lenient rate limiting for development
            current_rate_limit = max(self.rate_limit_requests, 1000)
            object.__setattr__(self, 'rate_limit_requests', current_rate_limit)

        elif self.environment == Environment.TESTING:
            # Testing optimizations
            object.__setattr__(self, 'debug', True)
            object.__setattr__(self, 'log_level', LogLevel.DEBUG)
            object.__setattr__(self, 'enable_metrics', False)
            object.__setattr__(self, 'enable_query_caching', False)
            object.__setattr__(self, 'query_timeout_seconds', 10)
            object.__setattr__(self, 'openai_timeout', 10)

@lru_cache()
def get_settings() -> UltimateSettings:
    """Get cached application settings (singleton pattern)"""
    return UltimateSettings()

# Export settings
settings = get_settings()
