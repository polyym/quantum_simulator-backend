# src/config.py

"""
Centralized configuration management for the Quantum Simulator Backend.
Uses pydantic-settings for typed environment variable loading.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables can be set directly or via a .env file.
    """

    # Application settings
    app_name: str = "Quantum Circuit Simulator"
    app_version: str = "3.0.0"
    debug: bool = False

    # CORS settings
    # Note: "null" origin is needed when opening test_form.html directly from filesystem
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "null"],
        description="List of allowed CORS origins. Use ['*'] only for development."
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]

    # HPC Resource settings
    hpc_total_cores: int = Field(default=64, ge=1, description="Total CPU cores available")
    hpc_total_gpus: int = Field(default=4, ge=0, description="Total GPU cards available")
    hpc_total_memory_gb: float = Field(default=128.0, gt=0, description="Total memory in GB")

    # Job settings
    hpc_job_cycle_delay: float = Field(default=0.1, gt=0, description="Delay between HPC job cycles in seconds")
    hpc_job_retention_hours: int = Field(default=24, ge=1, description="Hours to retain completed jobs")
    hpc_max_jobs: int = Field(default=1000, ge=1, description="Maximum number of jobs to track")

    # Quantum system settings
    quantum_max_qubits: int = Field(
        default=25, ge=1, le=40,
        description="Maximum qubits per system. Note: State vector simulation requires "
                    "O(2^n) memory. 20 qubits ≈ 16 MB, 25 qubits ≈ 512 MB, 30 qubits ≈ 16 GB."
    )
    quantum_practical_qubits: int = Field(
        default=20, ge=1, le=30,
        description="Recommended maximum qubits for practical simulation without HPC."
    )
    quantum_system_ttl_hours: int = Field(default=24, ge=1, description="TTL for quantum systems in hours")
    quantum_max_systems: int = Field(default=100, ge=1, description="Maximum concurrent quantum systems")

    # Rate limiting (requests per minute)
    rate_limit_enabled: bool = False
    rate_limit_requests_per_minute: int = Field(default=60, ge=1)

    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Server settings
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    model_config = {
        "env_prefix": "QSIM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings instance (cached for performance)
    """
    return Settings()


# Convenience function to reload settings (useful for testing)
def reload_settings() -> Settings:
    """
    Reload settings by clearing the cache.

    Returns:
        Fresh Settings instance
    """
    get_settings.cache_clear()
    return get_settings()
