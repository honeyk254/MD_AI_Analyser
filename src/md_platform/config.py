"""Global Configuration."""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    app_name: str = "MD AI Platform"
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    output_dir: str = os.getenv("OUTPUT_DIR", "data/outputs")
    data_dir: str = os.getenv("DATA_DIR", "data/inputs")
    
    class Config:
        env_file = ".env"

settings = Settings()
