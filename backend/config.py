# backend/config.py
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Absolute path to the .env file so it's found regardless of working directory
_ENV_FILE = HERE / ".env"

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        # App Settings
        APP_NAME: str = "ZenithDx API"
        DEBUG: bool = True

        # Security & JWT Settings
        SECRET_KEY: str = "SUPER_SECRET_KEY_CHANGE_IN_PROD"
        ALGORITHM: str = "HS256"
        ACCESS_TOKEN_MINS: int = 60 * 24

        # Database Settings
        DATABASE_URL: str = "dbname=zenithdx_db user=zenithdx password=zenithdxsecret host=localhost port=5432"

        # External Services
        OLLAMA_HOST: str = "http://localhost:11434"
        OLLAMA_MODEL: str = "doctor2"

        # Directory Paths
        BASE_DIR: Path = HERE
        UPLOAD_DIR: Path = HERE / "uploads"
        OUTPUT_DIR: Path = HERE / "outputs"
        STATIC_DIR: Path = HERE / "static"
        DATA_DIR: Path = HERE / "data"

        # Model Weights & Data File Paths
        # SA-UNet is a TF SavedModel directory (nested: data/image/sa_unet_savedmodel/sa_unet_savedmodel)
        SA_UNET_WEIGHTS: Path = HERE / "data" / "image" / "sa_unet_savedmodel" / "sa_unet_savedmodel"
        RESNET_WEIGHTS: Path = HERE / "data" / "image" / "best_model.pth"
        CHEST_LABEL_COLS_PATH: Path = HERE / "data" / "chest_label_cols.csv"

        FAISS_PATIENT_INDEX_PATH: Path = HERE / "faiss_patient_index.bin"
        FAISS_PATIENT_MAPPING_PATH: Path = HERE / "faiss_patient_mapping.pkl"
        VISIT_EMB_PATH: Path = HERE / "visit_patient_emb.npz"
        FAISS_DIAG_INDEX_PATH: Path = HERE / "faiss_index_D.idx"
        CHUNKS_PATH: Path = HERE / "chunks.pkl"

        model_config = SettingsConfigDict(
            env_file=str(_ENV_FILE),
            env_file_encoding="utf-8",
            extra="ignore",
        )

except ImportError as _pydantic_err:
    print(
        f"[config] WARNING: pydantic-settings not available ({_pydantic_err}). "
        "Using plain dataclass fallback.",
        file=sys.stderr,
    )

    class Settings:  # type: ignore[no-redef]
        APP_NAME: str = "ZenithDx API"
        DEBUG: bool = True
        SECRET_KEY: str = os.getenv("SECRET_KEY", "SUPER_SECRET_KEY_CHANGE_IN_PROD")
        ALGORITHM: str = "HS256"
        ACCESS_TOKEN_MINS: int = int(os.getenv("ACCESS_TOKEN_MINS", "1440"))
        DATABASE_URL: str = os.getenv(
            "DATABASE_URL",
            "dbname=zenithdx_db user=zenithdx password=zenithdxsecret host=localhost port=5432",
        )
        OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "doctor2")
        BASE_DIR: Path = HERE
        UPLOAD_DIR: Path = HERE / "uploads"
        OUTPUT_DIR: Path = HERE / "outputs"
        STATIC_DIR: Path = HERE / "static"
        DATA_DIR: Path = HERE / "data"
        SA_UNET_WEIGHTS: Path = HERE / "data" / "image" / "sa_unet_savedmodel" / "sa_unet_savedmodel"
        RESNET_WEIGHTS: Path = HERE / "data" / "image" / "best_model.pth"
        CHEST_LABEL_COLS_PATH: Path = HERE / "data" / "chest_label_cols.csv"
        FAISS_PATIENT_INDEX_PATH: Path = HERE / "faiss_patient_index.bin"
        FAISS_PATIENT_MAPPING_PATH: Path = HERE / "faiss_patient_mapping.pkl"
        VISIT_EMB_PATH: Path = HERE / "visit_patient_emb.npz"
        FAISS_DIAG_INDEX_PATH: Path = HERE / "faiss_index_D.idx"
        CHUNKS_PATH: Path = HERE / "chunks.pkl"


try:
    settings = Settings()
    if "0.0.0.0" in str(settings.OLLAMA_HOST):
        settings.OLLAMA_HOST = "http://localhost:11434"
    print(f"[config] Settings loaded OK. OLLAMA_HOST={settings.OLLAMA_HOST}", file=sys.stderr)
except Exception as _settings_err:
    print(
        f"[config] CRITICAL: Failed to instantiate Settings: {_settings_err}\n"
        "Falling back to hardcoded defaults.",
        file=sys.stderr,
    )
    # Create a minimal settings object so imports don't break the server
    class _FallbackSettings:
        APP_NAME = "ZenithDx API"
        DEBUG = True
        SECRET_KEY = os.getenv("SECRET_KEY", "SUPER_SECRET_KEY_CHANGE_IN_PROD")
        ALGORITHM = "HS256"
        ACCESS_TOKEN_MINS = 1440
        DATABASE_URL = os.getenv(
            "DATABASE_URL",
            "dbname=zenithdx_db user=zenithdx password=zenithdxsecret host=localhost port=5432",
        )
        OLLAMA_HOST = "http://localhost:11434"
        OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "doctor2")
        BASE_DIR = HERE
        UPLOAD_DIR = HERE / "uploads"
        OUTPUT_DIR = HERE / "outputs"
        STATIC_DIR = HERE / "static"
        DATA_DIR = HERE / "data"
        SA_UNET_WEIGHTS = HERE / "data" / "image" / "sa_unet_savedmodel" / "sa_unet_savedmodel"
        RESNET_WEIGHTS = HERE / "data" / "image" / "best_model.pth"
        CHEST_LABEL_COLS_PATH = HERE / "data" / "chest_label_cols.csv"
        FAISS_PATIENT_INDEX_PATH = HERE / "faiss_patient_index.bin"
        FAISS_PATIENT_MAPPING_PATH = HERE / "faiss_patient_mapping.pkl"
        VISIT_EMB_PATH = HERE / "visit_patient_emb.npz"
        FAISS_DIAG_INDEX_PATH = HERE / "faiss_index_D.idx"
        CHUNKS_PATH = HERE / "chunks.pkl"

    settings = _FallbackSettings()

# Ensure required directories exist
try:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    settings.STATIC_DIR.mkdir(parents=True, exist_ok=True)
except Exception as _mkdir_err:
    print(f"[config] WARNING: Could not create required directories: {_mkdir_err}", file=sys.stderr)

