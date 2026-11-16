"""Runtime settings read from the environment and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SERVER_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "catboost_iwpc6256_combined.cbm"
META_PATH = ARTIFACT_DIR / "catboost_iwpc6256_combined.json"
ENV_PATH = SERVER_DIR.parent / ".env"

DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
DEFAULT_MEDGEMMA_MODEL = "google/medgemma-27b-text-it:featherless-ai"


def read_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines so no python-dotenv dependency is needed."""
    if not path.exists():
        return {}
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    hf_token: str
    hf_medgemma_model: str

    @classmethod
    def load(cls, env_path: Path = ENV_PATH) -> Settings:
        file_values = read_env_file(env_path)

        def lookup(name: str, default: str = "") -> str:
            return os.environ.get(name) or file_values.get(name, default)

        return cls(
            openrouter_api_key=lookup("OPENROUTER_API_KEY"),
            openrouter_model=lookup("DEFAULT_MODEL", DEFAULT_OPENROUTER_MODEL),
            hf_token=lookup("HF_TOKEN"),
            hf_medgemma_model=lookup("HF_MEDGEMMA_MODEL", DEFAULT_MEDGEMMA_MODEL),
        )

    @property
    def explain_provider(self) -> str:
        if self.hf_token:
            return "huggingface (medgemma)"
        if self.openrouter_api_key:
            return "openrouter"
        return "none (deterministic fallback only)"
