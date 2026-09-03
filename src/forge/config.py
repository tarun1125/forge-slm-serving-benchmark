"""Typed settings loaded from .env — the single source of truth for paths and
keys. Per AGENTS.md: all external input (including config) is validated with
Pydantic at the boundary; nothing downstream should read os.environ directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = "INFO"
    mlflow_tracking_uri: str = "./mlruns"

    groq_api_key: str | None = None
    nim_api_key: str | None = None

    capstone_repo_path: Path = Field(
        default=Path(
            "/Users/tarungudapati/Documents/ai-projects/capstone-project/CSAIML-Capstone-Project-20"
        )
    )
    base_model: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"
    adapter_path: Path = Field(
        default=Path(
            "/Users/tarungudapati/Documents/ai-projects/capstone-project/"
            "CSAIML-Capstone-Project-20/fine_tuning/adapters_23db_1000iter"
        )
    )

    def require_capstone_repo(self) -> Path:
        if not self.capstone_repo_path.exists():
            raise RuntimeError(
                f"CAPSTONE_REPO_PATH={self.capstone_repo_path} does not exist. FORGE reuses the "
                "capstone's eval harness (rag_test.json, split_manifest.json, "
                "evaluation/execute_queries.py) rather than duplicating it — see "
                "docs/parity-check-design.md."
            )
        return self.capstone_repo_path


def get_settings() -> Settings:
    return Settings()
