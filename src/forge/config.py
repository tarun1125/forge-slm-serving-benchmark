"""Typed settings loaded from .env — the single source of truth for paths and
keys. Convention: all external input (including config) is validated with
Pydantic at the boundary; nothing downstream should read os.environ directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = "INFO"
    # sqlite, not the plain "./mlruns" folder store the master plan/handoff
    # guide originally specified: mlflow 3.15.2 has put the filesystem
    # backend into maintenance mode and hard-errors on it unless
    # MLFLOW_ALLOW_FILE_STORE=true is set — confirmed live, not assumed.
    # sqlite is what mlflow's own error message recommends instead, and it's
    # still just one local file, so nothing about "no server needed" is lost.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    groq_api_key: str | None = None
    nim_api_key: str | None = None

    # --- Phase 4: HF Hub model upload + Space deploy ---
    hf_token: str | None = None
    hf_username: str | None = None

    # No default: these are machine-specific paths, and hardcoding one
    # developer's home directory into a public repo both leaks that local
    # directory layout and silently hands anyone else a default that cannot
    # exist on their machine. Required via .env (see .env.example) and
    # validated by the require_* helpers below, which say exactly what to set.
    capstone_repo_path: Path | None = None
    base_model: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16"
    adapter_path: Path | None = None

    def require_capstone_repo(self) -> Path:
        if self.capstone_repo_path is None:
            raise RuntimeError(
                "CAPSTONE_REPO_PATH is not set. FORGE reuses the capstone's eval harness "
                "(rag_test.json, split_manifest.json, evaluation/execute_queries.py) rather "
                "than duplicating it — point this at your local clone of that repo. "
                "See .env.example and docs/parity-check-design.md."
            )
        if not self.capstone_repo_path.exists():
            raise RuntimeError(
                f"CAPSTONE_REPO_PATH={self.capstone_repo_path} does not exist. FORGE reuses the "
                "capstone's eval harness (rag_test.json, split_manifest.json, "
                "evaluation/execute_queries.py) rather than duplicating it — see "
                "docs/parity-check-design.md."
            )
        return self.capstone_repo_path

    def require_adapter_path(self) -> Path:
        if self.adapter_path is None:
            raise RuntimeError(
                "ADAPTER_PATH is not set. Phase 1 needs the LoRA adapter directory to fuse "
                "into the base model — point this at the fine-tuned adapter from the capstone "
                "project. See .env.example."
            )
        if not self.adapter_path.exists():
            raise RuntimeError(f"ADAPTER_PATH={self.adapter_path} does not exist.")
        return self.adapter_path


def get_settings() -> Settings:
    return Settings()
