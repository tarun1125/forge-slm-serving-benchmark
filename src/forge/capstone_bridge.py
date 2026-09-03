"""Single choke point for importing the capstone repo's eval harness.

FORGE deliberately does NOT re-derive the 1,517-case dataset's execution
scoring, AST safety allowlist, or result-comparison logic — that harness
already exists, is validated, and lives in the capstone repo. Re-deriving it
here would be exactly the kind of "two implementations that quietly diverge"
problem the capstone's own code comments repeatedly flag in itself. Every
cross-repo import goes through this module so there is one place that does
the sys.path surgery, not one per script.

See docs/parity-check-design.md for why FORGE depends on this repo at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from forge.config import Settings


def _import_from(repo_root: Path, module_name: str, subdir: str | None = None) -> ModuleType:
    import importlib

    search_path = str(repo_root / subdir) if subdir else str(repo_root)
    if search_path not in sys.path:
        sys.path.insert(0, search_path)
    return importlib.import_module(module_name)


class CapstoneHarness:
    """Lazily-imported handle onto the capstone repo's data files and eval code."""

    def __init__(self, settings: Settings):
        self.repo_root = settings.require_capstone_repo()
        self.credentials_file = self.repo_root / "atlas-credentials.env"

    # -- data files -----------------------------------------------------
    @property
    def rag_test_path(self) -> Path:
        return self.repo_root / "rag" / "data" / "rag_test.json"

    @property
    def gold_results_path(self) -> Path:
        return self.repo_root / "data" / "gold_results.json"

    @property
    def split_manifest_path(self) -> Path:
        return self.repo_root / "fine_tuning" / "data_23db" / "split_manifest.json"

    # -- code, imported cross-repo rather than duplicated ---------------
    @property
    def execute_queries(self) -> ModuleType:
        """evaluation/execute_queries.py — safe_eval_query, results_match,
        connect(), run_model(). connect() auto-discovers
        atlas-credentials.env by walking up from its own file, so no Mongo
        URI needs to be duplicated into FORGE's own .env."""
        return _import_from(self.repo_root, "execute_queries", subdir="evaluation")

    @property
    def spot_check(self) -> ModuleType:
        """fine_tuning/spot_check.py — STOP_MARKERS, clean(). Postprocesses
        raw mlx_lm output the same way the capstone's own generation scripts
        do, so parity comparisons aren't polluted by stop-token noise."""
        return _import_from(self.repo_root, "spot_check", subdir="fine_tuning")

    @property
    def normalize(self) -> ModuleType:
        """normalize.py — fixes the two confirmed Mongo-shell/JS dialect
        habits (bare null/true/false, camelCase method names) at the AST
        level before a generated query is executed."""
        return _import_from(self.repo_root, "normalize")

    def check_atlas_credentials(self) -> None:
        if not self.credentials_file.exists():
            raise RuntimeError(
                f"{self.credentials_file} not found. Execution-accuracy scoring needs a live "
                "Atlas connection — this file lives in the capstone repo, not FORGE, by design "
                "(one copy of the secret). See that repo's atlas_env.py / atlas-credentials.env."
            )
