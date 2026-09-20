"""The two guards in forge.phase4.upload_fused_bf16 are the only thing between
a drifted or incomplete checkpoint and a GPU that will happily serve it. They
run once, on a 3 GB directory, immediately before an outward-facing upload —
so they get tested here against small synthetic directories instead, where
every failure mode is cheap to construct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.artifact_hash import hash_directory
from forge.phase4.upload_fused_bf16 import (
    REQUIRED_FILES,
    build_model_card,
    check_required_files,
    verify_against_manifest,
)


def _model_dir(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    d = tmp_path / "fused-bf16"
    d.mkdir()
    for name in REQUIRED_FILES:
        (d / name).write_text(f"contents of {name}", encoding="utf-8")
    for name, contents in (extra or {}).items():
        (d / name).write_text(contents, encoding="utf-8")
    return d


def _manifest(tmp_path: Path, model_dir: Path, **overrides) -> Path:
    payload = {
        "base_model": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16",
        "adapter_path_hash": "sha256:deadbeef",
        "parity_check": {"passed": True, "exact_match_rate": 0.9, "n_cases": 50},
        "variants": [
            {
                "name": "bf16",
                "path": str(model_dir),
                "format": "mlx",
                "hash": "sha256:" + hash_directory(model_dir),
                "size_bytes": 123,
            }
        ],
    }
    payload.update(overrides)
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestVerifyAgainstManifest:
    def test_accepts_an_unchanged_directory(self, tmp_path):
        d = _model_dir(tmp_path)
        entry = verify_against_manifest(d, _manifest(tmp_path, d))
        assert entry["name"] == "bf16"

    def test_rejects_a_changed_file(self, tmp_path):
        d = _model_dir(tmp_path)
        manifest = _manifest(tmp_path, d)
        (d / "config.json").write_text("changed after the parity check", encoding="utf-8")

        with pytest.raises(RuntimeError, match="no longer matches"):
            verify_against_manifest(d, manifest)

    def test_rejects_an_added_file(self, tmp_path):
        """hash_directory is order-independent but path-sensitive, so a stray
        file counts as drift — including a leftover README.md.generated."""
        d = _model_dir(tmp_path)
        manifest = _manifest(tmp_path, d)
        (d / "README.md.generated").write_text("leftover", encoding="utf-8")

        with pytest.raises(RuntimeError, match="no longer matches"):
            verify_against_manifest(d, manifest)

    def test_rejects_a_failed_parity_check(self, tmp_path):
        d = _model_dir(tmp_path)
        manifest = _manifest(
            tmp_path, d, parity_check={"passed": False, "exact_match_rate": 0.4, "n_cases": 50}
        )

        with pytest.raises(RuntimeError, match="parity_check.passed=False"):
            verify_against_manifest(d, manifest)

    def test_rejects_a_missing_manifest(self, tmp_path):
        d = _model_dir(tmp_path)

        with pytest.raises(RuntimeError, match="run forge.phase1.manifest first"):
            verify_against_manifest(d, tmp_path / "nope.json")

    def test_rejects_a_missing_model_dir(self, tmp_path):
        with pytest.raises(RuntimeError, match="does not exist"):
            verify_against_manifest(tmp_path / "nope", tmp_path / "also-nope.json")

    def test_rejects_an_unknown_variant(self, tmp_path):
        d = _model_dir(tmp_path)

        with pytest.raises(RuntimeError, match="no variant named"):
            verify_against_manifest(d, _manifest(tmp_path, d), variant_name="fp8")


class TestCheckRequiredFiles:
    def test_returns_files_in_declared_order(self, tmp_path):
        d = _model_dir(tmp_path)
        assert [p.name for p in check_required_files(d)] == REQUIRED_FILES

    def test_names_every_missing_file(self, tmp_path):
        d = _model_dir(tmp_path)
        (d / "config.json").unlink()
        (d / "tokenizer.json").unlink()

        with pytest.raises(RuntimeError) as exc:
            check_required_files(d)
        assert "config.json" in str(exc.value)
        assert "tokenizer.json" in str(exc.value)

    def test_missing_chat_template_is_a_hard_failure(self, tmp_path):
        """The silent one: a repo without it serves a model that formats
        prompts differently from every other arm and scores near zero
        without erroring. It must fail here, on the laptop."""
        d = _model_dir(tmp_path)
        (d / "chat_template.jinja").unlink()

        with pytest.raises(RuntimeError, match="chat_template.jinja"):
            check_required_files(d)

    def test_a_local_readme_is_not_required(self, tmp_path):
        """models/fused-bf16/README.md is mlx_lm's stub and gets replaced by
        the generated card, so its absence must not block an upload."""
        assert "README.md" not in REQUIRED_FILES
        check_required_files(_model_dir(tmp_path))


class TestModelCard:
    def test_carries_the_provenance_a_reader_needs(self, tmp_path):
        d = _model_dir(tmp_path)
        manifest_path = _manifest(tmp_path, d)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = verify_against_manifest(d, manifest_path)

        card = build_model_card(
            "forge-bf16", entry, manifest, check_required_files(d), "https://example.invalid/repo"
        )

        assert "90% exact match, passed" in card
        assert entry["hash"] in card
        assert manifest["adapter_path_hash"] in card
        assert "https://example.invalid/repo" in card
        # Every uploaded file is listed with its own hash, so the repo can be
        # verified file-by-file rather than trusted.
        for name in REQUIRED_FILES:
            assert f"`{name}`" in card

    def test_documents_the_chat_template_flag(self, tmp_path):
        d = _model_dir(tmp_path)
        manifest_path = _manifest(tmp_path, d)
        card = build_model_card(
            "forge-bf16",
            verify_against_manifest(d, manifest_path),
            json.loads(manifest_path.read_text(encoding="utf-8")),
            check_required_files(d),
            "https://example.invalid/repo",
        )
        assert "--chat-template" in card
        assert "float16" in card  # the Turing caveat
