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
    assert_visibility,
    build_model_card,
    check_required_files,
    upload,
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
        entry, manifest = verify_against_manifest(d, _manifest(tmp_path, d))
        assert entry["name"] == "bf16"
        # The whole manifest comes back too, so the caller doesn't re-read it.
        assert manifest["parity_check"]["passed"] is True

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
        entry, manifest = verify_against_manifest(d, manifest_path)

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
        entry, manifest = verify_against_manifest(d, manifest_path)
        card = build_model_card(
            "forge-bf16", entry, manifest, check_required_files(d), "https://example.invalid/repo"
        )
        assert "--chat-template" in card
        assert "float16" in card  # the Turing caveat


class TestVisibilityGuard:
    """create_repo(exist_ok=True) silently leaves an existing repo's visibility
    alone, so "private=True" is a request, not a guarantee. These cover the
    case where that difference would publish 3 GB of weights by accident."""

    class _Info:
        def __init__(self, private: bool):
            self.private = private

    class _Api:
        def __init__(self, private: bool):
            self._private = private

        def repo_info(self, repo_id, repo_type):
            return TestVisibilityGuard._Info(self._private)

    def test_raises_when_an_existing_repo_is_public(self):
        with pytest.raises(RuntimeError, match="already exists and is PUBLIC"):
            assert_visibility(self._Api(private=False), "me/forge-bf16", private=True)

    def test_passes_when_the_repo_is_private(self):
        assert_visibility(self._Api(private=True), "me/forge-bf16", private=True)

    def test_public_upload_accepts_a_public_repo(self):
        assert_visibility(self._Api(private=False), "me/forge-bf16", private=False)


class TestTokenGuard:
    def test_upload_refuses_without_a_token(self, tmp_path):
        d = _model_dir(tmp_path)
        with pytest.raises(RuntimeError, match="HF token is required"):
            upload(
                model_dir=d,
                manifest_path=_manifest(tmp_path, d),
                hf_token="",
                hf_username="me",
                repo_suffix="forge-bf16",
                github_repo_url="https://example.invalid/repo",
            )

    def test_dry_run_needs_no_token(self, tmp_path, capsys):
        """The whole point of --dry-run is that it can be run before you have
        credentials, so it must not trip the token guard."""
        d = _model_dir(tmp_path)
        repo_id = upload(
            model_dir=d,
            manifest_path=_manifest(tmp_path, d),
            hf_token="",
            hf_username="me",
            repo_suffix="forge-bf16",
            github_repo_url="https://example.invalid/repo",
            dry_run=True,
        )
        assert repo_id == "me/forge-bf16"
        assert "nothing uploaded" in capsys.readouterr().out

    def test_dry_run_still_runs_the_guards(self, tmp_path):
        """A dry run that skipped verification would be worthless as a
        pre-flight check. The manifest is recorded from the already-incomplete
        directory on purpose: that is the one case the drift guard cannot see,
        and the case check_required_files exists for."""
        d = _model_dir(tmp_path)
        (d / "chat_template.jinja").unlink()
        manifest = _manifest(tmp_path, d)

        with pytest.raises(RuntimeError, match="chat_template.jinja"):
            upload(
                model_dir=d,
                manifest_path=manifest,
                hf_token="",
                hf_username="me",
                repo_suffix="forge-bf16",
                github_repo_url="https://example.invalid/repo",
                dry_run=True,
            )

    def test_dry_run_writes_nothing_into_the_model_dir(self, tmp_path):
        """The card is built in memory and committed as bytes; a file left in
        model_dir would break the drift guard on the next run."""
        d = _model_dir(tmp_path)
        before = sorted(p.name for p in d.iterdir())
        upload(
            model_dir=d,
            manifest_path=_manifest(tmp_path, d),
            hf_token="",
            hf_username="me",
            repo_suffix="forge-bf16",
            github_repo_url="https://example.invalid/repo",
            dry_run=True,
        )
        assert sorted(p.name for p in d.iterdir()) == before


class TestGuardOrdering:
    def test_a_missing_file_reports_itself_not_a_hash_mismatch(self, tmp_path):
        """Deleting a file trips BOTH guards: the directory no longer hashes to
        the manifest either. The specific error has to win, or the person
        reading it goes looking for a corrupted checkpoint instead of a file
        they forgot to copy."""
        d = _model_dir(tmp_path)
        manifest = _manifest(tmp_path, d)
        (d / "chat_template.jinja").unlink()

        with pytest.raises(RuntimeError, match="chat_template.jinja"):
            upload(
                model_dir=d,
                manifest_path=manifest,
                hf_token="",
                hf_username="me",
                repo_suffix="forge-bf16",
                github_repo_url="https://example.invalid/repo",
                dry_run=True,
            )
