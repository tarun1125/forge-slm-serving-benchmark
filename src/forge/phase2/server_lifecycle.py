"""Starts and stops a local arm's server subprocess (mlx_lm.server, vLLM)
around a block of sweep work, so only one heavy model server is resident in
unified memory at a time — this Mac has 24GB total, and three serving
stacks each holding their own copy of a model would not fit. Ollama and
hosted-API arms have launch_command=None (see arms.py) and are no-ops here:
Ollama's daemon is started once, outside the sweep, and hosted APIs need no
local process at all.

A launched server's stdout+stderr go to a per-launch file under log_dir
(default logs/servers/, gitignored), not to DEVNULL. An earlier version
discarded them, so a vLLM that died at startup — wrong venv, model config it
can't load, port in use — surfaced only as "exited with code 1" with the
reason thrown away. The file is kept after a successful run too, because what
vLLM prints at startup (resolved dtype, KV-cache blocks, max concurrency) is
the record of how the arm was actually configured. On a startup failure the
tail of that file is put into the exception itself.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import httpx

from forge.logging_config import get_logger
from forge.phase2.arms import ArmConfig

log = get_logger(__name__)

DEFAULT_STARTUP_TIMEOUT_S = 120.0
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 15.0
DEFAULT_LOG_DIR = Path("logs/servers")
STARTUP_ERROR_TAIL_LINES = 30


class ServerStartupError(RuntimeError):
    pass


def _tail(path: Path, n_lines: int = STARTUP_ERROR_TAIL_LINES) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(server log unreadable)"
    return "\n".join(lines[-n_lines:]) or "(server wrote nothing)"


class ManagedServer:
    """Context manager: `with ManagedServer(arm_config) as server: ...`
    starts the arm's server (if it has one), blocks until its health-check
    endpoint responds, yields, then terminates it on exit. A no-launch arm
    (Ollama, hosted API) is a no-op start/stop — the sweep can use this
    uniformly across all arms without special-casing which ones actually own
    a process.

    Use `server.arm_config` inside the block, not the one passed in: once the
    server is reachable its version is detected and stamped onto
    arm_config.server_software (see _detect_version)."""

    def __init__(
        self,
        arm_config: ArmConfig,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        shutdown_timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S,
        log_dir: Path = DEFAULT_LOG_DIR,
    ):
        self.arm_config = arm_config
        self.startup_timeout_s = startup_timeout_s
        self.poll_interval_s = poll_interval_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self.log_dir = log_dir
        self.log_path: Path | None = None
        self._log_file: IO[bytes] | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.arm_config.launch_command is None:
            log.info(
                "server_lifecycle.no_launch_needed",
                arm=self.arm_config.name,
                model_variant=self.arm_config.model_variant,
            )
            if self.arm_config.server_hardware is not None:
                self._preflight_remote()
            self._detect_version()
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = (
            self.log_dir / f"{self.arm_config.name}_{self.arm_config.model_variant}_{stamp}.log"
        )
        log.info(
            "server_lifecycle.start",
            arm=self.arm_config.name,
            model_variant=self.arm_config.model_variant,
            # Basename only for the executable: vllm_metal's is an absolute
            # path under the home directory, which AGENTS.md says not to log.
            command=[
                Path(self.arm_config.launch_command[0]).name,
                *self.arm_config.launch_command[1:],
            ],
            server_log=str(self.log_path),
        )
        self._log_file = self.log_path.open("wb")
        try:
            self._process = subprocess.Popen(
                self.arm_config.launch_command,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            # e.g. the vllm-metal venv doesn't exist: Popen raises before any
            # process exists, so there is no log to tail — the OSError is the reason.
            self._close_log()
            raise ServerStartupError(
                f"{self.arm_config.name}: could not launch "
                f"{Path(self.arm_config.launch_command[0]).name!r}: {exc}"
            ) from exc
        self._wait_for_health()
        self._detect_version()

    def _preflight_remote(self) -> None:
        """One-shot check for arms served from a rented machine. Without it a
        dead SSH tunnel, or a VM where the model was never registered under
        model_id, produces a *completed* sweep of `Connection error.` / 404
        rows (see docs/cloud-arm.md, layer 3) — indistinguishable from a real
        run until someone reads n_failed, by which point the VM may be gone.
        Checks both that the endpoint answers and that it serves model_id."""
        url = self.arm_config.base_url + self.arm_config.health_check_path
        try:
            response = httpx.get(url, headers=self._auth_headers(), timeout=10.0)
            response.raise_for_status()
            served = {m.get("id") for m in response.json().get("data", [])}
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise ServerStartupError(
                f"{self.arm_config.name}: remote endpoint {url} is not answering ({exc}). "
                "Is the SSH tunnel up and the server running on the VM?"
            ) from exc
        if self.arm_config.model_id not in served:
            raise ServerStartupError(
                f"{self.arm_config.name}: {url} does not serve {self.arm_config.model_id!r} "
                f"(it lists {sorted(str(s) for s in served)}). Register the model on the VM "
                "or set the *_MODEL_ID setting to the name it is served under."
            )
        log.info("server_lifecycle.remote_preflight_ok", arm=self.arm_config.name)

    def _detect_version(self) -> None:
        """Best-effort: stamps the live server's reported version onto
        arm_config.server_software. Asked of the server rather than typed into
        .env because the version that matters is the one actually running —
        on a VM provisioned from an install script, nobody chose it. A failure
        is a warning, not an error: a missing version weakens the record but
        doesn't invalidate a single measurement."""
        software = self.arm_config.server_software
        path = self.arm_config.version_path
        if software is None or path is None or software.version is not None:
            return
        root = self.arm_config.base_url.removesuffix("/").removesuffix("/v1")
        try:
            response = httpx.get(root + path, headers=self._auth_headers(), timeout=5.0)
            response.raise_for_status()
            version = str(response.json()["version"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            log.warning(
                "server_lifecycle.version_unknown",
                arm=self.arm_config.name,
                url=root + path,
                error=str(exc),
            )
            return
        self.arm_config = replace(
            self.arm_config, server_software=replace(software, version=version)
        )
        log.info(
            "server_lifecycle.server_software",
            arm=self.arm_config.name,
            stack=software.stack,
            version=version,
            settings=software.settings,
        )

    def _auth_headers(self) -> dict[str, str]:
        if not self.arm_config.api_key:
            return {}
        return {"Authorization": f"Bearer {self.arm_config.api_key}"}

    def _startup_failure(self, reason: str) -> ServerStartupError:
        tail = _tail(self.log_path) if self.log_path is not None else "(no server log)"
        log.error(
            "server_lifecycle.startup_failed",
            arm=self.arm_config.name,
            reason=reason,
            server_log=str(self.log_path),
        )
        return ServerStartupError(
            f"{reason}\n--- last {STARTUP_ERROR_TAIL_LINES} lines of {self.log_path} ---\n{tail}"
        )

    def _wait_for_health(self) -> None:
        health_url = self.arm_config.base_url + self.arm_config.health_check_path
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                returncode = self._process.returncode
                self.stop()
                raise self._startup_failure(
                    f"{self.arm_config.name} server process exited during startup "
                    f"(exit code {returncode}) before becoming healthy"
                )
            try:
                response = httpx.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    log.info(
                        "server_lifecycle.healthy",
                        arm=self.arm_config.name,
                        elapsed_s=round(self.startup_timeout_s - (deadline - time.monotonic()), 1),
                    )
                    return
            except httpx.HTTPError as exc:
                last_error = exc
            time.sleep(self.poll_interval_s)

        self.stop()
        raise self._startup_failure(
            f"{self.arm_config.name} server did not become healthy at {health_url} "
            f"within {self.startup_timeout_s}s. Last error: {last_error}"
        )

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def stop(self) -> None:
        if self._process is None:
            self._close_log()
            return
        log.info("server_lifecycle.stop", arm=self.arm_config.name)
        self._process.terminate()
        try:
            self._process.wait(timeout=self.shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("server_lifecycle.force_kill", arm=self.arm_config.name)
            self._process.kill()
            self._process.wait(timeout=self.shutdown_timeout_s)
        self._process = None
        self._close_log()

    def __enter__(self) -> ManagedServer:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
