"""Starts and stops a local arm's server subprocess (mlx_lm.server, vLLM)
around a block of sweep work, so only one heavy model server is resident in
unified memory at a time — this Mac has 24GB total, and three serving
stacks each holding their own copy of a model would not fit. Ollama and
hosted-API arms have launch_command=None (see arms.py) and are no-ops here:
Ollama's daemon is started once, outside the sweep, and hosted APIs need no
local process at all.
"""

from __future__ import annotations

import subprocess
import time
from types import TracebackType

import httpx

from forge.logging_config import get_logger
from forge.phase2.arms import ArmConfig

log = get_logger(__name__)

DEFAULT_STARTUP_TIMEOUT_S = 120.0
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 15.0


class ServerStartupError(RuntimeError):
    pass


class ManagedServer:
    """Context manager: `with ManagedServer(arm_config): ...` starts the
    arm's server (if it has one), blocks until its health-check endpoint
    responds, yields, then terminates it on exit. A no-launch arm (Ollama,
    hosted API) is a no-op start/stop — the sweep can use this uniformly
    across all arms without special-casing which ones actually own a
    process."""

    def __init__(
        self,
        arm_config: ArmConfig,
        startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        shutdown_timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S,
    ):
        self.arm_config = arm_config
        self.startup_timeout_s = startup_timeout_s
        self.poll_interval_s = poll_interval_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.arm_config.launch_command is None:
            log.info(
                "server_lifecycle.no_launch_needed",
                arm=self.arm_config.name,
                model_variant=self.arm_config.model_variant,
            )
            return

        log.info(
            "server_lifecycle.start",
            arm=self.arm_config.name,
            model_variant=self.arm_config.model_variant,
            command=self.arm_config.launch_command,
        )
        self._process = subprocess.Popen(
            self.arm_config.launch_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_health()

    def _wait_for_health(self) -> None:
        health_url = self.arm_config.base_url + self.arm_config.health_check_path
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise ServerStartupError(
                    f"{self.arm_config.name} server process exited during startup "
                    f"(exit code {self._process.returncode}) before becoming healthy"
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
        raise ServerStartupError(
            f"{self.arm_config.name} server did not become healthy at {health_url} "
            f"within {self.startup_timeout_s}s. Last error: {last_error}"
        )

    def stop(self) -> None:
        if self._process is None:
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

    def __enter__(self) -> ManagedServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
