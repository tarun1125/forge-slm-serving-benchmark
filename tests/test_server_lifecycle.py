import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from forge.phase2.arms import ArmConfig
from forge.phase2.server_lifecycle import ManagedServer, ServerStartupError


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _HealthyHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — stdlib method name
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002 — silence stdlib's default logging
        pass


class _RunningHealthServer:
    """A real local HTTP server answering 200 on any GET — used to test
    ManagedServer's polling logic against a genuine socket, not a mock."""

    def __init__(self):
        self.port = _free_port()
        self._server = HTTPServer(("127.0.0.1", self.port), _HealthyHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._thread.join(timeout=5)


class TestManagedServerNoLaunch:
    def test_start_and_stop_are_no_ops_when_launch_command_is_none(self):
        arm = ArmConfig(
            name="ollama",
            model_variant="q4",
            base_url="http://127.0.0.1:11434/v1",
            model_id="forge-qwen-coder-ft:q4",
            launch_command=None,
        )
        server = ManagedServer(arm, startup_timeout_s=1.0)
        server.start()  # must not attempt any health check or raise
        server.stop()


class TestManagedServerHealthCheck:
    def test_becomes_healthy_against_a_real_listening_server(self):
        health_server = _RunningHealthServer()
        try:
            arm = ArmConfig(
                name="fake_arm",
                model_variant="test",
                base_url=f"http://127.0.0.1:{health_server.port}",
                model_id="irrelevant",
                launch_command=[sys.executable, "-c", "import time; time.sleep(60)"],
                health_check_path="/v1/models",
            )
            server = ManagedServer(
                arm, startup_timeout_s=10.0, poll_interval_s=0.1, shutdown_timeout_s=5.0
            )
            server.start()  # should return once the health server answers 200
            server.stop()
        finally:
            health_server.stop()

    def test_raises_when_process_exits_before_becoming_healthy(self):
        # Points at a port nothing listens on — process exits almost
        # immediately, health check never has anything to poll.
        arm = ArmConfig(
            name="fake_arm",
            model_variant="test",
            base_url=f"http://127.0.0.1:{_free_port()}",
            model_id="irrelevant",
            launch_command=[sys.executable, "-c", "pass"],  # exits almost instantly
            health_check_path="/v1/models",
        )
        server = ManagedServer(
            arm, startup_timeout_s=5.0, poll_interval_s=0.05, shutdown_timeout_s=5.0
        )
        with pytest.raises(ServerStartupError, match="exited during startup"):
            server.start()

    def test_raises_on_timeout_when_nothing_ever_answers(self):
        arm = ArmConfig(
            name="fake_arm",
            model_variant="test",
            base_url=f"http://127.0.0.1:{_free_port()}",
            model_id="irrelevant",
            # Long-running process that never serves anything on the port above.
            launch_command=[sys.executable, "-c", "import time; time.sleep(60)"],
            health_check_path="/v1/models",
        )
        server = ManagedServer(
            arm, startup_timeout_s=0.3, poll_interval_s=0.05, shutdown_timeout_s=5.0
        )
        with pytest.raises(ServerStartupError, match="did not become healthy"):
            server.start()
