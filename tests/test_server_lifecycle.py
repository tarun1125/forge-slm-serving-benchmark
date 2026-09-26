import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from forge.hardware import ServerHardware, ServerSoftware
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
    def test_becomes_healthy_against_a_real_listening_server(self, tmp_path):
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
                arm,
                startup_timeout_s=10.0,
                poll_interval_s=0.1,
                shutdown_timeout_s=5.0,
                log_dir=tmp_path,
            )
            server.start()  # should return once the health server answers 200
            server.stop()
        finally:
            health_server.stop()

    def test_raises_when_process_exits_before_becoming_healthy(self, tmp_path):
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
            arm,
            startup_timeout_s=5.0,
            poll_interval_s=0.05,
            shutdown_timeout_s=5.0,
            log_dir=tmp_path,
        )
        with pytest.raises(ServerStartupError, match="exited during startup"):
            server.start()

    def test_raises_on_timeout_when_nothing_ever_answers(self, tmp_path):
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
            arm,
            startup_timeout_s=0.3,
            poll_interval_s=0.05,
            shutdown_timeout_s=5.0,
            log_dir=tmp_path,
        )
        with pytest.raises(ServerStartupError, match="did not become healthy"):
            server.start()


def _models_handler(
    model_ids: list[str], version: str | None = None
) -> type[BaseHTTPRequestHandler]:
    models = json.dumps({"object": "list", "data": [{"id": m} for m in model_ids]}).encode()

    class _ModelsHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — stdlib method name
            if self.path == "/api/version":
                if version is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps({"version": version}).encode()
            else:
                body = models
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            pass

    return _ModelsHandler


def _remote_arm(port: int, model_id: str = "forge-qwen-coder-ft:q4") -> ArmConfig:
    return ArmConfig(
        name="ollama_cloud",
        model_variant="q4",
        base_url=f"http://127.0.0.1:{port}/v1",
        model_id=model_id,
        launch_command=None,
        server_hardware=ServerHardware(processor="Neoverse-N2", provider="azure"),
        server_software=ServerSoftware(stack="ollama", settings={"OLLAMA_NUM_PARALLEL": "4"}),
        version_path="/api/version",
    )


class TestManagedServerRemotePreflight:
    """A remote arm launches nothing, so without a preflight a dead tunnel
    produces a completed sweep of failed rows — see docs/cloud-arm.md."""

    def _serve(self, model_ids: list[str], version: str | None = None) -> tuple[HTTPServer, int]:
        port = _free_port()
        server = HTTPServer(("127.0.0.1", port), _models_handler(model_ids, version))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, port

    def test_passes_when_the_endpoint_serves_the_model(self):
        server, port = self._serve(["forge-qwen-coder-ft:q4"])
        try:
            ManagedServer(_remote_arm(port)).start()
        finally:
            server.shutdown()

    def test_refuses_a_dead_endpoint(self):
        with pytest.raises(ServerStartupError, match="not answering"):
            ManagedServer(_remote_arm(_free_port())).start()

    def test_refuses_an_endpoint_missing_the_model(self):
        server, port = self._serve(["qwen2.5-coder:1.5b"])
        try:
            with pytest.raises(ServerStartupError, match="does not serve"):
                ManagedServer(_remote_arm(port)).start()
        finally:
            server.shutdown()


class TestServerSoftwareVersion:
    def _serve(self, version: str | None) -> tuple[HTTPServer, int]:
        port = _free_port()
        server = HTTPServer(
            ("127.0.0.1", port), _models_handler(["forge-qwen-coder-ft:q4"], version)
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, port

    def test_detected_version_is_stamped_onto_the_arm_config(self):
        server, port = self._serve("0.32.14")
        try:
            managed = ManagedServer(_remote_arm(port))
            managed.start()
        finally:
            server.shutdown()
        software = managed.arm_config.server_software
        assert software is not None
        assert software.version == "0.32.14"
        assert software.settings == {"OLLAMA_NUM_PARALLEL": "4"}  # kept, not replaced

    def test_an_undetectable_version_is_a_warning_not_a_failure(self):
        server, port = self._serve(None)
        try:
            managed = ManagedServer(_remote_arm(port))
            managed.start()  # must not raise
        finally:
            server.shutdown()
        assert managed.arm_config.server_software is not None
        assert managed.arm_config.server_software.version is None


class TestServerLog:
    """A launched server's output used to go to DEVNULL, so a startup failure
    reported an exit code and threw the reason away."""

    def test_startup_failure_quotes_the_servers_own_error(self, tmp_path):
        arm = ArmConfig(
            name="vllm_metal",
            model_variant="bf16",
            base_url=f"http://127.0.0.1:{_free_port()}",
            model_id="irrelevant",
            launch_command=[
                sys.executable,
                "-c",
                "import sys; print('ValueError: unsupported quantization mode affine', "
                "file=sys.stderr); sys.exit(1)",
            ],
        )
        server = ManagedServer(arm, startup_timeout_s=5.0, poll_interval_s=0.05, log_dir=tmp_path)
        with pytest.raises(ServerStartupError, match="unsupported quantization mode affine"):
            server.start()
        assert server.log_path is not None and server.log_path.exists()

    def test_log_is_kept_after_a_successful_run(self, tmp_path):
        # The child serves its own health endpoint only AFTER printing, as a
        # real server does — so healthy implies the line was written.
        port = _free_port()
        child = (
            "print('INFO: KV cache blocks: 1234', flush=True)\n"
            "import http.server as h\n"
            f"h.HTTPServer(('127.0.0.1', {port}), h.SimpleHTTPRequestHandler).serve_forever()"
        )
        arm = ArmConfig(
            name="fake_arm",
            model_variant="test",
            base_url=f"http://127.0.0.1:{port}",
            model_id="irrelevant",
            launch_command=[sys.executable, "-c", child],
            health_check_path="/",
        )
        with ManagedServer(arm, poll_interval_s=0.05, log_dir=tmp_path) as server:
            pass
        assert server.log_path is not None
        assert "KV cache blocks: 1234" in server.log_path.read_text()

    def test_a_missing_executable_is_a_startup_error_not_an_oserror(self, tmp_path):
        arm = ArmConfig(
            name="vllm_metal",
            model_variant="bf16",
            base_url="http://127.0.0.1:1",
            model_id="irrelevant",
            launch_command=[str(tmp_path / "no-such-venv" / "bin" / "vllm"), "serve"],
        )
        with pytest.raises(ServerStartupError, match="could not launch 'vllm'"):
            ManagedServer(arm, log_dir=tmp_path).start()
