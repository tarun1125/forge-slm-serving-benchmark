"""Covers the one arm whose configuration can be wrong in ways a live
server would never catch: vllm_cuda points at a machine this process can't
see, so everything about it — the endpoint, the model id, and above all the
record of what hardware served the request — comes from Settings rather than
from anything observable at runtime. These tests are the whole safety net
for that, and they need no GPU, no network and no cloud account.
"""

from __future__ import annotations

import pytest

from forge.config import Settings
from forge.hardware import ServerHardware
from forge.phase2.arms import (
    ArmConfig,
    mlx_lm_arm,
    ollama_arm,
    ollama_cloud_arm,
    vllm_cuda_arm,
    vllm_metal_arm,
)
from forge.phase2.sweep import ARM_BUILDERS, build_arm_config


def _settings(**overrides) -> Settings:
    base = {
        "vllm_cuda_base_url": "http://127.0.0.1:8001/v1",
        "vllm_cuda_model_id": "forge-bf16",
        "vllm_cuda_api_key": "test-key",
        "vllm_cuda_gpu_name": "NVIDIA A100 80GB PCIe",
        "vllm_cuda_provider": "azure",
        "vllm_cuda_instance_type": "Standard_NC24ads_A100_v4",
        "vllm_cuda_region": "eastus",
        "vllm_cuda_gpu_memory_gb": 80.0,
        "vllm_cuda_gpu_memory_bandwidth_gb_s": 2039.0,
        "vllm_cuda_hourly_usd": 3.673,
        "ollama_cloud_base_url": "http://127.0.0.1:11435/v1",
        "ollama_cloud_cpu_name": "AWS Graviton4",
        "ollama_cloud_provider": "aws",
        "ollama_cloud_instance_type": "c8g.2xlarge",
        "ollama_cloud_region": "ap-south-1",
        "ollama_cloud_memory_gb": 16.0,
        "ollama_cloud_hourly_usd": 0.2159,
    }
    base.update(overrides)
    return Settings(**base)


class TestVllmCudaArm:
    def test_builds_a_remote_arm_from_settings(self):
        arm = vllm_cuda_arm("bf16", settings=_settings())

        assert arm.name == "vllm_cuda"
        assert arm.model_variant == "bf16"
        assert arm.base_url == "http://127.0.0.1:8001/v1"
        assert arm.model_id == "forge-bf16"
        assert arm.api_key == "test-key"

    def test_launches_nothing(self):
        """The server is already running on a rented VM. A non-None
        launch_command here would make ManagedServer try to spawn a local
        process and then block on a health check against a tunnel."""
        assert vllm_cuda_arm("bf16", settings=_settings()).launch_command is None

    def test_records_the_serving_hardware_not_this_machine(self):
        arm = vllm_cuda_arm("bf16", settings=_settings())

        assert arm.server_hardware == ServerHardware(
            processor="NVIDIA A100 80GB PCIe",
            provider="azure",
            instance_type="Standard_NC24ads_A100_v4",
            region="eastus",
            processor_memory_gb=80.0,
            memory_bandwidth_gb_s=2039.0,
            hourly_usd=3.673,
        )

    def test_refuses_to_build_without_an_accelerator_name(self):
        """A remote row that can't say what ran it is the unlabelled number
        hardware.py's docstring exists to prevent — so this raises instead
        of recording None and discovering it after the GPU is deleted."""
        with pytest.raises(RuntimeError, match="VLLM_CUDA_GPU_NAME"):
            vllm_cuda_arm("bf16", settings=_settings(vllm_cuda_gpu_name=None))

    def test_refuses_to_build_without_a_base_url(self):
        with pytest.raises(RuntimeError, match="VLLM_CUDA_BASE_URL"):
            vllm_cuda_arm("bf16", settings=_settings(vllm_cuda_base_url=None))

    @pytest.mark.parametrize("variant", ["4bit", "8bit", "q4", "f16"])
    def test_rejects_every_variant_vllm_cuda_cannot_load(self, variant):
        """models/fused-4bit and fused-8bit are MLX affine-quantised. Failing
        here is the difference between a clear error on the laptop and a
        confusing one against a server that is billing by the second."""
        with pytest.raises(ValueError, match="only serves 'bf16'"):
            vllm_cuda_arm(variant, settings=_settings())

    def test_model_id_falls_back_when_unset(self):
        arm = vllm_cuda_arm("bf16", settings=_settings(vllm_cuda_model_id=None))
        assert arm.model_id == "forge-bf16"


class TestOllamaCloudArm:
    def test_builds_a_remote_arm_from_settings(self):
        arm = ollama_cloud_arm("q4", settings=_settings())

        assert arm.name == "ollama_cloud"
        assert arm.model_variant == "q4"
        assert arm.base_url == "http://127.0.0.1:11435/v1"
        assert arm.launch_command is None

    def test_defaults_to_the_local_arms_registered_tag(self):
        """Same daemon, same Modelfile, same tag — so the only difference
        between an `ollama` row and an `ollama_cloud` row is the hardware."""
        assert ollama_cloud_arm("q4", settings=_settings()).model_id == ollama_arm("q4").model_id

    def test_records_the_serving_cpu(self):
        arm = ollama_cloud_arm("q4", settings=_settings())

        assert arm.server_hardware == ServerHardware(
            processor="AWS Graviton4",
            provider="aws",
            instance_type="c8g.2xlarge",
            region="ap-south-1",
            processor_memory_gb=16.0,
            memory_bandwidth_gb_s=None,
            hourly_usd=0.2159,
        )

    def test_refuses_to_build_without_a_cpu_name(self):
        with pytest.raises(RuntimeError, match="OLLAMA_CLOUD_CPU_NAME"):
            ollama_cloud_arm("q4", settings=_settings(ollama_cloud_cpu_name=None))

    def test_refuses_to_build_without_a_base_url(self):
        with pytest.raises(RuntimeError, match="OLLAMA_CLOUD_BASE_URL"):
            ollama_cloud_arm("q4", settings=_settings(ollama_cloud_base_url=None))

    def test_rejects_an_unregistered_variant(self):
        with pytest.raises(ValueError, match="Unknown ollama_cloud variant"):
            ollama_cloud_arm("4bit", settings=_settings())

    @pytest.mark.parametrize("variant", ["f16", "q8", "q4"])
    def test_accepts_every_registered_variant(self, variant):
        """Unlike vllm_cuda's hard bf16-only rule, q8 and f16 are servable on
        a cloud CPU — they just aren't published yet, which is a sweep-map
        decision (sweep.arm_variant_map) rather than a builder one."""
        assert ollama_cloud_arm(variant, settings=_settings()).model_variant == variant


class TestCloudArmsDoNotCollideWithLocalOnes:
    """sweep.py writes f"{arm}_{variant}_c{n}_{bucket}.jsonl" and
    score_accuracy.py groups by (arm, variant, bucket). If a cloud arm reused
    a local arm's name, the cloud run would overwrite the local result files
    and the two populations would be scored as one."""

    def test_arm_names_are_distinct(self):
        local = {mlx_lm_arm("bf16").name, ollama_arm("q4").name, vllm_metal_arm("bf16").name}
        remote = {
            vllm_cuda_arm("bf16", settings=_settings()).name,
            ollama_cloud_arm("q4", settings=_settings()).name,
        }
        assert local.isdisjoint(remote)

    def test_result_filenames_differ_for_the_same_variant(self):
        local = ollama_arm("q4")
        cloud = ollama_cloud_arm("q4", settings=_settings())
        assert local.model_variant == cloud.model_variant
        assert f"{local.name}_{local.model_variant}" != f"{cloud.name}_{cloud.model_variant}"


class TestLocalArmsCarryNoServerHardware:
    """None here is a claim, not a gap: for these arms the machine that
    measured the request is the machine that served it, and `hardware`
    already describes it."""

    @pytest.mark.parametrize(
        "arm",
        [
            mlx_lm_arm("bf16"),
            ollama_arm("q4"),
            vllm_metal_arm("bf16"),
        ],
        ids=["mlx_lm", "ollama", "vllm_metal"],
    )
    def test_server_hardware_is_none(self, arm: ArmConfig):
        assert arm.server_hardware is None


class TestSweepWiring:
    @pytest.mark.parametrize("arm", ["vllm_cuda", "ollama_cloud"])
    def test_cloud_arms_are_registered(self, arm):
        assert arm in ARM_BUILDERS

    def test_build_arm_config_dispatches_to_it(self, monkeypatch):
        monkeypatch.setenv("VLLM_CUDA_BASE_URL", "http://127.0.0.1:9999/v1")
        monkeypatch.setenv("VLLM_CUDA_GPU_NAME", "Tesla T4")

        arm = build_arm_config("vllm_cuda", "bf16")

        assert arm.name == "vllm_cuda"
        assert arm.base_url == "http://127.0.0.1:9999/v1"
        assert arm.server_hardware is not None
        assert arm.server_hardware.processor == "Tesla T4"

    def test_unknown_arm_still_raises(self):
        with pytest.raises(ValueError, match="Unknown arm"):
            build_arm_config("vllm_rocm", "bf16")


class TestOllamaCloudSweepWiring:
    def test_build_arm_config_dispatches_to_it(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "http://127.0.0.1:11435/v1")
        monkeypatch.setenv("OLLAMA_CLOUD_CPU_NAME", "AWS Graviton4")

        arm = build_arm_config("ollama_cloud", "q4")

        assert arm.name == "ollama_cloud"
        assert arm.server_hardware is not None
        assert arm.server_hardware.processor == "AWS Graviton4"
