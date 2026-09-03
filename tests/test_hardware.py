from forge.hardware import assert_native_arm64, get_hardware_dict


def test_get_hardware_dict_shape():
    info = get_hardware_dict()
    assert set(info) == {
        "machine",
        "macos_version",
        "chip",
        "cpu_core_count",
        "gpu_core_count",
        "unified_memory_gb",
    }
    assert isinstance(info["cpu_core_count"], int)


def test_assert_native_arm64_passes_on_ci_and_dev_hardware():
    # Both the M5 Pro dev machine and the macos-14 GitHub Actions runner are
    # arm64 — this should never raise in either environment.
    assert_native_arm64()
