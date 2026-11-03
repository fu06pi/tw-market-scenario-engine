from types import SimpleNamespace

import pytest

from tw_scenario.quant.numeric_runtime import (
    NumericRuntimeMismatch,
    capture_torch_cpu_runtime,
    configure_reproducible_torch_cpu,
    require_matching_torch_cpu_runtime,
)


class FakeTorch:
    __version__ = "2.test"

    def __init__(self, *, threads=4, interop=3, interop_locked=False):
        self._threads = threads
        self._interop = interop
        self._interop_locked = interop_locked
        self._deterministic = False
        self._warn_only = False
        self._matmul = "highest"
        self.version = SimpleNamespace(git_version="torch-git-test")
        self.__config__ = SimpleNamespace(show=lambda: "test build")
        self.backends = SimpleNamespace(
            cpu=SimpleNamespace(get_cpu_capability=lambda: "TEST_CPU"),
            mkldnn=SimpleNamespace(enabled=True, deterministic=False),
        )

    def get_num_threads(self):
        return self._threads

    def set_num_threads(self, value):
        self._threads = value

    def get_num_interop_threads(self):
        return self._interop

    def set_num_interop_threads(self, value):
        if self._interop_locked:
            raise RuntimeError("cannot set interop threads after work starts")
        self._interop = value

    def use_deterministic_algorithms(self, value):
        self._deterministic = value

    def are_deterministic_algorithms_enabled(self):
        return self._deterministic

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self._warn_only

    def get_float32_matmul_precision(self):
        return self._matmul


def test_configure_and_capture_effective_numeric_runtime():
    torch = FakeTorch()
    runtime = configure_reproducible_torch_cpu(torch)
    assert runtime["schema_version"] == "torch-cpu-numeric-runtime-v1"
    assert runtime["intraop_threads"] == 1
    assert runtime["interop_threads"] == 1
    assert runtime["deterministic_algorithms"] is True
    assert runtime["deterministic_warn_only"] is False
    assert runtime["mkldnn_deterministic"] is True
    assert runtime["torch_version"] == "2.test"
    assert runtime["torch_git_version"] == "torch-git-test"
    assert runtime["cpu_capability"] == "TEST_CPU"


def test_configuration_fails_if_incompatible_interop_pool_already_started():
    torch = FakeTorch(interop=4, interop_locked=True)
    with pytest.raises(NumericRuntimeMismatch, match="inter-op pool"):
        configure_reproducible_torch_cpu(torch)


def test_configuration_accepts_locked_pool_when_it_is_already_one():
    torch = FakeTorch(interop=1, interop_locked=True)
    runtime = configure_reproducible_torch_cpu(torch)
    assert runtime["interop_threads"] == 1


def test_runtime_match_is_exact_and_reports_changed_thread_count():
    torch = FakeTorch(threads=1, interop=1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    expected = capture_torch_cpu_runtime(torch)
    require_matching_torch_cpu_runtime(expected, torch)

    torch.set_num_threads(2)
    with pytest.raises(NumericRuntimeMismatch, match="intraop_threads"):
        require_matching_torch_cpu_runtime(expected, torch)


def test_runtime_match_rejects_legacy_or_incomplete_metadata():
    torch = FakeTorch(threads=1, interop=1)
    with pytest.raises(NumericRuntimeMismatch, match="lacks a supported"):
        require_matching_torch_cpu_runtime({}, torch)

    expected = capture_torch_cpu_runtime(torch)
    expected.pop("torch_build_sha256")
    with pytest.raises(NumericRuntimeMismatch, match="schema fields differ"):
        require_matching_torch_cpu_runtime(expected, torch)
