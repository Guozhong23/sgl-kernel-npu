"""Isolated capability contracts; no NPU runtime or attention execution.

Unlike the positive device-accuracy cases in operators/attention/test_flash_attn.py,
these tests load the real capability helpers with simulated wheel metadata and
backend modules. The local runtime fixture overrides the framework's NPU bootstrap.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest


@pytest.fixture(scope="session", autouse=True)
def npu_runtime():
    """Supply only the source path required by the existing package_factory."""
    return SimpleNamespace(
        source=Path(__file__).resolve().parents[4]
        / "python"
        / "sgl_kernel_npu"
        / "sgl_kernel_npu"
    )


@pytest.fixture
def capability(package_factory, npu_runtime, monkeypatch, tmp_path):
    # Any accidental runtime import fails instead of initializing an installed
    # package. package_factory loads generation.py without running parent init.
    for name in ("acl", "torch", "torch_npu", "sgl_kernel_npu"):
        monkeypatch.setitem(sys.modules, name, None)
    package = package_factory()
    generation = package.generation
    hardware = Mock(
        return_value=generation.Hardware(3510, generation.Generation.ARCH35)
    )
    monkeypatch.setattr(generation, "get_hardware", hardware)

    # Reuse the real leaf modules and registry, but never execute production
    # operators/__init__.py, utils/__init__.py or attention/__init__.py.
    for suffix in ("operators", "operators.utils", "operators.attention"):
        name = f"{package.namespace}.{suffix}"
        parent = ModuleType(name)
        parent.__package__ = name
        parent.__path__ = [str(npu_runtime.source.joinpath(*suffix.split(".")))]
        monkeypatch.setitem(sys.modules, name, parent)
    module = package.import_module("operators.attention.flash_attn")
    hardware.reset_mock()  # Registry initialization already consulted the mock.
    sdk = Mock(return_value="9.2.0")
    monkeypatch.setattr(module, "detect_cann_version", sdk)

    wheel = tmp_path / "sgl_kernel_npu"
    wheel.mkdir()
    monkeypatch.setattr(
        module, "__file__", str(wheel / "operators/attention/flash_attn.py")
    )
    dependency = {
        "name": "sgl_ops_transformer",
        "python_package": "sgl_ops_transformer",
        "arch_version": "3510",
        "cann_version": "9.2.0",
        "ops": ["flash_attn", "flash_attn_metadata"],
        "python_bindings": True,
        "binding": "upstream_jit",
    }
    manifest = {
        "schema_version": 1,
        "arch_version": "3510",
        "third_party": [dependency],
    }
    manifest_path = wheel / "_build_manifest.json"

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest))

    ops_name = "sgl_kernel_npu._third_party.sgl_ops_transformer.ops"
    binding_name = ops_name + ".attention.flash_attn.flash_attn"
    ops, extension = ModuleType(ops_name), ModuleType("simulated_dense_extension")
    computations = []
    for target in (ops, extension):
        for name in ("flash_attn", "flash_attn_metadata"):
            function = Mock(
                name=name,
                side_effect=AssertionError("Capability queries must not execute ops"),
            )
            setattr(target, name, function)
            computations.append(function)
    load = Mock(return_value=extension)
    binding = ModuleType(binding_name)
    binding.flash_attn_op_builder = SimpleNamespace(load=load)

    def resolve(name):
        if name == ops_name:
            return ops
        if name == binding_name:
            return binding
        raise AssertionError(f"Unexpected backend import or fallback: {name}")

    importer = Mock(side_effect=resolve)
    # Replace this leaf's importlib reference, not the global import machinery.
    monkeypatch.setattr(module, "importlib", SimpleNamespace(import_module=importer))
    yield SimpleNamespace(
        module=module,
        generation=generation,
        hardware=hardware,
        sdk=sdk,
        wheel=wheel,
        manifest=manifest,
        dependency=dependency,
        manifest_path=manifest_path,
        write_manifest=write_manifest,
        ops=ops,
        extension=extension,
        importer=importer,
        load=load,
        ops_name=ops_name,
        binding_name=binding_name,
    )
    for function in computations:
        function.assert_not_called()


@pytest.mark.parametrize(
    "absence",
    [
        "manifest",
        "main-profile",
        "flash_attn",
        "flash_attn_metadata",
        "vendor-only",
        "wheel-architecture",
        "profile-architecture",
        "profile-cann-version",
    ],
)
def test_optional_pair_absence_needs_no_sdk_or_backend(capability, absence):
    if absence == "main-profile":
        capability.manifest["third_party"] = [{"name": "sgl_ops_transformer_dsv4"}]
    elif absence in ("flash_attn", "flash_attn_metadata"):
        capability.dependency["ops"].remove(absence)
    elif absence == "vendor-only":
        capability.dependency["python_bindings"] = False
        capability.dependency["binding"] = "vendor_only"
    elif absence == "wheel-architecture":
        capability.manifest["arch_version"] = "2201"
    elif absence == "profile-architecture":
        capability.dependency["arch_version"] = "2201"
    elif absence == "profile-cann-version":
        capability.dependency["cann_version"] = "9.1.0"
    if absence != "manifest":
        capability.write_manifest()
    capability.sdk.side_effect = RuntimeError("SDK is deliberately not initialized")

    assert capability.module.supports_dense_flash_attn_sinks() is False

    capability.sdk.assert_not_called()
    capability.importer.assert_not_called()
    capability.load.assert_not_called()


@pytest.mark.parametrize("unsupported", ["architecture", "cann-version"])
def test_unsupported_runtime_does_not_load_backend(capability, unsupported):
    capability.write_manifest()
    if unsupported == "architecture":
        generation = capability.generation
        capability.hardware.return_value = generation.Hardware(
            2201, generation.Generation.ARCH22
        )
    else:
        capability.sdk.return_value = "9.1.0"

    assert capability.module.supports_dense_flash_attn_sinks() is False

    if unsupported == "architecture":
        capability.sdk.assert_not_called()
    else:
        capability.sdk.assert_called_once_with()
    capability.importer.assert_not_called()
    capability.load.assert_not_called()


@pytest.mark.parametrize("detector", ["hardware", "sdk"])
def test_detection_failure_is_not_optional_absence(capability, detector):
    capability.write_manifest()
    error = RuntimeError(f"{detector} detection failed")
    getattr(capability, detector).side_effect = error

    with pytest.raises(RuntimeError) as caught:
        capability.module.supports_dense_flash_attn_sinks()

    assert caught.value is error
    capability.importer.assert_not_called()
    capability.load.assert_not_called()


@pytest.mark.parametrize("damage", ["invalid-json", "bundle-without-manifest"])
def test_corrupt_packaging_is_not_optional_absence(capability, damage):
    if damage == "invalid-json":
        capability.manifest_path.write_text("{")
        expected_error = json.JSONDecodeError
    else:
        (capability.wheel / "_third_party/sgl_ops_transformer").mkdir(parents=True)
        expected_error = RuntimeError

    with pytest.raises(expected_error):
        capability.module.supports_dense_flash_attn_sinks()

    capability.sdk.assert_not_called()
    capability.importer.assert_not_called()
    capability.load.assert_not_called()


@pytest.mark.parametrize(
    "python_bindings,binding",
    [(True, "vendor_only"), (False, "upstream_jit"), (True, "unknown_binding")],
    ids=["python-marked-vendor-only", "vendor-marked-python", "unknown-binding"],
)
def test_invalid_binding_manifest_propagates(capability, python_bindings, binding):
    capability.dependency.update(python_bindings=python_bindings, binding=binding)
    capability.write_manifest()

    with pytest.raises(RuntimeError):
        capability.module.supports_dense_flash_attn_sinks()

    capability.sdk.assert_not_called()
    capability.importer.assert_not_called()
    capability.load.assert_not_called()


@pytest.mark.parametrize(
    "failure", ["missing-bundle", "missing-dependency", "import-error"]
)
def test_declared_bundle_import_errors_propagate(capability, failure):
    capability.write_manifest()
    if failure == "import-error":
        error = ImportError("undefined symbol in bundled extension")
    else:
        name = (
            capability.ops_name
            if failure == "missing-bundle"
            else "bundled_extension_dependency"
        )
        error = ModuleNotFoundError("declared backend cannot be imported", name=name)
    capability.importer.side_effect = error

    with pytest.raises(type(error)) as caught:
        capability.module.supports_dense_flash_attn_sinks()

    assert caught.value is error
    capability.importer.assert_called_once_with(capability.ops_name)
    capability.load.assert_not_called()


@pytest.mark.parametrize("target", ["ops", "extension"])
@pytest.mark.parametrize("export", ["flash_attn", "flash_attn_metadata"])
def test_declared_pair_requires_both_python_and_native_exports(
    capability, target, export
):
    capability.write_manifest()
    delattr(getattr(capability, target), export)

    with pytest.raises(AttributeError, match=export):
        capability.module.supports_dense_flash_attn_sinks()

    if target == "ops":
        capability.load.assert_not_called()
    else:
        capability.load.assert_called_once_with()


def test_native_loader_error_propagates_and_can_retry(capability):
    capability.write_manifest()
    error = RuntimeError("native bridge ABI mismatch")
    capability.load.side_effect = error

    with pytest.raises(RuntimeError) as caught:
        capability.module.supports_dense_flash_attn_sinks()
    assert caught.value is error
    capability.load.assert_called_once_with()

    capability.load.side_effect = None
    assert capability.module.supports_dense_flash_attn_sinks() is True
    assert capability.module.supports_dense_flash_attn_sinks() is True
    assert capability.load.call_count == 2  # Failure is retried; success is cached.


def test_success_resolves_both_exports_and_caches_native_loading(capability):
    capability.write_manifest()

    assert capability.module.supports_dense_flash_attn_sinks() is True
    assert capability.module.supports_dense_flash_attn_sinks() is True

    assert capability.importer.call_args_list == [
        call(capability.ops_name),
        call(capability.binding_name),
    ]
    capability.load.assert_called_once_with()
