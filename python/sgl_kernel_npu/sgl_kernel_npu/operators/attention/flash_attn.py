"""Dense CANN FlashAttn and metadata from the wheel's arch35/9.2.0 profile."""

import importlib
import json
from functools import lru_cache
from pathlib import Path

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation, get_hardware
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_BUNDLE = "sgl_kernel_npu._third_party.sgl_ops_transformer"
_EXPORTS = ("flash_attn", "flash_attn_metadata")
_BACKEND = "cann_ops_transformers_cann920"


def _pair_is_packaged():
    """Consult the installed wheel, not registry declarations or external packages."""
    package = Path(__file__).resolve().parents[2]
    manifest_path = package / "_build_manifest.json"
    if not manifest_path.is_file():
        if (package / "_third_party/sgl_ops_transformer").exists():
            raise RuntimeError("Bundled ops-transformer has no build manifest")
        return False

    # Invalid JSON/structure must not be reported as an absent optional feature.
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise RuntimeError("Unsupported SGLang kernel build manifest schema")
    if manifest["arch_version"] != "3510":
        return False
    dependencies = manifest["third_party"]
    if not isinstance(dependencies, list):
        raise TypeError("Build manifest third_party must be a list")
    for dependency in dependencies:
        if dependency["name"] != "sgl_ops_transformer":
            continue
        if dependency["python_package"] != "sgl_ops_transformer":
            raise RuntimeError("Unexpected bundled ops-transformer package name")
        if (
            dependency["arch_version"] != "3510"
            or dependency["cann_version"] != "9.2.0"
        ):
            return False
        if not isinstance(dependency["ops"], (list, dict)):
            raise TypeError("Build manifest ops must be a list or mapping")
        if not set(_EXPORTS).issubset(dependency["ops"]):
            return False
        if not isinstance(dependency["python_bindings"], bool):
            raise TypeError("Build manifest python_bindings must be boolean")
        if not dependency["python_bindings"] or dependency["binding"] == "vendor_only":
            return False
        if dependency["binding"] != "upstream_jit":
            raise RuntimeError("Unexpected ops-transformer binding type")
        return True
    return False


@lru_cache(maxsize=1)
def _load_pair():
    if not _pair_is_packaged():
        raise RuntimeError(
            "This wheel does not bundle dense flash_attn and flash_attn_metadata "
            "for arch35/CANN 9.2.0; rebuild with the ops_transformer profile"
        )
    # Once the manifest declares the pair, missing modules/exports, load errors
    # and ABI failures are broken installations. Intentionally do not catch them.
    ops = importlib.import_module(_BUNDLE + ".ops")
    functions = tuple(getattr(ops, name) for name in _EXPORTS)
    for name, function in zip(_EXPORTS, functions):
        if not callable(function):
            raise TypeError(f"Bundled {name} is not callable")

    # In the pinned source both exports share this builder. Importing ops only
    # registers schemas; load the existing C++ bridge too, without calling an
    # attention/metadata kernel. The upstream loader may JIT-build it on first
    # use. This avoids reporting True for a merely registered, broken binding.
    binding = importlib.import_module(_BUNDLE + ".ops.attention.flash_attn.flash_attn")
    extension = binding.flash_attn_op_builder.load()
    for name in _EXPORTS:
        if not callable(getattr(extension, name)):
            raise TypeError(f"Bundled extension entry {name} is not callable")
    return functions


def supports_dense_flash_attn_sinks():
    """Whether this wheel can load the native dense sink-attention pair.

    False means unsupported architecture/CANN profile or optional bindings not
    packaged. Hardware/SDK detection, manifest, import and C++ loading errors
    propagate. Only the wheel's private ops-transformer package is considered.

    No NPU tensor is allocated and no NPU kernel is launched by this query. The
    upstream C++ bridge loader may compile on its first use. Success establishes
    Python/native bridge loadability. ACLNN symbols are resolved on execution;
    this query does not validate them, device kernels, numerical accuracy or
    any particular shape or SWA window. Execution errors also propagate.
    """
    if get_hardware().generation != Generation.ARCH35:
        return False
    if detect_cann_version() != "9.2.0" or not _pair_is_packaged():
        return False
    _load_pair()
    return True


def _require_cann_version():
    version = detect_cann_version()
    if version != "9.2.0":
        raise RuntimeError(
            f"Dense FlashAttn requires CANN 9.2.0; detected CANN {version}"
        )


@register_impl("attention.flash_attn", "auto", supported_generations={Generation.ARCH35})
@register_impl(
    "attention.flash_attn", _BACKEND, supported_generations={Generation.ARCH35}
)
def _cann_impl(*args, **kwargs):
    _require_cann_version()
    return _load_pair()[0](*args, **kwargs)


@register_impl(
    "attention.flash_attn_metadata", "auto", supported_generations={Generation.ARCH35}
)
@register_impl(
    "attention.flash_attn_metadata", _BACKEND, supported_generations={Generation.ARCH35}
)
def _cann_metadata(*args, **kwargs):
    _require_cann_version()
    return _load_pair()[1](*args, **kwargs)


@register_operator("attention.flash_attn")
def flash_attn(
    q,
    k,
    v,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False,
    *,
    impl="auto",
):
    """Forward the pinned CANN dense FlashAttn schema without a fallback.

    Both auto and cann_ops_transformers_cann920 require arch35 and CANN 9.2.0.
    Parameters/defaults match the bundled e7da18105f08b06e01ef0fff78f439098d483746
    binding; inputs are passed unchanged. This is not sparse MLA or quantized FA.

    For BF16 paged GQA: q is TND [T, Hq, D], k/v are PA_BBND
    [pages, page_size, Hkv, D], block_table is INT32 [B, max_pages], and sinks
    are FP32 [Hq] logits added to the softmax denominator (with no value row).
    cu_seqlens_q starts with zero; seqused_kv includes the prefix, and paged KV
    uses cu_seqlens_kv=None. Supply the actual flash_attn_metadata result and
    matching lengths, maximum lengths, layouts and masking attributes.

    mask_mode=3 is right-aligned causal, with win_left=win_right=-1. Mode 4
    takes sliding-window bounds. The tests propose left=127/right=0 for an
    inclusive 128-token GPT-OSS window; this mapping awaits device validation.
    Both modes require the pinned operator's compressed INT8 mask.

    Returns (output, softmax_lse). Output uses q.dtype and layout_out; TND is
    [T, Hq, Dv]. LSE is FP32 [Hq, T] for TND when enabled, otherwise shape [0].
    """
    return get_impl_by_name("attention.flash_attn", impl)(
        q,
        k,
        v,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        attn_mask=attn_mask,
        metadata=metadata,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=return_softmax_lse,
    )


@register_operator("attention.flash_attn_metadata")
def flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    head_dim_v=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    mask_mode=None,
    win_left=None,
    win_right=None,
    layout_q=None,
    layout_kv=None,
    layout_out=None,
    is_grad_enabled=False,
    impl="auto",
):
    """Build opaque native scheduling metadata; no attention computation.

    Defaults follow the exported Torch schema, including optional None scalar
    attributes. Callers should explicitly pass lengths, head_dim_v, layouts,
    maxima, mask_mode and windows matching flash_attn; the upstream pair cannot
    validate consistency between calls. For inference use is_grad_enabled=False.
    cu_seqlens_q/kv and seqused_q/kv are optional INT32 length tensors; paged KV
    uses seqused_kv and cu_seqlens_kv=None. batch_size is the request count.

    Returns a one-dimensional INT32 NPU tensor with upstream-defined capacity;
    callers must not hard-code its size. auto and cann_ops_transformers_cann920
    both use only the wheel's bundled arch35/CANN 9.2.0 implementation.
    """
    return get_impl_by_name("attention.flash_attn_metadata", impl)(
        num_heads_q,
        num_heads_kv,
        head_dim,
        head_dim_v=head_dim_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        is_grad_enabled=is_grad_enabled,
    )
