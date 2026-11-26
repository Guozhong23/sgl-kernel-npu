"""Native ARCH35 dense FlashAttention with an independent CPU FP32 reference.

Run TestFullAttention and TestSlidingWindowAttention independently. The latter
is an executable specification of the proposed mode=4, left=127, right=0
mapping to 128 visible causal tokens; that mapping still needs device validation.
These are single-device local-head cases, not distributed tensor-parallel tests.
"""

import json
import math
from dataclasses import dataclass

import pytest
import torch

from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.standards import DEFAULT_STANDARDS

# Register the bundled CANN vendor path before conftest initializes the device.
pytest.importorskip("torch_npu")
import sgl_kernel_npu  # noqa: F401


HEAD_CASES = [
    pytest.param((16, 2), id="hq16-hkv2"),
    pytest.param((64, 8), id="hq64-hkv8"),
]
HEAD_DIM = 64
PAGE_SIZE = 64
SOFTMAX_SCALE = HEAD_DIM**-0.5
BF16_STANDARD = DEFAULT_STANDARDS["torch.bfloat16"]


@dataclass
class LogicalInputs:
    """BF16 source tensors retained on CPU before constructing the page pool."""

    q: torch.Tensor
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    sinks: torch.Tensor
    q_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]


def _logical_inputs(heads, q_lengths, kv_lengths, *, seed=42, zeros=False):
    q_heads, kv_heads = heads
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def values(shape):
        if zeros:
            return torch.zeros(shape, dtype=torch.bfloat16, device="cpu")
        return torch.randn(
            shape, generator=generator, dtype=torch.float32, device="cpu"
        ).to(torch.bfloat16)

    # Even head counts make every value nonzero; sinks differ across all heads.
    sinks = torch.linspace(-1.25, 1.25, q_heads, dtype=torch.float32, device="cpu")
    if zeros:
        sinks.zero_()
    return LogicalInputs(
        q=values((sum(q_lengths), q_heads, HEAD_DIM)),
        keys=[values((length, kv_heads, HEAD_DIM)) for length in kv_lengths],
        values=[values((length, kv_heads, HEAD_DIM)) for length in kv_lengths],
        sinks=sinks,
        q_lengths=tuple(q_lengths),
        kv_lengths=tuple(kv_lengths),
    )


def _pack_pages(logical):
    """Copy logical KV into shuffled, noncontiguous pages with poisoned gaps."""
    device = torch.get_default_device()
    generator = torch.Generator(device="cpu").manual_seed(20261125)
    pages_per_request = [math.ceil(n / PAGE_SIZE) for n in logical.kv_lengths]
    used_pages = sum(pages_per_request)
    total_pages = 2 * used_pages + 1
    # Only nonzero, even page IDs are used: adjacent logical pages are always
    # physically noncontiguous, regardless of the seeded permutation.
    permutation = 2 * (
        torch.randperm(used_pages, generator=generator, device="cpu") + 1
    )
    kv_heads = logical.keys[0].shape[1]
    pool_shape = (total_pages, PAGE_SIZE, kv_heads, HEAD_DIM)
    keys = torch.full(pool_shape, 37.0, dtype=torch.bfloat16, device="cpu")
    values = torch.full(pool_shape, -23.0, dtype=torch.bfloat16, device="cpu")
    table = torch.zeros(
        (len(logical.q_lengths), max(pages_per_request)),
        dtype=torch.int32,
        device="cpu",
    )
    page_offset = 0
    for request, num_pages in enumerate(pages_per_request):
        for logical_page in range(num_pages):
            physical_page = permutation[page_offset + logical_page].item()
            table[request, logical_page] = physical_page
            start = logical_page * PAGE_SIZE
            end = min(start + PAGE_SIZE, logical.kv_lengths[request])
            keys[physical_page, : end - start].copy_(logical.keys[request][start:end])
            values[physical_page, : end - start].copy_(
                logical.values[request][start:end]
            )
        page_offset += num_pages
    cu_q = torch.tensor(
        [0, *logical.q_lengths], dtype=torch.int32, device="cpu"
    ).cumsum(0, dtype=torch.int32)
    return dict(
        q=logical.q.to(device),
        k=keys.to(device),
        v=values.to(device),
        sinks=logical.sinks.to(device),
        block_table=table.to(device),
        cu_seqlens_q=cu_q.to(device),
        seqused_kv=torch.tensor(
            logical.kv_lengths, dtype=torch.int32, device="cpu"
        ).to(device),
    )


def _run_attention(logical, paged, *, sliding_window, explicit_lengths=False):
    from sgl_kernel_npu.operators.attention import flash_attn, flash_attn_metadata

    impl, _ = get_test_impls()
    require_test_impl(flash_attn_metadata, impl)
    require_test_impl(flash_attn, impl)
    shared = dict(
        cu_seqlens_q=paged["cu_seqlens_q"],
        cu_seqlens_kv=None,
        seqused_q=(
            torch.tensor(logical.q_lengths, dtype=torch.int32, device="cpu").to(
                paged["q"].device
            )
            if explicit_lengths
            else None
        ),
        seqused_kv=paged["seqused_kv"],
        max_seqlen_q=max(logical.q_lengths) if explicit_lengths else -1,
        max_seqlen_kv=max(logical.kv_lengths) if explicit_lengths else -1,
        mask_mode=4 if sliding_window else 3,
        win_left=127 if sliding_window else -1,
        win_right=0 if sliding_window else -1,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_out="TND",
    )
    metadata = flash_attn_metadata(
        logical.q.shape[1],
        logical.keys[0].shape[1],
        HEAD_DIM,
        head_dim_v=HEAD_DIM,
        batch_size=len(logical.q_lengths),
        is_grad_enabled=False,
        **shared,
        impl=impl,
    )
    assert metadata.dtype == torch.int32
    assert metadata.device == paged["q"].device
    assert metadata.ndim == 1 and metadata.numel() > 0

    # Pinned CANN contract: modes 3 and 4 both consume the same compressed INT8
    # upper-triangular template, not a per-request or precomputed SWA band mask.
    attn_mask = torch.triu(
        torch.ones((2048, 2048), dtype=torch.int8, device="cpu"), diagonal=1
    ).to(paged["q"].device)
    output, _ = flash_attn(
        paged["q"],
        paged["k"],
        paged["v"],
        block_table=paged["block_table"],
        sinks=paged["sinks"],
        attn_mask=attn_mask,
        metadata=metadata,
        softmax_scale=SOFTMAX_SCALE,
        return_softmax_lse=False,
        **shared,
        impl=impl,
    )
    assert output.device == paged["q"].device
    return output


def _cpu_reference(logical, *, sliding_window):
    """Evaluate softmax([QK * scale, sink])[:-1] @ V from logical BF16 KV.

    This oracle never sees a page table, packed cache, compressed mask, metadata,
    or operator implementation. Visibility is decided from absolute positions.
    """
    query = logical.q.float()
    result = torch.empty(query.shape, dtype=torch.float32, device="cpu")
    q_heads = query.shape[1]
    offset = 0
    for request, (q_length, kv_length) in enumerate(
        zip(logical.q_lengths, logical.kv_lengths)
    ):
        group_size = q_heads // logical.keys[request].shape[1]
        keys = logical.keys[request].float().repeat_interleave(group_size, dim=1)
        values = logical.values[request].float().repeat_interleave(group_size, dim=1)
        prefix_length = kv_length - q_length
        for local_position in range(q_length):
            absolute_position = prefix_length + local_position
            visible_positions = [
                position
                for position in range(kv_length)
                if position <= absolute_position
                and (not sliding_window or absolute_position - position < 128)
            ]
            indices = torch.tensor(
                visible_positions, dtype=torch.int64, device="cpu"
            )
            visible_keys = keys.index_select(0, indices)
            visible_values = values.index_select(0, indices)
            scores = torch.einsum(
                "hd,shd->hs", query[offset + local_position], visible_keys
            ) * SOFTMAX_SCALE
            scores_with_sink = torch.cat((scores, logical.sinks[:, None]), dim=-1)
            weights = torch.softmax(scores_with_sink, dim=-1)[:, :-1]
            result[offset + local_position] = torch.einsum(
                "hs,shd->hd", weights, visible_values
            )
        offset += q_length
    return result


def _output_and_reference(
    output, logical, record_property, *, sliding_window, label="output"
):
    assert output.shape == logical.q.shape
    assert output.dtype == torch.bfloat16
    actual = output.detach().to(device="cpu", dtype=torch.float32)
    expected = _cpu_reference(logical, sliding_window=sliding_window)
    actual_finite = torch.isfinite(actual).all().item()
    expected_finite = torch.isfinite(expected).all().item()
    metrics = dict(
        reference="cpu_fp32_from_bf16_qkv",
        actual_finite=actual_finite,
        reference_finite=expected_finite,
        max_abs=None,
        mean_abs=None,
        relative_l2=None,
    )
    if actual_finite and expected_finite:
        # FP64 diagnostics also remain representable for finite but corrupted
        # BF16 outputs. The oracle and correctness comparison stay FP32.
        difference = actual.double() - expected.double()
        absolute_error = difference.abs()
        # Normalize globally, rather than dividing each near-zero element.
        # These are diagnostics, not additional unvalidated pass thresholds.
        reference_norm = torch.linalg.vector_norm(expected.double()).item()
        metrics.update(
            max_abs=absolute_error.max().item(),
            mean_abs=absolute_error.mean().item(),
            relative_l2=(
                torch.linalg.vector_norm(difference).item()
                / max(reference_norm, torch.finfo(torch.float32).eps)
            ),
        )
    mode = "SWA" if sliding_window else "Full"
    serialized = json.dumps(metrics, sort_keys=True, allow_nan=False)
    # Record before asserting so both passing cases and numerical failures retain
    # diagnostics in captured output and, with --junitxml, testcase properties.
    record_property(f"dense_flash_attn_{mode}_{label}", serialized)
    print(f"dense_flash_attn {mode} {label}: {serialized}")
    assert actual_finite
    assert expected_finite
    return actual, expected


def _assert_output(output, logical, record_property, *, sliding_window):
    actual, expected = _output_and_reference(
        output, logical, record_property, sliding_window=sliding_window
    )
    torch.testing.assert_close(
        actual, expected, atol=BF16_STANDARD.atol, rtol=BF16_STANDARD.rtol
    )
    return actual


def _check_swapped_request_pages(heads, record_property, *, sliding_window):
    # Equal lengths keep both swapped rows valid, including the final page tail.
    # Opposite values make a request mix-up much larger than the BF16 tolerance.
    logical = _logical_inputs(heads, (1, 1), (129, 129), zeros=True)
    logical.values[0].fill_(1.0)
    logical.values[1].fill_(-1.0)
    paged = _pack_pages(logical)
    output = _run_attention(logical, paged, sliding_window=sliding_window)
    _assert_output(output, logical, record_property, sliding_window=sliding_window)

    swapped = {**paged, "block_table": paged["block_table"].flip(0).contiguous()}
    wrong_output = _run_attention(logical, swapped, sliding_window=sliding_window)
    wrong_actual, expected = _output_and_reference(
        wrong_output,
        logical,
        record_property,
        sliding_window=sliding_window,
        label="swapped_request_pages",
    )
    # Only a finite numerical mismatch counts as detecting the swapped mapping.
    # Kernel/metadata calls and shape/dtype/finite checks stay outside raises.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            wrong_actual, expected, atol=BF16_STANDARD.atol, rtol=BF16_STANDARD.rtol
        )


@pytest.mark.parametrize("heads", HEAD_CASES)
class TestFullAttention:
    @pytest.mark.parametrize(
        "q_lengths,kv_lengths",
        [
            pytest.param((33, 17), (33, 17), id="ordinary-prefill"),
            pytest.param((5, 3), (137, 66), id="prefix-short-chunk"),
            pytest.param((1, 1), (129, 257), id="decode"),
        ],
    )
    def test_random(self, heads, q_lengths, kv_lengths, record_property):
        logical = _logical_inputs(heads, q_lengths, kv_lengths)
        output = _run_attention(logical, _pack_pages(logical), sliding_window=False)
        _assert_output(output, logical, record_property, sliding_window=False)

    def test_ragged_prefix_crosses_pages(self, heads, record_property):
        # Prefixes [120, 63], cuQ [0, 8, 10], usedKV [128, 65]. The first KV ends
        # on a page edge; the second has a one-token tail on its second page.
        logical = _logical_inputs(heads, (8, 2), (128, 65), seed=123)
        # Exercise explicit usedQ=[8, 2] and exact maxQ/maxKV=8/128; the other
        # cases retain unspecified maxima and derive query lengths from cuQ.
        output = _run_attention(
            logical, _pack_pages(logical), sliding_window=False, explicit_lengths=True
        )
        _assert_output(output, logical, record_property, sliding_window=False)

    def test_single_visible_token_sink_weights(self, heads, record_property):
        logical = _logical_inputs(heads, (1,), (1,), zeros=True)
        logical.values[0].fill_(1.0)
        logical.sinks[0::2] = math.log(3.0)
        logical.sinks[1::2] = -math.log(3.0)
        output = _run_attention(logical, _pack_pages(logical), sliding_window=False)
        actual = _assert_output(output, logical, record_property, sliding_window=False)
        expected = torch.empty(logical.q.shape, dtype=torch.float32, device="cpu")
        expected[:, 0::2, :] = 0.25
        expected[:, 1::2, :] = 0.75
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)

    def test_multiple_queries_cannot_see_future(self, heads, record_property):
        logical = _logical_inputs(heads, (4,), (4,), zeros=True)
        # Only the final key has a nonzero value. Earlier queries must remain
        # exactly zero; the last has four visible keys plus one sink: 5 / 5 = 1.
        logical.values[0][3].fill_(5.0)
        output = _run_attention(logical, _pack_pages(logical), sliding_window=False)
        actual = _assert_output(output, logical, record_property, sliding_window=False)
        expected = torch.zeros(logical.q.shape, dtype=torch.float32, device="cpu")
        expected[3].fill_(1.0)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)

    def test_swapping_request_pages_is_detected(self, heads, record_property):
        _check_swapped_request_pages(heads, record_property, sliding_window=False)


@pytest.mark.parametrize("heads", HEAD_CASES)
class TestSlidingWindowAttention:
    @pytest.mark.parametrize("query_kind", ["prefill", "prefix-chunk", "decode"])
    def test_random_127_128_129_tokens(self, heads, query_kind, record_property):
        kv_lengths = (127, 128, 129)
        q_lengths = {
            "prefill": kv_lengths,
            "prefix-chunk": (8, 5, 2),
            "decode": (1, 1, 1),
        }[query_kind]
        logical = _logical_inputs(heads, q_lengths, kv_lengths, seed=2026)
        output = _run_attention(logical, _pack_pages(logical), sliding_window=True)
        _assert_output(output, logical, record_property, sliding_window=True)

    @pytest.mark.parametrize("query_kind", ["prefill", "prefix-chunk", "decode"])
    @pytest.mark.parametrize("nonzero_key", [0, 1], ids=["k0-excluded", "k1-kept"])
    def test_exact_128_token_boundary(
        self, heads, query_kind, nonzero_key, record_property
    ):
        # All three forms end at the same absolute query position p=128.
        # Prefix/chunk: prefix=120, Q=9; decode: prefix=128, Q=1.
        q_length = {"prefill": 129, "prefix-chunk": 9, "decode": 1}[query_kind]
        logical = _logical_inputs(heads, (q_length,), (129,), zeros=True)
        logical.values[0][nonzero_key, :, 0] = 1.0
        paged = _pack_pages(logical)

        # Reuse the same Q/K/V/sinks/page table for Full and SWA; each call builds
        # its own real public metadata with matching mode and window attributes.
        full = _run_attention(logical, paged, sliding_window=False)
        sliding = _run_attention(logical, paged, sliding_window=True)
        full_cpu = _assert_output(full, logical, record_property, sliding_window=False)
        sliding_cpu = _assert_output(
            sliding, logical, record_property, sliding_window=True
        )

        # Full: 129 keys + sink. SWA: keys 1..128 + sink. Keep atol below the
        # 1/129 vs 1/130 gap (~5.96e-5), while allowing BF16 rounding (<2e-6).
        expected_full = torch.zeros(
            (heads[0], HEAD_DIM), dtype=torch.float32, device="cpu"
        )
        expected_full[:, 0] = 1.0 / 130.0
        expected_sliding = torch.zeros(
            (heads[0], HEAD_DIM), dtype=torch.float32, device="cpu"
        )
        expected_sliding[:, 0] = 0.0 if nonzero_key == 0 else 1.0 / 129.0
        torch.testing.assert_close(full_cpu[-1], expected_full, atol=2e-5, rtol=0)
        torch.testing.assert_close(
            sliding_cpu[-1], expected_sliding, atol=2e-5, rtol=0
        )

    def test_swapping_request_pages_is_detected(self, heads, record_property):
        _check_swapped_request_pages(heads, record_property, sliding_window=True)
