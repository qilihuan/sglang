"""Pointer-stability tests for fixed-capacity Collective Opt workspaces."""

from __future__ import annotations

import torch

from opt.aiter_pyhip import (
    PhysicalScaleWorkspace,
    transpose_canonical_a1_scale,
)
from opt.runtime_patch import (
    _A1_SCALE_WORKSPACE,
    _FIXED_GLOBAL_ROWS,
    _LOCAL_QUANT_WORKSPACE,
    _NATIVE_GATHER_WORKSPACE,
    _LocalQuantWorkspace,
    _NativeGatherWorkspace,
)


def _gather_inputs(rows: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((rows, 16), dtype=torch.uint8, device="cuda"),
        torch.empty((rows, 4), dtype=torch.float32, device="cuda"),
        torch.empty((rows, 3), dtype=torch.int32, device="cuda"),
        torch.empty((rows, 3), dtype=torch.float32, device="cuda"),
    )


def main() -> None:
    torch.cuda.set_device(0)
    if _FIXED_GLOBAL_ROWS != 32768:
        raise AssertionError(f"unexpected production capacity: {_FIXED_GLOBAL_ROWS}")
    if (
        _NATIVE_GATHER_WORKSPACE.capacity_rows != _FIXED_GLOBAL_ROWS
        or _A1_SCALE_WORKSPACE.capacity_rows != _FIXED_GLOBAL_ROWS
        or _LOCAL_QUANT_WORKSPACE.global_capacity_rows != _FIXED_GLOBAL_ROWS
    ):
        raise AssertionError("production workspaces do not use fixed 32K capacity")

    gather = _NativeGatherWorkspace(capacity_rows=32)
    first = gather.reserve(
        *_gather_inputs(2),
        world_size=4,
        group_name="test",
    )
    gather_ptrs = tuple(
        tensor.data_ptr()
        for tensor in (gather.fp8, gather.scale, gather.ids, gather.weights)
    )
    second = gather.reserve(
        *_gather_inputs(8),
        world_size=4,
        group_name="test",
    )
    if tuple(tensor.shape[0] for tensor in first) != (8, 8, 8, 8):
        raise AssertionError("first gather views have incorrect row counts")
    if tuple(tensor.shape[0] for tensor in second) != (32, 32, 32, 32):
        raise AssertionError("second gather views have incorrect row counts")
    if gather.allocation_count != 1 or gather.capacity_rows != 32:
        raise AssertionError("gather workspace was not allocated exactly once")
    if gather_ptrs != tuple(
        tensor.data_ptr()
        for tensor in (gather.fp8, gather.scale, gather.ids, gather.weights)
    ):
        raise AssertionError("gather workspace pointers changed")
    try:
        gather.reserve(
            *_gather_inputs(9),
            world_size=4,
            group_name="test",
        )
    except RuntimeError as error:
        if "fixed capacity exceeded" not in str(error):
            raise
    else:
        raise AssertionError("oversized gather did not fail")

    scale = PhysicalScaleWorkspace(capacity_rows=32)

    def copy_transpose(output, canonical, *, num_rows):
        if int(num_rows.item()) != canonical.shape[0]:
            raise AssertionError("num_rows was not updated")
        output.copy_(canonical)

    canonical_small = torch.randn((8, 4), dtype=torch.float32, device="cuda")
    physical_small = scale.transpose(
        canonical_small,
        partial_transpose=copy_transpose,
    )
    scale_output_ptr = scale.output.data_ptr()
    scale_rows_ptr = scale.num_rows.data_ptr()
    torch.testing.assert_close(physical_small, canonical_small, rtol=0, atol=0)

    canonical_full = torch.randn((32, 4), dtype=torch.float32, device="cuda")
    physical_full = scale.transpose(
        canonical_full,
        partial_transpose=copy_transpose,
    )
    torch.testing.assert_close(physical_full, canonical_full, rtol=0, atol=0)
    if (
        scale.allocation_count != 1
        or scale.output.data_ptr() != scale_output_ptr
        or scale.num_rows.data_ptr() != scale_rows_ptr
    ):
        raise AssertionError("physical-scale workspace pointers changed")
    try:
        scale.transpose(
            torch.empty((33, 4), dtype=torch.float32, device="cuda"),
            partial_transpose=copy_transpose,
        )
    except RuntimeError as error:
        if "fixed capacity exceeded" not in str(error):
            raise
    else:
        raise AssertionError("oversized physical scale did not fail")

    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.quant import dynamic_per_token_scaled_quant

    local_quant = _LocalQuantWorkspace()
    quantize_reference = aiter.get_hip_quant(QuantType.per_1x128)
    quant_ptrs = None
    for local_rows in (1024, 4096):
        hidden_states = torch.randn(
            (local_rows, 6144),
            dtype=torch.bfloat16,
            device="cuda",
        )
        expected_fp8, expected_scale = quantize_reference(
            hidden_states,
            quant_dtype=dtypes.fp8,
            transpose_scale=False,
        )
        actual_fp8, actual_scale = local_quant.quantize(
            hidden_states,
            world_size=8,
            quant_dtype=dtypes.fp8,
            quant_op=dynamic_per_token_scaled_quant,
        )
        torch.testing.assert_close(actual_fp8, expected_fp8, rtol=0, atol=0)
        torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
        current_ptrs = (
            local_quant.fp8.data_ptr(),
            local_quant.scale.data_ptr(),
        )
        if quant_ptrs is None:
            quant_ptrs = current_ptrs
        elif current_ptrs != quant_ptrs:
            raise AssertionError("local quant workspace pointers changed")
    if (
        local_quant.allocation_count != 1
        or local_quant.local_capacity_rows != 4096
    ):
        raise AssertionError("local quant workspace was not allocated once at 4K")

    actual_scale = PhysicalScaleWorkspace()
    actual_output_ptr = None
    for rows in (8192, 32768):
        canonical = torch.rand(
            (rows, 48),
            dtype=torch.float32,
            device="cuda",
        )
        expected = transpose_canonical_a1_scale(
            canonical,
            partial_transpose=aiter.partial_transpose,
        )
        actual = actual_scale.transpose(
            canonical,
            partial_transpose=aiter.partial_transpose,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if actual_output_ptr is None:
            actual_output_ptr = actual_scale.output.data_ptr()
        elif actual_scale.output.data_ptr() != actual_output_ptr:
            raise AssertionError("real a1 transpose workspace pointer changed")
    if actual_scale.allocation_count != 1:
        raise AssertionError("real a1 transpose workspace was reallocated")

    print(
        {
            "a1_transpose_parity": "exact",
            "capacity_rows": _FIXED_GLOBAL_ROWS,
            "gather_allocations": gather.allocation_count,
            "local_quant_allocations": local_quant.allocation_count,
            "local_quant_parity": "exact",
            "scale_allocations": scale.allocation_count,
            "pointer_stable": True,
        }
    )


if __name__ == "__main__":
    main()
