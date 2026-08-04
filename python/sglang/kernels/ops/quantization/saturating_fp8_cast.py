# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _saturating_fp8_cast_kernel(
    input_ptr,
    output_ptr,
    numel,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    values = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    values = tl.clamp(values, FP8_MIN, FP8_MAX)
    # The FP8 output pointer makes the store perform the final conversion.
    tl.store(output_ptr + offsets, values, mask=mask)


def saturating_fp8_cast(
    input: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Cast a contiguous BF16/FP16 tensor to FP8 with finite-range saturation.

    The Triton HIP/CUDA kernel performs load, clamp, conversion, and store in a
    single launch, avoiding the BF16 intermediate created by
    ``input.clamp(...).to(fp8)``.
    """
    assert input.dtype in (torch.bfloat16, torch.float16)
    assert input.is_contiguous(), f"input must be contiguous, got {input.stride()}"
    assert fp8_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

    if output is None:
        output = torch.empty_like(input, dtype=fp8_dtype)
    else:
        assert output.shape == input.shape
        assert output.dtype == fp8_dtype
        assert output.is_contiguous()

    if input.numel() == 0:
        return output

    fp8_max = torch.finfo(output.dtype).max
    block_size = 1024
    _saturating_fp8_cast_kernel[(triton.cdiv(input.numel(), block_size),)](
        input,
        output,
        input.numel(),
        FP8_MIN=-fp8_max,
        FP8_MAX=fp8_max,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output
