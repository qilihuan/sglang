"""Scoped workaround for AITER PyHIP two-stage external-FP8 inputs.

PyHIP expects physical-transposed per-1x128 scales in both stages. AITER's
BF16-input path accidentally satisfies that contract by reusing a wrapped
quantizer for a1 and a2, while the already-FP8 path skips the wrap and emits a
canonical a2 scale. This module fixes only the affected eager external call.
"""

from __future__ import annotations

import functools
import importlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch


_FORCE_TRANSPOSED_A2_QUANT: ContextVar[bool] = ContextVar(
    "collective_opt_pyhip_transposed_a2_quant",
    default=False,
)
_PROXY_MARKER = "_collective_opt_pyhip_quant_proxy"


class PhysicalScaleWorkspace:
    """One fixed-capacity destination for global a1 scale transpose."""

    def __init__(self, *, capacity_rows: int = 32768):
        if capacity_rows <= 0:
            raise ValueError("physical-scale capacity_rows must be positive")
        self.owner: tuple[Any, ...] | None = None
        self.output: torch.Tensor | None = None
        self.num_rows: torch.Tensor | None = None
        self.capacity_rows = capacity_rows
        self.allocation_count = 0

    def transpose(
        self,
        canonical_scale: torch.Tensor,
        *,
        partial_transpose,
        owner_tag: Any = None,
    ) -> torch.Tensor:
        if canonical_scale.ndim != 2:
            raise TypeError("canonical a1 scale must be rank-2")
        if canonical_scale.dtype != torch.float32:
            raise TypeError("canonical a1 scale must be FP32")
        if not canonical_scale.is_contiguous():
            raise ValueError("canonical a1 scale must be contiguous")

        rows, width = canonical_scale.shape
        stream_handle = int(
            torch.cuda.current_stream(canonical_scale.device).cuda_stream
        )
        owner = (
            canonical_scale.device.type,
            canonical_scale.device.index,
            stream_handle,
            canonical_scale.dtype,
            width,
            owner_tag,
        )
        if self.owner is not None and self.owner != owner:
            raise RuntimeError("physical-scale workspace execution lane changed")
        if rows > self.capacity_rows:
            raise RuntimeError(
                "physical-scale fixed capacity exceeded: "
                f"{rows} > {self.capacity_rows} rows"
            )

        if self.output is None:
            self.output = torch.empty(
                (self.capacity_rows, width),
                dtype=canonical_scale.dtype,
                device=canonical_scale.device,
            )
            self.num_rows = torch.empty(
                (1,),
                dtype=torch.int32,
                device=canonical_scale.device,
            )
            self.owner = owner
            self.allocation_count += 1

        self.num_rows.fill_(rows)
        output = self.output.narrow(0, 0, rows)
        partial_transpose(
            output,
            canonical_scale,
            num_rows=self.num_rows,
        )
        return output


def _code_markers(function: Any) -> set[str]:
    code = getattr(function, "__code__", None)
    if code is None:
        return set()
    return set(code.co_names) | set(code.co_freevars)


def is_verified_pyhip_two_stage(metadata: Any) -> bool:
    """Recognize the exact Python PyHIP two-stage metadata contract."""

    if bool(getattr(metadata, "run_1stage", False)):
        return False
    stage1 = getattr(metadata, "stage1", None)
    stage2 = getattr(metadata, "stage2", None)
    if stage1 is None or stage2 is None:
        return False
    if not bool(getattr(stage1, "transpose_quant", False)):
        return False
    if not bool(getattr(stage2, "transpose_quant", False)):
        return False
    return (
        "moe_gemm_8wave_g1u1" in _code_markers(stage1)
        and "moe_gemm_8wave_down" in _code_markers(stage2)
    )


def install_pyhip_external_quant_workaround(
    fused_moe_module: Any | None = None,
) -> bool:
    """Install an idempotent ContextVar-gated ``get_quant`` proxy."""

    if fused_moe_module is None:
        fused_moe_module = importlib.import_module("aiter.fused_moe")
    current = fused_moe_module.get_quant
    if bool(getattr(current, _PROXY_MARKER, False)):
        return True
    original = current
    per_1x128 = fused_moe_module.QuantType.per_1x128

    @functools.wraps(original)
    def scoped_get_quant(quant_type):
        quantizer = original(quant_type)
        if (
            _FORCE_TRANSPOSED_A2_QUANT.get()
            and quant_type == per_1x128
        ):
            return functools.partial(
                quantizer,
                transpose_scale=True,
            )
        return quantizer

    setattr(scoped_get_quant, _PROXY_MARKER, True)
    scoped_get_quant._collective_opt_original = original
    fused_moe_module.get_quant = scoped_get_quant
    return True


def pyhip_external_quant_workaround_active() -> bool:
    return _FORCE_TRANSPOSED_A2_QUANT.get()


@contextmanager
def pyhip_external_quant_context() -> Iterator[None]:
    """Transpose only PyHIP's inter-stage a2 scale for this eager call."""

    install_pyhip_external_quant_workaround()
    token = _FORCE_TRANSPOSED_A2_QUANT.set(True)
    try:
        yield
    finally:
        _FORCE_TRANSPOSED_A2_QUANT.reset(token)


def transpose_canonical_a1_scale(
    canonical_scale: torch.Tensor,
    *,
    partial_transpose,
) -> torch.Tensor:
    """Convert global canonical a1 scales to PyHIP's physical layout."""

    if canonical_scale.ndim != 2 or canonical_scale.dtype != torch.float32:
        raise TypeError("canonical a1 scale must be rank-2 FP32")
    if not canonical_scale.is_contiguous():
        raise ValueError("canonical a1 scale must be contiguous")
    num_rows = torch.tensor(
        [canonical_scale.shape[0]],
        dtype=torch.int32,
        device=canonical_scale.device,
    )
    physical_scale = torch.empty_like(canonical_scale)
    partial_transpose(
        physical_scale,
        canonical_scale,
        num_rows=num_rows,
    )
    return physical_scale
