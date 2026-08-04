import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.kernels.ops.quantization.saturating_fp8_cast import saturating_fp8_cast
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu")


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
def test_saturating_fp8_cast_boundaries(input_dtype):
    fp8_max = torch.finfo(fp8_dtype).max
    input = torch.tensor(
        [
            -2 * fp8_max,
            -fp8_max - 1,
            -fp8_max,
            -1,
            0,
            1,
            fp8_max,
            fp8_max + 1,
            2 * fp8_max,
        ],
        device="cuda",
        dtype=input_dtype,
    )

    actual = saturating_fp8_cast(input, fp8_dtype=fp8_dtype)
    expected = input.clamp(-fp8_max, fp8_max).to(fp8_dtype)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    assert torch.isfinite(actual.float()).all()


@pytest.mark.parametrize("shape", [(1,), (1, 64, 576), (8, 64, 576)])
def test_saturating_fp8_cast_shapes_and_output_reuse(shape):
    torch.manual_seed(0)
    fp8_max = torch.finfo(fp8_dtype).max
    input = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * fp8_max
    output = torch.empty(shape, device="cuda", dtype=fp8_dtype)

    actual = saturating_fp8_cast(input, output=output, fp8_dtype=fp8_dtype)
    expected = input.clamp(-fp8_max, fp8_max).to(fp8_dtype)

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
