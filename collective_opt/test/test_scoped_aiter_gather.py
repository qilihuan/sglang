"""8-GPU parity test for native AITER collectives."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment  # noqa: E402
from sglang.srt.distributed.parallel_state import (  # noqa: E402
    get_tensor_model_parallel_group,
    initialize_model_parallel,
)
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)


def _input(shape, rank, dtype):
    values = torch.arange(
        int(torch.tensor(shape).prod()),
        device="cuda",
        dtype=torch.float32,
    )
    return (values.reshape(shape) % 997 + rank * 1_000).to(dtype)


def _expected(shape, world_size, dtype):
    return torch.cat(
        [_input(shape, rank, dtype) for rank in range(world_size)],
        dim=0,
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError(f"expected 8 ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        attention_context_model_parallel_size=world_size,
    )
    group = get_tensor_model_parallel_group()

    warmup = torch.zeros(1, device="cuda")
    dist.all_reduce(warmup, group=group.device_group)
    torch.cuda.synchronize()

    if not group._has_aiter_custom_all_gather():
        raise RuntimeError("SGLang TP group has no AITER custom all-gather")

    calls = 0
    original_unreg = group.ca_comm.all_gather_unreg

    def tracked_unreg(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_unreg(*args, **kwargs)

    group.ca_comm.all_gather_unreg = tracked_unreg

    shape = (256, 128)
    inp = _input(shape, rank, torch.float16).contiguous()
    expected = _expected(shape, world_size, inp.dtype)
    output = torch.empty(
        (shape[0] * world_size, shape[1]),
        dtype=inp.dtype,
        device="cuda",
    )

    group.all_gather_into_tensor(output, inp)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if calls != 1:
        raise AssertionError(f"native custom AG calls={calls}, expected 1")

    reduced = group.all_reduce(torch.full((128,), rank + 1.0, device="cuda"))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        reduced,
        torch.full_like(reduced, sum(range(1, world_size + 1))),
        rtol=0,
        atol=0,
    )

    if rank == 0:
        print(
            json.dumps(
                {
                    "all_reduce": "native",
                    "parity": "exact",
                    "all_gather": "aiter",
                    "world_size": world_size,
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
