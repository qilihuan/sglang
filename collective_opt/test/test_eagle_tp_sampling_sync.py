"""8-GPU smoke test for PR35079 eager and capture-mode TP broadcasts."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group, init_distributed_environment
from sglang.srt.distributed.parallel_state import initialize_model_parallel
from sglang.srt.model_executor.runner import model_capture_mode
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.eagle_worker_v2 import (
    _sync_draft_sampling_across_tp,
)


def _rank_tensors(rank: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((8, 1), rank, dtype=torch.int64, device="cuda"),
        torch.full((8, 1), rank + 0.25, dtype=torch.float32, device="cuda"),
        torch.full((8, 32), rank + 0.5, dtype=torch.bfloat16, device="cuda"),
    )


def _assert_rank0_values(tensors: tuple[torch.Tensor, ...]) -> None:
    expected = (0, 0.25, 0.5)
    for tensor, value in zip(tensors, expected, strict=True):
        torch.testing.assert_close(
            tensor,
            torch.full_like(tensor, value),
            rtol=0,
            atol=0,
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
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    group = get_tp_group()

    eager_tensors = _rank_tensors(rank)
    _sync_draft_sampling_across_tp(*eager_tensors)
    torch.cuda.synchronize()
    _assert_rank0_values(eager_tensors)

    pynccl = group.pynccl_comm
    if pynccl is None:
        raise RuntimeError("capture-mode PyNccl communicator is unavailable")
    capture_calls = 0
    original_broadcast = pynccl.broadcast

    def tracked_broadcast(*args, **kwargs):
        nonlocal capture_calls
        capture_calls += 1
        return original_broadcast(*args, **kwargs)

    pynccl.broadcast = tracked_broadcast
    capture_tensors = _rank_tensors(rank)
    with pynccl.change_state(enable=True), model_capture_mode():
        _sync_draft_sampling_across_tp(*capture_tensors)
    torch.cuda.synchronize()
    _assert_rank0_values(capture_tensors)
    if capture_calls != len(capture_tensors):
        raise AssertionError(
            f"capture-mode PyNccl calls={capture_calls}, "
            f"expected={len(capture_tensors)}"
        )

    if rank == 0:
        print(
            json.dumps(
                {
                    "capture_mode": "pynccl",
                    "capture_tensors": capture_calls,
                    "eager_mode": "torch_distributed",
                    "parity": "exact",
                    "world_size": world_size,
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
