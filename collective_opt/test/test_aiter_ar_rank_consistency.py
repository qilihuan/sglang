"""Compare rank consistency of AITER two-stage AR and RCCL at EAGLE shapes."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    initialize_model_parallel,
)
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)


def _rank_mismatch(outputs: list[torch.Tensor]) -> tuple[int, float]:
    reference = outputs[0]
    mismatch = sum(
        int(torch.count_nonzero(output != reference).item())
        for output in outputs[1:]
    )
    max_abs = max(
        float((output.float() - reference.float()).abs().max().item())
        for output in outputs[1:]
    )
    return mismatch, max_abs


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
    group = get_tensor_model_parallel_group()

    warmup = torch.zeros(1, device="cuda")
    dist.all_reduce(warmup, group=group.device_group)
    torch.cuda.synchronize()
    if group.ca_comm is None or group.ca_comm.disabled:
        raise RuntimeError("AITER custom all-reduce is unavailable")

    rows = []
    for batch_size in (40, 48, 56):
        torch.manual_seed(20260828 + rank)
        scale = 2.0 ** ((rank % 4) * 3)
        partial = (
            torch.randn(
                (batch_size, 6144),
                dtype=torch.float32,
                device="cuda",
            )
            * scale
        ).to(torch.bfloat16)

        aiter_output = group.ca_comm.custom_all_reduce(partial.clone())
        torch.cuda.synchronize()
        aiter_by_rank = [torch.empty_like(aiter_output) for _ in range(world_size)]
        dist.all_gather(
            aiter_by_rank,
            aiter_output,
            group=group.device_group,
        )
        aiter_mismatch, aiter_max_abs = _rank_mismatch(aiter_by_rank)

        rccl_output = partial.clone()
        dist.all_reduce(rccl_output, group=group.device_group)
        torch.cuda.synchronize()
        rccl_by_rank = [torch.empty_like(rccl_output) for _ in range(world_size)]
        dist.all_gather(
            rccl_by_rank,
            rccl_output,
            group=group.device_group,
        )
        rccl_mismatch, rccl_max_abs = _rank_mismatch(rccl_by_rank)

        torch.manual_seed(20260828)
        residual = torch.randn_like(partial)
        weight = torch.randn((partial.shape[-1],), dtype=partial.dtype, device="cuda")
        fused_output = group.fused_allreduce_rmsnorm(
            partial.clone(),
            residual,
            weight,
            1e-6,
        )
        if fused_output is None:
            raise RuntimeError("AITER fused all-reduce + RMSNorm is unavailable")
        fused_hidden, fused_residual = fused_output
        fused_hidden_by_rank = [
            torch.empty_like(fused_hidden) for _ in range(world_size)
        ]
        fused_residual_by_rank = [
            torch.empty_like(fused_residual) for _ in range(world_size)
        ]
        dist.all_gather(
            fused_hidden_by_rank,
            fused_hidden,
            group=group.device_group,
        )
        dist.all_gather(
            fused_residual_by_rank,
            fused_residual,
            group=group.device_group,
        )
        fused_hidden_mismatch, fused_hidden_max_abs = _rank_mismatch(
            fused_hidden_by_rank
        )
        fused_residual_mismatch, fused_residual_max_abs = _rank_mismatch(
            fused_residual_by_rank
        )

        rows.append(
            {
                "batch_size": batch_size,
                "aiter_max_abs": aiter_max_abs,
                "aiter_rank_mismatch": aiter_mismatch,
                "fused_hidden_max_abs": fused_hidden_max_abs,
                "fused_hidden_rank_mismatch": fused_hidden_mismatch,
                "fused_residual_max_abs": fused_residual_max_abs,
                "fused_residual_rank_mismatch": fused_residual_mismatch,
                "rccl_max_abs": rccl_max_abs,
                "rccl_rank_mismatch": rccl_mismatch,
            }
        )

    if rank == 0:
        print(json.dumps(rows, sort_keys=True))
    if any(row["rccl_rank_mismatch"] != 0 for row in rows):
        raise AssertionError(f"RCCL outputs differ across ranks: {rows}")
    if any(
        row["aiter_rank_mismatch"] != 0
        or row["fused_hidden_rank_mismatch"] != 0
        or row["fused_residual_rank_mismatch"] != 0
        for row in rows
    ):
        raise AssertionError(f"AITER outputs differ across ranks: {rows}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
