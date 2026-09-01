"""Minimal opt-only bridge for SGLang's native FP8 CP gather path.

Production work stays in SGLang's communicator, router, StandardDispatcher,
AITER runner, and BF16 ReduceScatter. This module only skips the original BF16
gather, adds the missing FP8 branch to StandardDispatcher, and adapts PyHIP's
external-scale ABI.
"""

from __future__ import annotations

import builtins
import functools
import logging
import os
import sys
from dataclasses import replace
from typing import Any

import torch

from .aiter_pyhip import (
    PhysicalScaleWorkspace,
    install_pyhip_external_quant_workaround,
    is_verified_pyhip_two_stage,
    pyhip_external_quant_context,
)


LOGGER = logging.getLogger("collective_opt")
_TARGET_MODULE = "sglang.srt.models.deepseek_v2"
_LOCAL_MARKER = "_collective_opt_local_hidden"
_PREQUANT_MARKER = "_collective_opt_pyhip_prequantized"
_RUNNER_MARKER = "_collective_opt_pyhip_runner"
_PATCHED = False
_PATCHING = False
_IMPORT_HOOK_INSTALLED = False
_ORIGINAL_IMPORT = None
_LOGGED_SHAPES: set[int] = set()
_LOGGED_FALLBACKS: set[str] = set()
_FIXED_WORKSPACE_LOGGED = False
_LOCAL_METADATA_CONVERSIONS = 0
_FIXED_GLOBAL_ROWS = 32768


class _NativeGatherWorkspace:
    """Fixed 32K outputs; transport remains GroupCoordinator-owned."""

    def __init__(self, *, capacity_rows: int = _FIXED_GLOBAL_ROWS):
        if capacity_rows <= 0:
            raise ValueError("gather capacity_rows must be positive")
        self.owner = None
        self.capacity_rows = capacity_rows
        self.fp8 = None
        self.scale = None
        self.ids = None
        self.weights = None
        self.allocation_count = 0

    def reserve(
        self,
        local_fp8,
        local_scale,
        local_ids,
        local_weights,
        *,
        world_size,
        group_name,
    ):
        global_rows = local_fp8.shape[0] * world_size
        owner = (
            local_fp8.device,
            int(torch.cuda.current_stream().cuda_stream),
            local_fp8.dtype,
            local_fp8.shape[1],
            local_scale.shape[1],
            local_ids.shape[1],
            world_size,
            group_name,
        )
        if global_rows > self.capacity_rows:
            raise RuntimeError(
                "native gather fixed capacity exceeded: "
                f"{global_rows} > {self.capacity_rows} rows"
            )
        if self.owner is not None and self.owner != owner:
            raise RuntimeError("native gather workspace owner changed")
        if self.fp8 is None:
            device = local_fp8.device
            self.fp8 = torch.empty(
                (self.capacity_rows, local_fp8.shape[1]),
                dtype=local_fp8.dtype,
                device=device,
            )
            self.scale = torch.empty(
                (self.capacity_rows, local_scale.shape[1]),
                dtype=local_scale.dtype,
                device=device,
            )
            self.ids = torch.empty(
                (self.capacity_rows, local_ids.shape[1]),
                dtype=local_ids.dtype,
                device=device,
            )
            self.weights = torch.empty(
                (self.capacity_rows, local_weights.shape[1]),
                dtype=local_weights.dtype,
                device=device,
            )
            self.owner = owner
            self.allocation_count += 1
        return (
            self.fp8.narrow(0, 0, global_rows),
            self.scale.narrow(0, 0, global_rows),
            self.ids.narrow(0, 0, global_rows),
            self.weights.narrow(0, 0, global_rows),
        )


class _LocalQuantWorkspace:
    """Fixed local slice of the 32K global per-1x128 quant workspace."""

    def __init__(
        self,
        *,
        global_capacity_rows: int = _FIXED_GLOBAL_ROWS,
        group_size: int = 128,
    ):
        if global_capacity_rows <= 0 or group_size <= 0:
            raise ValueError("local quant workspace dimensions must be positive")
        self.global_capacity_rows = global_capacity_rows
        self.group_size = group_size
        self.local_capacity_rows = 0
        self.owner = None
        self.fp8 = None
        self.scale = None
        self.allocation_count = 0

    def quantize(
        self,
        hidden_states,
        *,
        world_size,
        quant_dtype,
        quant_op,
    ):
        if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
            raise ValueError("local quant input must be contiguous rank-2")
        rows, width = hidden_states.shape
        if width % self.group_size:
            raise ValueError("local quant width must divide the group size")
        if self.global_capacity_rows % world_size:
            raise ValueError("global quant capacity must divide world size")
        local_capacity_rows = self.global_capacity_rows // world_size
        if rows > local_capacity_rows:
            raise RuntimeError(
                "local quant fixed capacity exceeded: "
                f"{rows} > {local_capacity_rows} rows"
            )
        owner = (
            hidden_states.device,
            int(torch.cuda.current_stream().cuda_stream),
            hidden_states.dtype,
            quant_dtype,
            width,
            world_size,
        )
        if self.owner is not None and self.owner != owner:
            raise RuntimeError("local quant workspace owner changed")
        if self.fp8 is None:
            self.fp8 = torch.empty(
                (local_capacity_rows, width),
                dtype=quant_dtype,
                device=hidden_states.device,
            )
            self.scale = torch.empty(
                (local_capacity_rows, width // self.group_size),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            self.local_capacity_rows = local_capacity_rows
            self.owner = owner
            self.allocation_count += 1

        fp8 = self.fp8.narrow(0, 0, rows)
        scale = self.scale.narrow(0, 0, rows)
        quant_op(
            fp8,
            hidden_states.view(-1, self.group_size),
            scale,
            shuffle_scale=False,
        )
        return fp8, scale


_NATIVE_GATHER_WORKSPACE = _NativeGatherWorkspace()
_LOCAL_QUANT_WORKSPACE = _LocalQuantWorkspace()
_A1_SCALE_WORKSPACE = PhysicalScaleWorkspace(
    capacity_rows=_FIXED_GLOBAL_ROWS,
)


def _enabled() -> bool:
    return os.environ.get("SGLANG_ENABLE_QUANTIZED_CP_MOE_AG", "0") == "1"


def _same_group_ranks(left: Any, right: Any) -> bool:
    left_ranks = tuple(getattr(left, "ranks", ()))
    right_ranks = tuple(getattr(right, "ranks", ()))
    return bool(left_ranks) and left_ranks == right_ranks


def _fallback(reason: str, world_rank: int) -> None:
    if world_rank == 0 and reason not in _LOGGED_FALLBACKS:
        _LOGGED_FALLBACKS.add(reason)
        LOGGER.info("[collective-opt] native-path fallback: %s", reason)


def _verified_pyhip_plan(moe, global_rows: int) -> bool:
    from aiter import QuantType, dtypes
    from aiter.fused_moe import get_2stage_cfgs, get_inter_dim, get_padded_M
    from aiter.ops.flydsl.moe_common import GateMode
    from sglang.srt.layers.moe.moe_runner.aiter import (
        AiterQuantType,
        _aiter_activation,
    )

    experts = moe.experts
    quant_method = experts.quant_method
    runner = getattr(quant_method, "runner", None)
    if runner is None or not runner.runner_backend.is_aiter():
        return False
    if runner.lora_enabled:
        return False
    config = experts.moe_runner_config
    if config.no_combine or config.apply_router_weight_on_input:
        return False
    quant_info = quant_method.maybe_get_hip_aiter_quant_info(
        experts,
        config.no_combine,
    )
    if (
        quant_info is None
        or quant_info.quant_type != AiterQuantType.PER_128X128
        or quant_info.w13_weight.dtype != dtypes.fp8
    ):
        return False

    expert_count, model_dim, inter_dim = get_inter_dim(
        quant_info.w13_weight.shape,
        quant_info.w2_weight.shape,
    )
    metadata = get_2stage_cfgs(
        get_padded_M(global_rows),
        model_dim,
        inter_dim,
        expert_count,
        config.top_k,
        torch.bfloat16,
        dtypes.fp8,
        quant_info.w13_weight.dtype,
        QuantType.per_1x128,
        config.is_gated,
        _aiter_activation(config.activation),
        quant_info.doweight_stage1,
        quant_info.hidden_pad,
        quant_info.intermediate_pad,
        bool(
            getattr(quant_info.w13_weight, "is_shuffled", False)
            or getattr(quant_info.w2_weight, "is_shuffled", False)
        ),
        GateMode.SEPARATED.value,
    )
    return bool(
        model_dim == 6144
        and inter_dim == 256
        and expert_count == 257
        and config.top_k == 9
        and int(metadata.block_m) == 256
        and int(metadata.ksplit) == 0
        and is_verified_pyhip_two_stage(metadata)
    )


def _prepare_native_marker(
    decoder,
    hidden_states: torch.Tensor,
    forward_batch,
) -> dict[str, Any] | None:
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatcher,
    )
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE, get_server_args
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    server_args = get_server_args()
    moe = decoder.mlp
    cp_metadata = getattr(forward_batch, "attn_cp_metadata", None)
    per_rank_rows = tuple(
        int(value)
        for value in (
            getattr(cp_metadata, "per_rank_actual_token", None) or ()
        )
    )
    # DSA round-robin metadata intentionally has no per-rank row list. Its
    # splitter instead asserts global_rows % cp_size == 0 and reshapes into
    # equal local tensors, which is exactly the fixed-size AG/RS contract.
    round_robin_cp = is_dsa_prefill_cp_round_robin_split()
    reason = None
    if not isinstance(moe, DeepseekV2MoE):
        reason = "layer is not sparse MoE"
    elif hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        reason = "hidden input is not rank-2 BF16"
    elif hidden_states.shape[0] <= 0 or hidden_states.shape[1] != 6144:
        reason = "hidden shape is unsupported"
    elif torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        reason = "compiled/captured execution is unsupported"
    elif moe.is_nextn or moe.is_hash:
        reason = "nextn/hash MoE is unsupported"
    elif moe._enable_a2a_moe or moe._fuse_shared_experts_inside_sbo:
        reason = "A2A/SBO MoE is unsupported"
    elif int(moe.n_shared_experts or 0) != moe.num_fused_shared_experts:
        reason = "shared experts are not fused"
    elif hasattr(moe, "shared_experts"):
        reason = "separate shared expert is unsupported"
    elif server_args.enable_eplb or server_args.ep_num_redundant_experts:
        reason = "EPLB/redundant experts are unsupported"
    elif (
        server_args.expert_distribution_recorder_mode is not None
        or server_args.enable_return_routed_experts
    ):
        reason = "expert recording is unsupported"
    elif not isinstance(moe.experts.dispatcher, StandardDispatcher):
        reason = "dispatcher is not StandardDispatcher"
    elif getattr(moe.experts, "_dwdp_bound", False):
        reason = "DWDP-bound experts are unsupported"
    elif (
        parallel.attn_dp_size != 1
        or parallel.attn_tp_size != 1
        or parallel.moe_ep_size != 1
        or parallel.attn_cp_size != parallel.tp_size
        or parallel.moe_tp_size != parallel.tp_size
        or parallel.attn_cp_rank != parallel.tp_rank
    ):
        reason = "parallel topology is unsupported"
    elif not _same_group_ranks(parallel.attn_cp_group, parallel.tp_group):
        reason = "attention-CP and TP groups differ"
    elif not round_robin_cp and (
        len(per_rank_rows) != parallel.attn_cp_size
        or len(set(per_rank_rows)) != 1
    ):
        reason = "CP logical rows are not equal"
    elif not decoder.layer_communicator.should_use_reduce_scatter(forward_batch):
        reason = "native post-MoE ReduceScatter is inactive"

    global_rows = hidden_states.shape[0] * parallel.attn_cp_size
    minimum = int(
        os.environ.get(
            "SGLANG_QUANTIZED_CP_MOE_MIN_GLOBAL_ROWS",
            "16384",
        )
    )
    if reason is None and global_rows < minimum:
        reason = f"global rows {global_rows} below threshold {minimum}"
    if reason is None and global_rows > _FIXED_GLOBAL_ROWS:
        reason = (
            f"global rows {global_rows} exceed fixed workspace "
            f"capacity {_FIXED_GLOBAL_ROWS}"
        )
    if reason is None and not _verified_pyhip_plan(moe, global_rows):
        reason = "selected FMoE backend is not verified PyHIP two-stage"
    if reason is not None:
        _fallback(reason, parallel.world_rank)
        return None
    return {
        "global_rows": global_rows,
        "group": parallel.attn_cp_group,
    }


def _patch_standard_dispatcher() -> None:
    from aiter import dtypes
    from aiter.ops.quant import dynamic_per_token_scaled_quant
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatcher,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import (
        StandardTopKOutput,
        TopKOutputChecker,
    )
    from sglang.srt.runtime_context import get_parallel

    original = StandardDispatcher.dispatch
    if getattr(original, "_collective_opt_native_patched", False):
        return

    @functools.wraps(original)
    def dispatch(self, hidden_states, topk_output):
        global _LOCAL_METADATA_CONVERSIONS
        marker = getattr(hidden_states, _LOCAL_MARKER, None)
        if marker is None:
            return original(self, hidden_states, topk_output)
        try:
            delattr(hidden_states, _LOCAL_MARKER)
        except AttributeError:
            pass
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise RuntimeError("collective opt requires standard TopK output")

        group = marker["group"]
        local_fp8, local_scale = _LOCAL_QUANT_WORKSPACE.quantize(
            hidden_states,
            world_size=group.world_size,
            quant_dtype=dtypes.fp8,
            quant_op=dynamic_per_token_scaled_quant,
        )
        local_fp8_view = local_fp8.view(torch.float16)
        local_weights = topk_output.topk_weights
        if local_weights.dtype != torch.float32 or not local_weights.is_contiguous():
            local_weights = local_weights.to(torch.float32).contiguous()
            _LOCAL_METADATA_CONVERSIONS += 1
        local_ids = topk_output.topk_ids
        if local_ids.dtype != torch.int32 or not local_ids.is_contiguous():
            local_ids = local_ids.to(torch.int32).contiguous()
            _LOCAL_METADATA_CONVERSIONS += 1
        (
            gathered_fp8,
            gathered_scale,
            gathered_ids,
            gathered_weights,
        ) = _NATIVE_GATHER_WORKSPACE.reserve(
            local_fp8,
            local_scale,
            local_ids,
            local_weights,
            world_size=group.world_size,
            group_name=group.unique_name,
        )
        custom_ag = bool(
            group._has_aiter_custom_all_gather()
            and all(
                group.ca_comm.should_custom_ag(tensor)
                for tensor in (local_fp8_view, local_scale, local_weights)
            )
        )
        group.all_gather_into_tensor(
            gathered_fp8.view(torch.float16),
            local_fp8_view,
        )
        group.all_gather_into_tensor(
            gathered_scale,
            local_scale,
        )
        group.all_gather_into_tensor(
            gathered_weights,
            local_weights,
        )
        group.all_gather_into_tensor(
            gathered_ids,
            local_ids,
        )
        setattr(
            gathered_fp8,
            _PREQUANT_MARKER,
            {"global_rows": marker["global_rows"]},
        )
        if get_parallel().world_rank == 0:
            global_rows = marker["global_rows"]
            if global_rows not in _LOGGED_SHAPES:
                _LOGGED_SHAPES.add(global_rows)
                LOGGER.info(
                    "[collective-opt] native GroupCoordinator active "
                    "backend=pyhip_two_stage routing=local "
                    "transport=%s global_rows=%s "
                    "gather_ratio=0.521484375 gather_allocs=%s "
                    "capacity_rows=%s",
                    (
                        "aiter_custom_ag"
                        if custom_ag
                        else "group_coordinator_fallback"
                    ),
                    global_rows,
                    _NATIVE_GATHER_WORKSPACE.allocation_count,
                    _FIXED_GLOBAL_ROWS,
                )
        return StandardDispatchOutput(
            hidden_states=gathered_fp8,
            hidden_states_scale=gathered_scale,
            topk_output=StandardTopKOutput(
                topk_weights=gathered_weights,
                topk_ids=gathered_ids,
                router_logits=None,
            ),
        )

    dispatch._collective_opt_native_patched = True
    StandardDispatcher.dispatch = dispatch


def _patch_aiter_runner() -> bool:
    from sglang.srt.layers.moe.moe_runner import aiter as aiter_runner
    from sglang.srt.layers.moe.moe_runner.base import PermuteMethodPool

    key = ("standard", "aiter")
    original_permute = PermuteMethodPool._pre_permute_methods.get(key)
    if original_permute is None:
        return False
    if not getattr(
        original_permute,
        "_collective_opt_native_patched",
        False,
    ):

        @functools.wraps(original_permute)
        def pre_permute(
            dispatch_output,
            quant_info,
            runner_config,
            running_state,
        ):
            global _FIXED_WORKSPACE_LOGGED
            runner_input = original_permute(
                dispatch_output,
                quant_info,
                runner_config,
                running_state,
            )
            marker = getattr(
                dispatch_output.hidden_states,
                _PREQUANT_MARKER,
                None,
            )
            if marker is None:
                return runner_input
            if (
                dispatch_output.hidden_states_scale is None
                or dispatch_output.hidden_states.element_size() != 1
                or runner_config.apply_router_weight_on_input
            ):
                raise RuntimeError("invalid native prequantized dispatch")
            import aiter

            physical_scale = _A1_SCALE_WORKSPACE.transpose(
                dispatch_output.hidden_states_scale,
                partial_transpose=aiter.partial_transpose,
            )
            if not _FIXED_WORKSPACE_LOGGED:
                if (
                    not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0
                ):
                    LOGGER.info(
                        "[collective-opt] fixed workspaces ready "
                        "capacity_rows=%s local_capacity_rows=%s "
                        "quant_allocs=%s gather_allocs=%s scale_allocs=%s "
                        "metadata_conversions=%s",
                        _FIXED_GLOBAL_ROWS,
                        _LOCAL_QUANT_WORKSPACE.local_capacity_rows,
                        _LOCAL_QUANT_WORKSPACE.allocation_count,
                        _NATIVE_GATHER_WORKSPACE.allocation_count,
                        _A1_SCALE_WORKSPACE.allocation_count,
                        _LOCAL_METADATA_CONVERSIONS,
                    )
                _FIXED_WORKSPACE_LOGGED = True
            runner_input = replace(
                runner_input,
                a1_scale=physical_scale,
                num_local_tokens=None,
                output_dtype=torch.bfloat16,
            )
            setattr(runner_input, _RUNNER_MARKER, marker)
            return runner_input

        pre_permute._collective_opt_native_patched = True
        PermuteMethodPool._pre_permute_methods[key] = pre_permute

    original_run = aiter_runner.AiterRunnerCore.run
    if not getattr(original_run, "_collective_opt_native_patched", False):

        @functools.wraps(original_run)
        def run(self, runner_input, quant_info, running_state, hooks=None):
            if getattr(runner_input, _RUNNER_MARKER, None) is None:
                return original_run(
                    self,
                    runner_input,
                    quant_info,
                    running_state,
                    hooks=hooks,
                )
            with pyhip_external_quant_context():
                return original_run(
                    self,
                    runner_input,
                    quant_info,
                    running_state,
                    hooks=hooks,
                )

        run._collective_opt_native_patched = True
        aiter_runner.AiterRunnerCore.run = run
    install_pyhip_external_quant_workaround()
    return True


def _patch_decoder() -> None:
    import sglang.srt.models.deepseek_v2 as deepseek_v2
    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
    from sglang.srt.layers.communicator import ScatterMode
    from sglang.srt.layers.communicator_dsa_cp import (
        DSACPLayerCommunicator,
        dsa_cp_gather_hidden_states,
    )
    from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp

    decoder_cls = deepseek_v2.DeepseekV2DecoderLayer
    original_init = decoder_cls.__init__
    if getattr(original_init, "_collective_opt_native_patched", False):
        return

    @functools.wraps(original_init)
    def decoder_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _patch_aiter_runner()
        if (
            not self.is_layer_sparse
            or not isinstance(
                self.layer_communicator,
                DSACPLayerCommunicator,
            )
            or self.layer_scatter_modes.mlp_mode != ScatterMode.FULL
        ):
            return
        communicator = self.layer_communicator
        original_prepare = (
            communicator._communicate_with_all_reduce_and_layer_norm_fn
        )

        def prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            layernorm,
            context,
        ):
            use_cp = dsa_use_prefill_cp(
                forward_batch
            ) or mla_use_prefill_cp(forward_batch)
            if (
                not _enabled()
                or not use_cp
                or hidden_states.shape[0] <= 0
            ):
                return original_prepare(
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                    layernorm=layernorm,
                    context=context,
                )
            hidden_states, residual = layernorm(
                hidden_states,
                residual,
            )
            marker = _prepare_native_marker(
                self,
                hidden_states,
                forward_batch,
            )
            if marker is None:
                hidden_states = dsa_cp_gather_hidden_states(
                    hidden_states
                )
            else:
                setattr(hidden_states, _LOCAL_MARKER, marker)
            return hidden_states, residual

        communicator._communicate_with_all_reduce_and_layer_norm_fn = (
            prepare_mlp
        )

    decoder_init._collective_opt_native_patched = True
    decoder_cls.__init__ = decoder_init


def apply_patches() -> None:
    global _PATCHED, _PATCHING
    if _PATCHED or _PATCHING or not _enabled():
        return
    target = sys.modules.get(_TARGET_MODULE)
    if target is None or not hasattr(target, "DeepseekV2DecoderLayer"):
        return
    _PATCHING = True
    try:
        _patch_standard_dispatcher()
        _patch_decoder()
        _PATCHED = True
        LOGGER.info(
            "[collective-opt] minimal native patch installed "
            "transport=native-GroupCoordinator"
        )
    finally:
        _PATCHING = False


def install_import_hook() -> None:
    global _IMPORT_HOOK_INSTALLED, _ORIGINAL_IMPORT
    if _IMPORT_HOOK_INSTALLED or not _enabled():
        return
    if _TARGET_MODULE in sys.modules:
        apply_patches()
        return
    _ORIGINAL_IMPORT = builtins.__import__

    @functools.wraps(_ORIGINAL_IMPORT)
    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        if (
            not _PATCHED
            and not _PATCHING
            and _TARGET_MODULE in sys.modules
            and hasattr(
                sys.modules[_TARGET_MODULE],
                "DeepseekV2DecoderLayer",
            )
        ):
            apply_patches()
            if _PATCHED:
                builtins.__import__ = _ORIGINAL_IMPORT
        return module

    builtins.__import__ = importing
    _IMPORT_HOOK_INSTALLED = True
