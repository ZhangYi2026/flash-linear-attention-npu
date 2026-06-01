# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import torch
import sys
import os
from typing import Optional, Tuple, List

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from chunk_bwd_dqkwg_cpu import chunk_bwd_dqkwg_cpu


def create_gate_g(B: int, H: int, T: int, gtype):
    lo, hi = -5e-2, -5e-5
    span = hi - lo
    margin = max(span * 1e-7, 1e-12)
    g_t = torch.linspace(
        float(hi) - margin,
        float(lo) + margin,
        T,
        dtype=torch.float64,
    )
    return g_t.unsqueeze(0).unsqueeze(0).expand(B, H, T).contiguous().to(gtype)


def prepare_chunk_indices(cu_seqlens, chunk_size=64):
    chunk_indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx)
    return chunk_indices


def cumsum_cu_seqlens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.nn.functional.pad(
        torch.cumsum(cu_seqlens, dim=0),
        (1, 0),
        value=0,
    )


def _as_int_list_cu_seqlens(cu_seqlens) -> Optional[List[int]]:
    if cu_seqlens is None:
        return None
    if isinstance(cu_seqlens, torch.Tensor):
        return [int(x) for x in cu_seqlens.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in cu_seqlens]


def _as_int_list_chunk_indices(chunk_indices) -> Optional[List[int]]:
    if chunk_indices is None:
        return None
    if isinstance(chunk_indices, torch.Tensor):
        return [int(x) for x in chunk_indices.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in chunk_indices]


def chunk_bwd_dqkwg_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: Optional[torch.Tensor],
    g: Optional[torch.Tensor],
    dv: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()
    dv_t = dv.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous() if g is not None else None
    w_t = w.transpose(1, 2).contiguous() if w is not None else None
    h_t = h.permute(0, 2, 1, 3, 4).contiguous()
    dh_t = dh.permute(0, 2, 1, 3, 4).contiguous()

    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int64) if cu_seqlens is not None else None

    dq, dk, dw, dg = chunk_bwd_dqkwg_cpu(
        q_t, k_t, v_t, do_t, h_t, dh_t, w_t, g_t, dv_t, scale, cu_seqlens_tensor, chunk_size
    )

    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()
    dw = dw.transpose(1, 2).contiguous()
    if dg is not None:
        dg = dg.transpose(1, 2).contiguous()

    return dq, dk, dw, dg


@register("executor_chunk_bwd_dqkwg")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.qkv_type = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        do = input_data.kwargs["do"]
        h = input_data.kwargs["h"]
        dh = input_data.kwargs["dh"]
        w = input_data.kwargs.get("w", None)
        g = input_data.kwargs["g"]
        dv = input_data.kwargs["dv"]
        cu_seqlens = input_data.kwargs.get("cu_seqlens", None)
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs["scale"]

        dq, dk, dw_out, dg = chunk_bwd_dqkwg_torch(
            q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens, chunk_size
        )

        if self.qkv_type == "bf16":
            dq = dq.to(torch.bfloat16)
            dk = dk.to(torch.bfloat16)
            dw_out = dw_out.to(torch.bfloat16) if dw_out is not None else None
        if self.qkv_type == "fp16":
            dq = dq.to(torch.float16)
            dk = dk.to(torch.float16)
            dw_out = dw_out.to(torch.float16) if dw_out is not None else None

        is_mix = input_data.kwargs.get("is_mix", True)
        if not is_mix:
            if self.qkv_type == "bf16":
                dg = dg.to(torch.bfloat16)
            if self.qkv_type == "fp16":
                dg = dg.to(torch.float16)

        return dq, dk, dw_out, dg

    def init_by_input_data(self, input_data: InputDataset):
        B, H, T_json, K = input_data.kwargs["q"].shape
        V = input_data.kwargs["v"].shape[3]
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        do = input_data.kwargs["do"]
        h = input_data.kwargs["h"]
        dh = input_data.kwargs["dh"]
        w = input_data.kwargs["w"]
        g = input_data.kwargs["g"]
        dv = input_data.kwargs["dv"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs["scale"]

        is_fix = input_data.kwargs["is_fix"]
        self.qkv_type = input_data.kwargs["qkv_type"]

        qkv_type = input_data.kwargs["q"].dtype
        g_type = input_data.kwargs["g"].dtype
        is_mix = input_data.kwargs["is_mix"]
        if not is_mix:
            g_type = qkv_type

        is_fix = False
        if not is_fix:
            cu_seqlens = cumsum_cu_seqlens(cu_seqlens).tolist()
            T = cu_seqlens[-1]
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            num_chunks = len(chunk_indices) // 2
            q = torch.rand((B, H, T, K), dtype=qkv_type) * 5e-7
            k = torch.rand((B, H, T, K), dtype=qkv_type) * 5e-2
            v = torch.rand((B, H, T, V), dtype=qkv_type) * 5e-2
            do = torch.rand((B, H, T, V), dtype=qkv_type) * 5e-7
            dv = torch.rand((B, H, T, V), dtype=qkv_type) * 5e-1
            w = torch.rand((B, H, T, K), dtype=qkv_type) * 5e-2
            g = create_gate_g(B, H, T, g_type)
            h = torch.rand((B, H, num_chunks, K, V), dtype=qkv_type) * 5e-2
            dh = torch.rand((B, H, num_chunks, K, V), dtype=qkv_type) * 5e-2
        else:
            cu_seqlens = None
            chunk_indices = None
            num_chunks = (T_json + chunk_size - 1) // chunk_size
            g = create_gate_g(B, H, T_json, g_type)

        q = q.to(qkv_type)
        k = k.to(qkv_type)
        v = v.to(qkv_type)
        do = do.to(qkv_type)
        dv = dv.to(qkv_type)
        w = w.to(qkv_type) if w is not None else None
        h = h.to(qkv_type)
        dh = dh.to(qkv_type)
        g = g.to(g_type)

        if self.device == "pyaclnn":
            q = q.npu()
            k = k.npu()
            v = v.npu()
            do = do.npu()
            dv = dv.npu()
            w = w.npu() if w is not None else None
            g = g.npu()
            h = h.npu()
            dh = dh.npu()
            if cu_seqlens is not None:
                cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device=q.device)
            if chunk_indices is not None:
                chunk_indices = torch.tensor(chunk_indices, dtype=torch.int64, device=q.device)

        input_data.kwargs["q"] = q
        input_data.kwargs["k"] = k
        input_data.kwargs["v"] = v
        input_data.kwargs["do"] = do
        input_data.kwargs["h"] = h
        input_data.kwargs["dh"] = dh
        input_data.kwargs["w"] = None
        input_data.kwargs["g"] = g
        input_data.kwargs["dv"] = dv
        input_data.kwargs["cu_seqlens"] = cu_seqlens
        input_data.kwargs["chunk_indices"] = chunk_indices
        input_data.kwargs["scale"] = scale
        input_data.kwargs["chunk_size"] = chunk_size
        input_data.kwargs["g_gamma"] = None
        input_data.kwargs.pop("is_mix", None)
        input_data.kwargs.pop("is_fix", None)
        input_data.kwargs.pop("qkv_type", None)
