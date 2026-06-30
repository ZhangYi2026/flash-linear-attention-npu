#!/usr/bin/env python3
import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path

import torch
import torch_npu
import fla_npu  # noqa: F401 - registers custom NPU ops


DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

INPUT_NAMES = ("q", "k", "v", "g", "do", "dv", "h", "dh")
OUTPUT_NAMES = ("dq", "dk", "dw", "dg")
ATOL_DEFAULT = 1e-2
RTOL_DEFAULT = 1e-2
SEED_BASE = 20240617
USE_EXP2_KWARG = True
DQ_DK_HEAD_SEMANTICS = "key_heads"


RAW_CASES = {
    # B, HV, HK, T, chunk_size, dtype, Gtype, scale, cu_seqlens_kind, K, V
    "case_step2_01": (1, 32, 16, 16384, 64, "float16", "float32", 0.03125, ("generated", 0, 16384, 128), 128, 256),
    "case_step2_02": (1, 63, 21, 16384, 64, "bfloat16", "float32", 0.03125, [0, 1066, 2048, 5000, 9000, 10000, 12000, 14000, 16384], 128, 256),
    "case_step2_03": (1, 32, 8, 65536, 128, "float16", "float32", 0.03125, ("generated", 0, 65536, 172), 128, 256),
    "case_step2_04": (1, 32, 16, 65536, 64, "bfloat16", "float32", 0.03125, ("generated", 0, 65536, 668), 128, 128),
    "case_step2_05": (1, 32, 4, 65536, 64, "float16", "float32", 0.03125, ("generated", 0, 65536, 17), 128, 128),
    "case_step2_06": (1, 64, 2, 65519, 64, "bfloat16", "float32", 0.03125, ("generated", 0, 65519, 30), 128, 256),
    "case_step2_07": (1, 32, 16, 4096, 64, "float16", "float32", 0.03125, None, 128, 256),
    "case_step2_08": (16, 63, 21, 2048, 64, "bfloat16", "float32", 0.03125, None, 128, 256),
    "case_step2_09": (711, 32, 4, 196, 128, "float16", "float32", 0.03125, None, 128, 128),
    "case_step2_10": (176, 64, 2, 24, 64, "bfloat16", "float32", 0.03125, None, 128, 256),
    "case_step2_11": (1, 48, 16, 65536, 64, "float16", "float32", 0.03125, ("generated", 0, 65536, 667), 128, 256),
    "case_step2_12": (1, 48, 16, 65536, 128, "bfloat16", "float32", 0.03125, ("generated", 0, 65536, 13), 128, 256),
}


def log(message):
    print(f"[{datetime.now().strftime('%F %T')}] {message}", flush=True)


def case_sort_key(name):
    try:
        return int(str(name).split("_")[-1])
    except ValueError:
        return str(name)


def generated_increasing_sequence(start, end, count, seed):
    if count < 2:
        return [int(start)]
    rng = random.Random(seed)
    middle = sorted(rng.sample(range(int(start) + 1, int(end)), int(count) - 2))
    return [int(start)] + middle + [int(end)]


def total_chunks_from_cu(cu_seqlens, chunk_size):
    total = 0
    for start, end in zip(cu_seqlens, cu_seqlens[1:]):
        length = int(end) - int(start)
        if length > 0:
            total += (length + chunk_size - 1) // chunk_size
    return total


def prepare_chunk_indices(cu_seqlens, chunk_size):
    if cu_seqlens is None:
        return None
    indices = []
    for i, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        length = int(end) - int(start)
        if length <= 0:
            continue
        for chunk_id in range((length + chunk_size - 1) // chunk_size):
            indices.append(i)
            indices.append(chunk_id)
    return indices


def build_case_specs():
    specs = {}
    for idx, name in enumerate(sorted(RAW_CASES, key=case_sort_key), start=1):
        b, hv, hk, t, chunk_size, dtype, gtype, scale, cu_kind, kdim, vdim = RAW_CASES[name]
        seed = SEED_BASE + 2000 + idx
        cu_seed = seed + 100000
        if isinstance(cu_kind, tuple):
            _, start, end, count = cu_kind
            cu_seqlens = generated_increasing_sequence(start, end, count, cu_seed)
            cu_source = {"kind": "generated", "start": start, "end": end, "count": count, "seed": cu_seed}
        elif cu_kind is None:
            cu_seqlens = None
            cu_source = {"kind": "none"}
        else:
            cu_seqlens = [int(item) for item in cu_kind]
            cu_source = {"kind": "literal"}
        if hv % hk != 0:
            raise ValueError(f"{name}: HV must be divisible by HK")
        num_chunks = (
            (int(t) + int(chunk_size) - 1) // int(chunk_size)
            if cu_seqlens is None
            else total_chunks_from_cu(cu_seqlens, int(chunk_size))
        )
        specs[name] = {
            "case": name,
            "seed": seed,
            "cu_seed": cu_seed,
            "layout": "BTH_inputs_BHT_outputs",
            "B": int(b),
            "HV": int(hv),
            "HK": int(hk),
            "T": int(t),
            "K": int(kdim),
            "V": int(vdim),
            "chunk_size": int(chunk_size),
            "num_chunks": int(num_chunks),
            "dtype": dtype,
            "Gtype": gtype,
            "scale": float(scale),
            "cu_seqlens": cu_seqlens,
            "cu_source": cu_source,
            "input_scale": {
                "q": 5e-2,
                "k": 5e-2,
                "v": 5e-2,
                "do": 5e-2,
                "dv": 5e-1,
                "h": 5e-2,
                "dh": 5e-4,
                "g_uniform_sorted_negative": 100.0,
            },
        }
    return specs


CASE_SPECS = build_case_specs()


def parse_case_selector(selectors):
    if not selectors or selectors == ["all"]:
        return sorted(CASE_SPECS, key=case_sort_key)
    cases = []
    for item in selectors:
        for token in item.split(","):
            token = token.strip()
            if not token:
                continue
            if token.startswith("case_step2_"):
                case_name = token
            else:
                case_name = f"case_step2_{int(token):02d}"
            if case_name not in CASE_SPECS:
                raise SystemExit(f"unknown case {case_name}")
            cases.append(case_name)
    return cases


def load_cpu_ref(path):
    dummy = types.ModuleType("ct")
    dummy.single = lambda *args, **kwargs: None
    sys.modules.setdefault("ct", dummy)
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location("step2_cpu_ref", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.chunk_bwd_dqkwg_cpu


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)
        fp.write("\n")
    tmp.replace(path)


def save_tensor(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.contiguous(), path)


def file_size_sum(path):
    total = 0
    for item in Path(path).rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def randn_tensor(shape, scale, dtype):
    try:
        tensor = torch.randn(shape, dtype=dtype)
    except RuntimeError:
        tensor = torch.randn(shape, dtype=torch.float32).to(dtype)
    tensor.mul_(scale)
    return tensor.contiguous()


def make_cpu_inputs(spec):
    dtype = DTYPES[spec["dtype"]]
    gtype = DTYPES[spec["Gtype"]]
    b = spec["B"]
    hv = spec["HV"]
    hk = spec["HK"]
    t = spec["T"]
    kdim = spec["K"]
    vdim = spec["V"]
    nt = spec["num_chunks"]
    seed = int(spec["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    q = randn_tensor((b, t, hk, kdim), 5e-2, dtype)
    k = randn_tensor((b, t, hk, kdim), 5e-2, dtype)
    v = randn_tensor((b, t, hv, vdim), 5e-2, dtype)
    do = randn_tensor((b, t, hv, vdim), 5e-2, dtype)
    dv = randn_tensor((b, t, hv, vdim), 5e-1, dtype)
    h = randn_tensor((b, nt, hv, kdim, vdim), 5e-2, dtype)
    dh = randn_tensor((b, nt, hv, kdim, vdim), 5e-4, dtype)
    g = torch.rand((b, t, hv), dtype=torch.float32)
    g = -torch.sort(g.mul(100.0), dim=1, descending=False).values.to(gtype).contiguous()
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "do": do,
        "dv": dv,
        "h": h,
        "dh": dh,
    }


def cache_complete(case_dir, spec):
    marker = Path(case_dir) / ".complete"
    meta_path = Path(case_dir) / "meta.json"
    if not marker.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("dq_dk_head_semantics") != DQ_DK_HEAD_SEMANTICS:
        return False
    keys = ("seed", "cu_seed", "B", "HV", "HK", "T", "K", "V", "chunk_size", "dtype", "Gtype", "scale", "cu_seqlens", "num_chunks")
    if any(meta.get(key) != spec.get(key) for key in keys):
        return False
    for name in INPUT_NAMES:
        if not (Path(case_dir) / "in" / f"{name}.pt").exists():
            return False
    for name in OUTPUT_NAMES:
        if not (Path(case_dir) / "out" / f"{name}.pt").exists():
            return False
    return True


def generate_cache(case_name, spec, cache_root, cpu_ref, refresh=False):
    case_dir = Path(cache_root) / case_name
    if not refresh and cache_complete(case_dir, spec):
        log(f"{case_name}: cache hit {case_dir}")
        return "hit", case_dir

    tmp_dir = Path(cache_root) / f".{case_name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log(f"{case_name}: cache miss, generating fixed-seed CPU golden")
    started = time.perf_counter()

    tensors = make_cpu_inputs(spec)
    for name in INPUT_NAMES:
        save_tensor(tmp_dir / "in" / f"{name}.pt", tensors[name])

    cu = spec["cu_seqlens"]
    cu_tensor = torch.tensor(cu, dtype=torch.int64) if cu is not None else None
    w_dummy = torch.empty(0, dtype=DTYPES[spec["dtype"]])
    dq, dk, dw, dg = cpu_ref(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["do"],
        tensors["h"],
        tensors["dh"],
        w_dummy,
        tensors["g"],
        tensors["dv"],
        float(spec["scale"]),
        cu_tensor,
        int(spec["chunk_size"]),
    )
    outputs = {
        "dq": dq.transpose(1, 2).contiguous(),
        "dk": dk.transpose(1, 2).contiguous(),
        "dw": dw.transpose(1, 2).contiguous(),
        "dg": dg.transpose(1, 2).contiguous(),
    }
    for name in OUTPUT_NAMES:
        save_tensor(tmp_dir / "out" / f"{name}.pt", outputs[name])

    elapsed = time.perf_counter() - started
    meta = dict(spec)
    meta.update(
        {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "generation_seconds": elapsed,
            "dq_dk_head_semantics": DQ_DK_HEAD_SEMANTICS,
            "input_files": list(INPUT_NAMES),
            "output_files": list(OUTPUT_NAMES),
        }
    )
    atomic_write_json(tmp_dir / "meta.json", meta)
    (tmp_dir / ".complete").write_text("ok\n", encoding="utf-8")
    if case_dir.exists():
        shutil.rmtree(case_dir)
    tmp_dir.replace(case_dir)
    meta["size_bytes"] = file_size_sum(case_dir)
    meta["size_gib"] = round(meta["size_bytes"] / (1024 ** 3), 3)
    atomic_write_json(case_dir / "meta.json", meta)
    del tensors, outputs, dq, dk, dw, dg
    gc.collect()
    log(f"{case_name}: cache saved {case_dir} ({meta['size_gib']} GiB, {elapsed:.1f}s)")
    return "miss_saved", case_dir


def load_inputs_to_npu(case_dir, spec, device):
    dtype = DTYPES[spec["dtype"]]
    gtype = DTYPES[spec["Gtype"]]
    torch_npu.npu.set_device(device)
    tensors = {
        name: torch.load(Path(case_dir) / "in" / f"{name}.pt", map_location="cpu", weights_only=False)
        for name in INPUT_NAMES
    }
    q = tensors["q"].to(dtype).transpose(1, 2).contiguous().npu()
    k = tensors["k"].to(dtype).transpose(1, 2).contiguous().npu()
    v = tensors["v"].to(dtype).transpose(1, 2).contiguous().npu()
    g = tensors["g"].to(gtype).transpose(1, 2).contiguous().npu()
    do = tensors["do"].to(dtype).transpose(1, 2).contiguous().npu()
    dv = tensors["dv"].to(dtype).transpose(1, 2).contiguous().npu()
    h = tensors["h"].to(dtype).transpose(1, 2).contiguous().npu()
    dh = tensors["dh"].to(dtype).transpose(1, 2).contiguous().npu()
    del tensors
    gc.collect()
    return q, k, v, g, h, do, dh, dv


def invoke_op(inputs, spec):
    global USE_EXP2_KWARG
    cu_seqlens = spec["cu_seqlens"]
    chunk_indices = prepare_chunk_indices(cu_seqlens, int(spec["chunk_size"]))
    kwargs = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        w=None,
        g_gamma=None,
        scale=float(spec["scale"]),
        transpose_state_layout=None,
    )
    if USE_EXP2_KWARG:
        try:
            return torch.ops.npu.npu_chunk_bwd_dqkwg(
                *inputs,
                int(spec["chunk_size"]),
                use_exp2=None,
                **kwargs,
            )
        except TypeError as exc:
            if "use_exp2" not in str(exc):
                raise
            USE_EXP2_KWARG = False
    return torch.ops.npu.npu_chunk_bwd_dqkwg(
        *inputs,
        int(spec["chunk_size"]),
        **kwargs,
    )


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def compare_tensor(case_name, output_name, actual, expected, atol, rtol, rel_eps):
    actual_f = actual.to(torch.float32)
    expected_f = expected.to(torch.float32)
    finite = bool(torch.isfinite(actual_f).all().item() and torch.isfinite(expected_f).all().item())
    if finite:
        diff = (actual_f - expected_f).abs()
        rel = diff / expected_f.abs().clamp_min(rel_eps)
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        max_rel = float(rel.max().item()) if rel.numel() else 0.0
        mean_rel = float(rel.mean().item()) if rel.numel() else 0.0
        passed = (max_abs <= atol) or (max_rel <= rtol)
    else:
        max_abs = math.inf
        mean_abs = math.inf
        max_rel = math.inf
        mean_rel = math.inf
        passed = False
    return {
        "case": case_name,
        "output": output_name,
        "status": "PASS" if passed else "FAIL",
        "shape_actual": list(actual.shape),
        "shape_expected": list(expected.shape),
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "atol": atol,
        "rtol": rtol,
    }


def run_npu_validation(case_name, spec, case_dir, device, atol, rtol, rel_eps):
    rows = []
    inputs = load_inputs_to_npu(case_dir, spec, device)
    torch_npu.npu.synchronize()
    out = invoke_op(inputs, spec)
    torch_npu.npu.synchronize()
    for idx, output_name in enumerate(OUTPUT_NAMES):
        actual = out[idx].cpu()
        expected = torch.load(Path(case_dir) / "out" / f"{output_name}.pt", map_location="cpu", weights_only=False)
        row = compare_tensor(case_name, output_name, actual, expected, atol, rtol, rel_eps)
        row.update({
            "seed": spec["seed"],
            "cu_seed": spec["cu_seed"],
            "B": spec["B"],
            "HV": spec["HV"],
            "HK": spec["HK"],
            "T": spec["T"],
            "K": spec["K"],
            "V": spec["V"],
            "chunk_size": spec["chunk_size"],
            "dtype": spec["dtype"],
            "Gtype": spec["Gtype"],
            "cache_dir": str(case_dir),
        })
        log("[golden] " + json.dumps(row, sort_keys=True))
        rows.append(row)
        del actual, expected
        gc.collect()
    del out
    return rows, inputs


def run_perf(case_name, spec, inputs, device, warmup, repeat):
    result = {
        "case": case_name,
        "device": device,
        "warmup": warmup,
        "repeat": repeat,
        "spec": spec,
        "status": "unknown",
        "use_exp2_kwarg": USE_EXP2_KWARG,
    }
    try:
        with torch.no_grad():
            for _ in range(warmup):
                out = invoke_op(inputs, spec)
                torch_npu.npu.synchronize()
                del out
            elapsed_ms = []
            for _ in range(repeat):
                start = time.perf_counter()
                out = invoke_op(inputs, spec)
                torch_npu.npu.synchronize()
                elapsed_ms.append((time.perf_counter() - start) * 1000.0)
                del out
        result.update({
            "status": "ok",
            "elapsed_ms": elapsed_ms,
            "min_ms": min(elapsed_ms),
            "p50_ms": statistics.median(elapsed_ms),
            "mean_ms": statistics.mean(elapsed_ms),
            "p90_ms": percentile(elapsed_ms, 90),
            "max_ms": max(elapsed_ms),
        })
    except BaseException as exc:
        result.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        log("[perf-error] " + result["traceback"])
    log("[perf] " + json.dumps({k: v for k, v in result.items() if k != "elapsed_ms"}, sort_keys=True))
    return result


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path, rows):
    fieldnames = [
        "case", "output", "status", "max_abs", "mean_abs", "max_rel", "mean_rel",
        "finite", "atol", "rtol", "seed", "cu_seed", "B", "HV", "HK", "T", "K",
        "V", "chunk_size", "dtype", "Gtype", "cache_status", "cache_dir",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(path, summaries):
    fields = [
        "case", "status", "cache_status", "golden_status", "perf_status",
        "mean_ms", "p50_ms", "min_ms", "max_ms", "cache_gib", "error",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fields})


def load_existing_summary(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp, delimiter="\t"))


def cleanup_npu():
    gc.collect()
    try:
        torch.npu.empty_cache()
    except Exception:
        pass


def run_case(case_name, args, cpu_ref):
    spec = dict(CASE_SPECS[case_name])
    result_dir = Path(args.results_dir)
    cache_status = ""
    case_summary = {
        "case": case_name,
        "status": "unknown",
        "cache_status": "",
        "golden_status": "",
        "perf_status": "",
        "error": "",
    }
    try:
        cache_status, case_dir = generate_cache(case_name, spec, args.cache_root, cpu_ref, args.refresh_cache)
        case_summary["cache_status"] = cache_status
        case_summary["cache_gib"] = round(file_size_sum(case_dir) / (1024 ** 3), 3)

        golden_rows = []
        inputs = None
        if not args.skip_golden:
            golden_rows, inputs = run_npu_validation(case_name, spec, case_dir, args.device, args.atol, args.rtol, args.rel_eps)
            for row in golden_rows:
                row["cache_status"] = cache_status
            write_jsonl(result_dir / f"golden_{case_name}_results.jsonl", golden_rows)
            write_csv(result_dir / f"golden_{case_name}_results.csv", golden_rows)
            case_summary["golden_status"] = "PASS" if all(row["status"] == "PASS" for row in golden_rows) else "FAIL"

        if not args.skip_perf:
            if inputs is None:
                inputs = load_inputs_to_npu(case_dir, spec, args.device)
                torch_npu.npu.synchronize()
            perf = run_perf(case_name, spec, inputs, args.device, args.warmup, args.repeat)
            atomic_write_json(result_dir / f"perf_{case_name}.json", perf)
            case_summary["perf_status"] = perf.get("status", "")
            for key in ("mean_ms", "p50_ms", "min_ms", "max_ms"):
                if key in perf:
                    case_summary[key] = perf[key]
        case_summary["status"] = (
            "ok"
            if case_summary.get("golden_status", "PASS") != "FAIL" and case_summary.get("perf_status", "ok") != "error"
            else "failed"
        )
    except BaseException as exc:
        case_summary["status"] = "error"
        case_summary["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(
            result_dir / f"error_{case_name}.json",
            {
                "case": case_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "spec": spec,
            },
        )
        log("[case-error] " + traceback.format_exc())
    finally:
        cleanup_npu()
    return case_summary


def write_manifest(cache_root, results_dir, cases, summaries):
    manifest_cases = []
    for case_name in cases:
        case_dir = Path(cache_root) / case_name
        spec = dict(CASE_SPECS[case_name])
        spec["path"] = str(case_dir)
        if case_dir.exists():
            spec["size_bytes"] = file_size_sum(case_dir)
            spec["size_gib"] = round(spec["size_bytes"] / (1024 ** 3), 3)
            spec["cache_complete"] = cache_complete(case_dir, CASE_SPECS[case_name])
        manifest_cases.append(spec)
    atomic_write_json(
        Path(results_dir) / "step2_cache_manifest.json",
        {
            "cache_root": str(cache_root),
            "seed_base": SEED_BASE,
            "generated_from": "case_step2 definitions in pr96_clean_dqkwg_20260628_125741/tests/cases.py",
            "cases": manifest_cases,
            "summaries": summaries,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Generate fixed-seed step2 golden cache and run DQKWG NPU golden/perf.")
    parser.add_argument("cases", nargs="*", help="case selectors: all, 1, case_step2_01, or comma lists")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--cpu-ref", required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "14")))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--atol", type=float, default=ATOL_DEFAULT)
    parser.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    parser.add_argument("--rel-eps", type=float, default=1e-6)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--skip-perf", action="store_true")
    args = parser.parse_args()

    try:
        torch.npu.config.allow_internal_format = False
        torch.npu.set_compile_mode(jit_compile=False)
    except Exception:
        pass

    cases = parse_case_selector(args.cases)
    Path(args.cache_root).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    cpu_ref = load_cpu_ref(args.cpu_ref)

    log(f"cache_root={args.cache_root}")
    log(f"results_dir={args.results_dir}")
    log(f"device={args.device}")
    log(f"cases={' '.join(cases)}")
    summary_path = Path(args.results_dir) / "step2_summary.tsv"
    summaries = load_existing_summary(summary_path)
    for case_name in cases:
        log(f"START {case_name}")
        summary = run_case(case_name, args, cpu_ref)
        summaries = [row for row in summaries if row.get("case") != case_name]
        summaries.append(summary)
        summaries = sorted(summaries, key=lambda row: case_sort_key(row.get("case", "")))
        write_summary(summary_path, summaries)
        write_manifest(args.cache_root, args.results_dir, sorted(CASE_SPECS, key=case_sort_key), summaries)
        log(f"END {case_name} status={summary.get('status')}")

    run_case_set = set(cases)
    failed = any(row.get("case") in run_case_set and row.get("status") != "ok" for row in summaries)
    log("STEP2_DONE status=" + ("failed" if failed else "ok"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
