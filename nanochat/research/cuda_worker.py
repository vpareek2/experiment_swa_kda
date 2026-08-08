"""GPU subprocesses for the protected CUDA-ownership supervisor.

The worker emits bounded JSON only.  FLA may be used by the unmodified baseline
or in an isolated oracle process, but ``runtime-audit`` installs an import
blocker before exercising the production path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from nanochat.research.cuda_config import cuda_campaign_config_from_dict, load_cuda_campaign_config


class _ForbiddenRuntimeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, prefixes: tuple[str, ...]):
        self.prefixes = prefixes
        self.attempts: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in self.prefixes):
            self.attempts.append(fullname)
            raise ModuleNotFoundError(f"runtime module forbidden by CUDA-ownership audit: {fullname}", name=fullname)
        return None


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_oracle(root: Path):
    spec = importlib.util.spec_from_file_location("protected_kda_oracle", root / "tests" / "kda_oracle.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent KDA oracle")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def _provenance(root: Path, module, config) -> dict[str, Any]:
    callable_name = config.ownership.metadata_callable
    describe = getattr(module, callable_name, None)
    if not callable(describe):
        return {"status": "missing", "reason": f"nanochat.mixers.kda.{callable_name} is required", "owned_fraction": 0.0, "components": {}}
    raw = describe()
    if not isinstance(raw, dict) or not isinstance(raw.get("components"), dict):
        return {"status": "invalid", "reason": "backend provenance must contain a components table", "owned_fraction": 0.0, "components": {}}
    if set(raw["components"]) != set(config.ownership.required_components):
        return {"status":"invalid","reason":"provenance must name exactly every frozen component","owned_fraction":0.0,"components":{}}
    candidate_root=root / config.campaign.candidate_paths[0]
    forbidden_tokens=(b"import fla",b"from fla",b"_run_fla",b"_fla_ops",b"_fla_causal_conv1d",b"_reference_kda",b"tests.kda_oracle",b"/ref/")
    for candidate_source in candidate_root.rglob("*"):
        if candidate_source.is_file() and candidate_source.suffix.lower() in {".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".ptx"}:
            lowered=candidate_source.read_bytes().lower()
            if any(token in lowered for token in forbidden_tokens):
                return {"status":"invalid","reason":f"candidate source references a forbidden runtime/oracle path: {candidate_source.relative_to(root)}","owned_fraction":0.0,"components":{}}
            if candidate_source.suffix.lower()==".py" and (b"torch.library" in lowered or b"from torch import library" in lowered):
                return {"status":"invalid","reason":f"CUDA operator registration must come from the loaded native extension, not Python: {candidate_source.relative_to(root)}","owned_fraction":0.0,"components":{}}
    components, evidence = {}, {}
    inline_ptx_sources: set[str] = set()
    weights = dict(zip(config.ownership.required_components, config.ownership.component_weights))
    owned = 0.0
    for name in config.ownership.required_components:
        entry = raw["components"].get(name)
        if not isinstance(entry, dict) or entry.get("owner") not in {"project", "third_party"}:
            return {"status": "invalid", "reason": f"missing valid provenance for {name}", "owned_fraction": 0.0, "components": components}
        sources = entry.get("sources", [])
        symbols = entry.get("kernel_symbols", [])
        torch_operator = entry.get("torch_operator")
        if not isinstance(sources, list) or not all(isinstance(item, str) and item for item in sources):
            return {"status": "invalid", "reason": f"invalid source list for {name}", "owned_fraction": 0.0, "components": components}
        if entry["owner"] == "project" and (not isinstance(symbols, list) or not symbols or not all(isinstance(item, str) and item for item in symbols)):
            return {"status": "invalid", "reason": f"project-owned component {name} requires expected kernel symbols", "owned_fraction": 0.0, "components": components}
        if entry["owner"] == "third_party" and (sources or symbols or torch_operator not in {None, ""}):
            return {"status": "invalid", "reason": f"unclaimed component {name} must not attach project evidence", "owned_fraction": 0.0, "components": components}
        if entry["owner"] == "project":
            import torch
            if not isinstance(torch_operator, str) or not torch_operator.startswith("nanochat_kda::") or not torch._C._dispatch_has_kernel_for_dispatch_key(torch_operator, "CUDA"):
                return {"status": "invalid", "reason": f"project-owned component {name} requires a registered nanochat_kda CUDA operator", "owned_fraction": 0.0, "components": components}
        checked = []
        for source in sources:
            relative = Path(source)
            resolved = (root / relative).resolve()
            allowed_root = any(source == prefix.rstrip("/") or source.startswith(prefix) for prefix in config.ownership.project_source_roots)
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", source], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if relative.is_absolute() or root not in resolved.parents or not resolved.is_file() or not allowed_root or not tracked:
                return {"status": "invalid", "reason": f"source outside tracked project-owned roots: {source}", "owned_fraction": 0.0, "components": components}
            if resolved.suffix.lower() not in config.ownership.source_extensions:
                return {"status": "invalid", "reason": f"unapproved CUDA/build source extension: {source}", "owned_fraction": 0.0, "components": components}
            source_bytes = resolved.read_bytes(); lowered = source_bytes.lower()
            if resolved.suffix.lower() in {".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"}:
                forbidden_tokens=(b"import fla",b"from fla",b"_run_fla",b"_reference_kda",b"tests.kda_oracle",b"/ref/")
                if any(token in lowered for token in forbidden_tokens):
                    return {"status": "invalid", "reason": f"candidate source references a forbidden runtime/oracle path: {source}", "owned_fraction": 0.0, "components": components}
            if b"tcgen05" in lowered or b"tmem" in lowered:
                return {"status": "invalid", "reason": f"source uses unsupported SM121 instruction family: {source}", "owned_fraction": 0.0, "components": components}
            if resolved.suffix.lower() in {".cu", ".cuh"} and (b"asm(" in lowered or b"asm volatile" in lowered):
                inline_ptx_sources.add(source)
            checked.append({"path": source, "sha256": hashlib.sha256(source_bytes).hexdigest()})
        if entry["owner"] == "project" and not any(Path(item["path"]).suffix.lower() in config.ownership.native_cuda_extensions for item in checked):
            return {"status": "invalid", "reason": f"project-owned component {name} has no tracked native CUDA source", "owned_fraction": 0.0, "components": components}
        components[name] = {"owner": entry["owner"], "sources": checked, "kernel_symbols": symbols, "torch_operator": torch_operator}
        if entry["owner"] == "project":
            owned += weights[name]
        evidence[name] = entry.get("runtime", "declared")
    project_ops=[component["torch_operator"] for component in components.values() if component["owner"]=="project"]
    if len(project_ops) != len(set(project_ops)):
        return {"status":"invalid","reason":"each project-owned component requires a distinct CUDA operator","owned_fraction":0.0,"components":components}
    project_sources = {entry["path"] for component in components.values() if component["owner"] == "project" for entry in component["sources"]}
    project_native_sources = {source for source in project_sources if Path(source).suffix.lower() in config.ownership.native_cuda_extensions}
    build = raw.get("build")
    required_build = {"library_path", "source_paths", "compiler_command", "target_arch"}
    if project_sources:
        if not isinstance(build, dict) or set(build) != required_build:
            return {"status": "invalid", "reason": "project CUDA requires an exact protected build receipt", "owned_fraction": 0.0, "components": components}
        cache_value=os.environ.get("TORCH_EXTENSIONS_DIR")
        if not cache_value:
            return {"status": "invalid", "reason": "isolated TORCH_EXTENSIONS_DIR is required", "owned_fraction": 0.0, "components": components}
        library = Path(build["library_path"]).resolve(); cache_root = Path(cache_value).resolve()
        build_sources = set(build["source_paths"]) if isinstance(build["source_paths"], list) else set()
        mapped = library.is_file() and any(str(library) in line for line in Path("/proc/self/maps").read_text(errors="replace").splitlines())
        if library.suffix != ".so" or cache_root not in library.parents or not mapped:
            return {"status": "invalid", "reason": "loaded CUDA library is not mapped from the isolated extension cache", "owned_fraction": 0.0, "components": components}
        if not project_native_sources <= build_sources or not isinstance(build["compiler_command"], str) or not build["compiler_command"] or "121" not in str(build["target_arch"]):
            return {"status": "invalid", "reason": "build receipt does not bind reviewed sources and SM121 target", "owned_fraction": 0.0, "components": components}
        build_receipt = {**build, "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
    else:
        build_receipt = None
    selective_ptx = raw.get("selective_ptx", [])
    if not isinstance(selective_ptx, list):
        return {"status": "invalid", "reason": "selective_ptx must be a list", "owned_fraction": 0.0, "components": components}
    declared_ptx_sources = set()
    for item in selective_ptx:
        required = {"source", "rationale", "architecture_guard", "cuda_fallback", "profile_evidence"}
        if not isinstance(item, dict) or set(item) != required or not all(isinstance(item[key], str) and item[key] for key in required):
            return {"status": "invalid", "reason": "selective PTX requires exact source/rationale/guard/fallback/profile evidence", "owned_fraction": 0.0, "components": components}
        if "tcgen05" in json.dumps(item).lower() or "tmem" in json.dumps(item).lower():
            return {"status": "invalid", "reason": "SM121 does not support tcgen05 or TMEM", "owned_fraction": 0.0, "components": components}
        declared_ptx_sources.add(item["source"])
    all_sources = {entry["path"] for component in components.values() for entry in component["sources"]}
    raw_ptx_sources = {source for source in all_sources if Path(source).suffix.lower() == ".ptx"}
    if (raw_ptx_sources | inline_ptx_sources) - declared_ptx_sources:
        return {"status": "invalid", "reason": "tracked or inline PTX source lacks a selective-PTX declaration", "owned_fraction": 0.0, "components": components}
    return {"status": "complete", "owned_fraction": owned, "components": components, "runtime_evidence": evidence, "build": build_receipt, "selective_ptx": selective_ptx}


def _inputs(torch, length: int, dim: int, *, gradients: bool, value_dim: int | None = None,
            batch: int = 1, heads: int = 1):
    value_dim = dim if value_dim is None else value_dim
    torch.manual_seed(1234 + length + dim + value_dim + batch + heads)
    q = torch.randn(batch, length, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, length, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(q)
    beta = torch.randn(batch, length, heads, device="cuda", dtype=torch.bfloat16)
    if gradients:
        for tensor in (q, k, v, gate, beta): tensor.requires_grad_(True)
    A = torch.zeros(heads, device="cuda", dtype=torch.float32, requires_grad=gradients)
    dt = torch.zeros(heads * dim, device="cuda", dtype=torch.float32, requires_grad=gradients)
    return q, k, v, gate, beta, A, dt


class _NativeOperatorRecorder:
    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        class Recorder(TorchDispatchMode):
            def __init__(inner): super().__init__(); inner.names=[]
            def __torch_dispatch__(inner,func,types,args=(),kwargs=None):
                inner.names.append(str(func)); return func(*args,**(kwargs or {}))
        self.mode=Recorder()
    def __enter__(self): self.mode.__enter__(); return self
    def __exit__(self,*args): return self.mode.__exit__(*args)
    @property
    def names(self): return self.mode.names


def _operator_observed(declared: str, observed: list[str]) -> bool:
    normalized=declared.replace("::", ".")
    return any(name == declared or name.startswith(declared + ".") or name == normalized or name.startswith(normalized + ".") for name in observed)


def runtime_audit(root: Path, config, lane: str) -> dict[str, Any]:
    import torch
    if lane not in {"bootstrap", "migration", "optimization"}:
        return {"status": "invalid", "reason": f"unknown campaign lane: {lane}"}
    if not torch.cuda.is_available():
        return {"status": "invalid", "reason": "CUDA is required"}
    implementation = importlib.import_module("nanochat.mixers.kda")
    transitional = lane in {"bootstrap", "migration"}
    # Protected routing may use FLA only for explicitly unclaimed components
    # in transitional lanes. Load those callables before candidate glue, then
    # remove public modules and install a blocker so candidate code cannot
    # import or capture FLA itself.
    if transitional:
        implementation._load_fla_ops()
        implementation._load_fla_causal_conv1d()
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in config.ownership.forbid_runtime_modules):
            del sys.modules[name]
    finder = _ForbiddenRuntimeFinder(config.ownership.forbid_runtime_modules)
    sys.meta_path.insert(0, finder)
    original_reference = implementation._reference_kda
    def blocked_reference(*args, **kwargs):
        raise RuntimeError("sequential reference backend is forbidden in project CUDA runtime audit")
    implementation._reference_kda = blocked_reference
    checks: list[dict[str, Any]] = []; provenance: dict[str, Any] = {}; operator_recorder=None
    try:
        project = implementation._load_project_backend()
        project.prepare()
        # Refresh the protected dispatcher's cached claims after the build.
        implementation._PROJECT_PROVENANCE = None
        provenance = _provenance(root, implementation, config)
        if provenance.get("status") != "complete":
            raise RuntimeError(provenance.get("reason", "invalid project provenance"))
        components = provenance["components"]
        for forward, backward in (("chunk_forward", "chunk_backward"), ("causal_convolution_forward", "causal_convolution_backward")):
            if (components[forward]["owner"] == "project") != (components[backward]["owner"] == "project"):
                raise RuntimeError(f"{forward}/{backward} must migrate as one autograd capability")
        oracle = _load_oracle(root)
        implementation.reset_project_runtime_events()
        operator_recorder=_NativeOperatorRecorder(); operator_recorder.__enter__()

        # Protected hybrid causal-convolution routing, including gradients and cache.
        torch.manual_seed(991)
        conv = implementation.ShortConvolution(32, 4).cuda()
        conv_x = torch.randn(2, 65, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        conv_initial = torch.randn(2, 32, 4, device="cuda", dtype=torch.bfloat16); conv_snapshot = conv_initial.clone()
        expected_x = conv_x.detach().clone().requires_grad_(True)
        expected_weight = conv.weight.detach().squeeze(1).to(torch.bfloat16).requires_grad_(True)
        conv_actual, conv_state = conv(conv_x, conv_initial, output_final_state=True, backend="project_cuda")
        conv_expected, expected_conv_state = oracle.causal_depthwise_conv(expected_x, expected_weight, initial_state=conv_initial)
        torch.testing.assert_close(conv_actual, conv_expected, atol=config.correctness.forward_atol, rtol=config.correctness.forward_rtol)
        torch.testing.assert_close(conv_state, expected_conv_state, atol=config.correctness.forward_atol, rtol=config.correctness.forward_rtol)
        conv_actual.float().square().mean().backward(); conv_expected.float().square().mean().backward()
        torch.testing.assert_close(conv_x.grad, expected_x.grad, atol=config.correctness.gradient_atol, rtol=config.correctness.gradient_rtol)
        torch.testing.assert_close(conv.weight.grad.squeeze(1).to(torch.bfloat16), expected_weight.grad, atol=config.correctness.gradient_atol, rtol=config.correctness.gradient_rtol)
        torch.testing.assert_close(conv_initial, conv_snapshot, atol=0, rtol=0)
        checks.append({"kind": "causal_convolution_forward_backward_cache", "length": 65})

        layer = implementation.KimiDeltaAttention(128, 1, 128, mode="project_chunk", allow_fallback=False).cuda()
        hidden = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        layer_output, layer_state = layer(hidden, output_final_state=True, mode="project_chunk", allow_fallback=False)
        if layer_state is None or not bool(torch.isfinite(layer_output).all() and torch.isfinite(layer_state.memory).all()):
            raise AssertionError("project CUDA hybrid layer produced missing or non-finite state")
        layer_output.float().square().mean().backward()
        if hidden.grad is None or not bool(torch.isfinite(hidden.grad).all()) or any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in layer.parameters()):
            raise AssertionError("project CUDA hybrid-layer gradients are missing or non-finite")
        checks.append({"kind": "full_layer_forward_backward_state", "length": 4, "head_dim": 128})

        lane_limit = config.bootstrap.maximum_sequence_length if lane == "bootstrap" else (config.migration.maximum_sequence_length if lane == "migration" else config.measurement.sequence_length)
        boundaries = [length for length in config.correctness.boundary_lengths if length <= lane_limit]
        for length in boundaries:
            q, k, v, gate, beta, A, dt = _inputs(torch, length, config.correctness.production_head_dim, gradients=False)
            expected, expected_state = oracle.recurrent_kda(q, k, v, gate, beta, A, dt)
            actual, state = implementation.kda(q, k, v, gate, beta, A, dt, output_final_state=True, mode="project_chunk", allow_fallback=False)
            torch.testing.assert_close(actual, expected, atol=config.correctness.forward_atol, rtol=config.correctness.forward_rtol)
            torch.testing.assert_close(state, expected_state, atol=config.correctness.forward_atol, rtol=config.correctness.forward_rtol)
            if not bool(torch.isfinite(actual).all() and torch.isfinite(state).all()): raise AssertionError("non-finite chunk output/state")
            checks.append({"kind": "boundary_forward_state", "length": length})

        q, k, v, gate, beta, A, dt = _inputs(torch, 33, 16, value_dim=24, batch=2, heads=2, gradients=False)
        initial = torch.randn(2, 2, 24, 16, device="cuda", dtype=torch.float32); snapshot = initial.clone()
        expected, expected_state = oracle.recurrent_kda(q, k, v, gate, beta, A, dt, initial_state=initial)
        actual, state = implementation.kda(q, k, v, gate, beta, A, dt, initial_state=initial, output_final_state=True, mode="project_chunk", allow_fallback=False)
        torch.testing.assert_close(actual, expected, atol=config.correctness.forward_atol, rtol=config.correctness.forward_rtol)
        torch.testing.assert_close(state, expected_state, atol=config.correctness.forward_atol, rtol=config.correctness.forward_rtol)
        torch.testing.assert_close(initial, snapshot, atol=0, rtol=0)
        prefix, _ = implementation.kda(q[:, :17], k[:, :17], v[:, :17], gate[:, :17], beta[:, :17], A, dt, initial_state=initial, mode="project_chunk", allow_fallback=False)
        torch.testing.assert_close(prefix, actual[:, :17], atol=config.correctness.forward_atol, rtol=config.correctness.forward_rtol)
        checks.append({"kind": "unequal_dimensions_state_causality", "length": 33})

        extreme_length = min(257, lane_limit)
        qx, kx, vx, gx, bx, Ax, dtx = _inputs(torch, extreme_length, 16, gradients=False)
        gx[0, ::2].fill_(30); gx[0, 1::2].fill_(-30); bx[0, ::2].fill_(30); bx[0, 1::2].fill_(-30)
        extreme, extreme_state = implementation.kda(qx, kx, vx, gx, bx, Ax, dtx, output_final_state=True, mode="project_chunk", allow_fallback=False)
        if not bool(torch.isfinite(extreme).all() and torch.isfinite(extreme_state).all()): raise AssertionError("extreme gates produced non-finite values")
        checks.append({"kind": "extreme_gate_finiteness", "length": extreme_length})

        values = _inputs(torch, 65, 16, gradients=True)
        actual, actual_state = implementation.kda(*values, output_final_state=True, mode="project_chunk", allow_fallback=False)
        (actual.float().square().mean() + actual_state.square().mean()).backward()
        actual_grads = [item.grad.detach().clone() for item in values]
        expected_values = [item.detach().clone().requires_grad_(True) for item in values]
        expected, expected_state = oracle.recurrent_kda(*expected_values)
        (expected.float().square().mean() + expected_state.square().mean()).backward()
        for actual_grad, expected_value in zip(actual_grads, expected_values):
            torch.testing.assert_close(actual_grad, expected_value.grad, atol=config.correctness.gradient_atol, rtol=config.correctness.gradient_rtol)
        checks.append({"kind": "independent_output_and_state_backward", "length": 65, "head_dim": 16})

        q, k, v, gate, beta, A, dt = _inputs(torch, 4, config.correctness.production_head_dim, gradients=False)
        chunk, chunk_state = implementation.kda(q, k, v, gate, beta, A, dt, output_final_state=True, mode="project_chunk", allow_fallback=False)
        recurrent_outputs=[]; state=None
        with torch.no_grad():
            for index in range(q.shape[1]):
                out, state = implementation.kda(q[:, index:index+1], k[:, index:index+1], v[:, index:index+1], gate[:, index:index+1], beta[:, index:index+1], A, dt, initial_state=state, output_final_state=True, mode="project_recurrent", allow_fallback=False)
                recurrent_outputs.append(out)
        torch.testing.assert_close(torch.cat(recurrent_outputs, dim=1), chunk, atol=config.correctness.forward_atol, rtol=config.correctness.forward_rtol)
        torch.testing.assert_close(state, chunk_state, atol=config.correctness.forward_atol, rtol=config.correctness.forward_rtol)
        checks.append({"kind": "chunk_recurrent_equivalence", "length": 4})

        operator_recorder.__exit__(None,None,None)
        native_operator_events=operator_recorder.names; operator_recorder=None
        for name,component in components.items():
            if component["owner"]=="project" and not _operator_observed(component["torch_operator"],native_operator_events):
                raise RuntimeError(f"declared project CUDA operator did not execute for {name}: {component['torch_operator']}")
        events = implementation.project_runtime_events()
        observed = {event["component"] for event in events}
        missing = set(config.ownership.required_components) - observed
        if missing: raise RuntimeError(f"runtime audit did not observe components: {sorted(missing)}")
        route_violations=[]
        for event in events:
            claimed = components[event["component"]]["owner"] == "project"
            expected_backend = "project" if claimed else "fla"
            if event["backend"] != expected_backend:
                route_violations.append({**event, "expected_backend": expected_backend})
        if route_violations: raise RuntimeError(f"component ownership route violations: {route_violations}")
        owned_fraction=float(provenance.get("owned_fraction", 0.0))
        if lane == "bootstrap" and owned_fraction < config.bootstrap.minimum_owned_fraction:
            raise RuntimeError("bootstrap must implement at least one weighted native CUDA capability")
        runtime_fla_free=not any(event["backend"] == "fla" for event in events)
        core_owned=all(components[name]["owner"] == "project" for name in config.ownership.majority_core_components)
        migration_ready=owned_fraction >= config.ownership.minimum_project_owned_fraction and core_owned and runtime_fla_free
        if lane == "optimization" and not migration_ready:
            raise RuntimeError("optimization requires majority/core ownership and zero runtime FLA")
    except Exception as error:
        return {"status": "invalid", "reason": f"{type(error).__name__}: {error}", "lane": lane, "provenance": provenance,
                "runtime_forbidden_module_attempts": finder.attempts, "checks": checks}
    finally:
        if operator_recorder is not None: operator_recorder.__exit__(None,None,None)
        implementation._reference_kda = original_reference
        sys.meta_path.remove(finder)
    loaded_forbidden = sorted(name for name in sys.modules if any(name == prefix or name.startswith(prefix + ".") for prefix in config.ownership.forbid_runtime_modules))
    return {"status": "complete", "lane": lane, "runtime_fla_free": runtime_fla_free,
            "loaded_forbidden_modules": loaded_forbidden, "runtime_forbidden_module_attempts": finder.attempts,
            "runtime_events": events, "native_operator_events":sorted(set(native_operator_events)), "provenance": provenance, "checks": checks, "migration_ready": migration_ready}

def sanitizer_smoke(config, lane: str) -> dict[str, Any]:
    import torch
    implementation = importlib.import_module("nanochat.mixers.kda")
    if not torch.cuda.is_available(): return {"status": "invalid", "reason": "CUDA is required"}
    try:
        implementation.prepare_kda_backend("project_cuda")
        conv_x=torch.randn(1,65,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        conv=implementation.ShortConvolution(128,4).cuda()
        conv_output,_=conv(conv_x,output_final_state=True,backend="project_cuda")
        conv_output.float().sum().backward(); torch.cuda.synchronize()
        values = _inputs(torch, 65, config.correctness.production_head_dim, gradients=True)
        output, state = implementation.kda(*values, output_final_state=True, mode="project_chunk", allow_fallback=False)
        output.float().sum().backward(); torch.cuda.synchronize()
        with torch.no_grad():
            implementation.kda(*(item.detach() for item in _inputs(torch, 1, config.correctness.production_head_dim, gradients=False)),
                               initial_state=state.detach(), output_final_state=True, mode="project_recurrent", allow_fallback=False)
        torch.cuda.synchronize()
        return {"status": "complete", "lane": lane, "checks": ["causal_convolution_forward_backward_tail", "chunk_forward_backward_tail", "recurrent_nonzero_state"]}
    except Exception as error:
        return {"status": "invalid", "reason": f"{type(error).__name__}: {error}"}


def profile_audit(root: Path, config, lane: str) -> dict[str, Any]:
    import torch
    implementation=importlib.import_module("nanochat.mixers.kda")
    if not torch.cuda.is_available(): return {"status":"invalid","reason":"CUDA is required"}
    try:
        implementation.prepare_kda_backend("project_cuda")
        provenance=_provenance(root,implementation,config)
        if provenance.get("status")!="complete": raise RuntimeError(provenance.get("reason","invalid provenance"))
        length=min(65,config.bootstrap.maximum_sequence_length if lane=="bootstrap" else config.migration.maximum_sequence_length if lane=="migration" else 65)
        values=_inputs(torch,length,config.correctness.production_head_dim,gradients=True)
        conv=implementation.ShortConvolution(128,4).cuda(); conv_x=torch.randn(1,length,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        recurrent_values=_inputs(torch,1,config.correctness.production_head_dim,gradients=False)
        activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
            output,state=implementation.kda(*values,output_final_state=True,mode="project_chunk",allow_fallback=False)
            (output.float().square().mean()+state.square().mean()).backward()
            conv_output,_=conv(conv_x,output_final_state=True,backend="project_cuda"); conv_output.float().sum().backward()
            with torch.no_grad(): implementation.kda(*recurrent_values,output_final_state=True,mode="project_recurrent",allow_fallback=False)
        all_names=[]
        for event in prof.events():
            for value in (getattr(event,"name",None),getattr(event,"key",None)):
                if value: all_names.append(str(value))
        missing_ops=[]; missing_symbols=[]
        for name,component in provenance["components"].items():
            if component["owner"]!="project": continue
            if not _operator_observed(component["torch_operator"],all_names): missing_ops.append({"component":name,"operator":component["torch_operator"]})
            for symbol in component["kernel_symbols"]:
                if not any(symbol in observed for observed in all_names): missing_symbols.append({"component":name,"symbol":symbol})
        if missing_ops or missing_symbols: raise RuntimeError(f"profile does not bind claimed operators/symbols: operators={missing_ops}, symbols={missing_symbols}")
        rows=[{"name":item.key,"calls":int(item.count),"self_device_time_us":float(item.self_device_time_total)} for item in sorted(prof.key_averages(),key=lambda row:row.self_device_time_total,reverse=True)[:config.kernel_gates.profile_rows]]
        payload={"status":"complete","lane":lane,"profile_mode":"bounded_key_averages_no_trace","profile":rows,"observed_project_operators":[component["torch_operator"] for component in provenance["components"].values() if component["owner"]=="project"],"observed_kernel_symbols":[symbol for component in provenance["components"].values() if component["owner"]=="project" for symbol in component["kernel_symbols"]]}
        if len(json.dumps(payload,sort_keys=True).encode())>config.kernel_gates.profile_max_bytes: raise RuntimeError("bounded profile exceeds profile_max_bytes")
        return payload
    except Exception as error:
        return {"status":"invalid","reason":f"{type(error).__name__}: {error}","lane":lane}


def _time_call(torch, function, warmups: int, iterations: int) -> dict[str, Any]:
    for _ in range(warmups): function()
    torch.cuda.synchronize(); values=[]
    for _ in range(iterations):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); function(); end.record(); end.synchronize(); values.append(float(start.elapsed_time(end)))
    return {"median_ms": statistics.median(values), "values_ms": values,
            "coefficient_of_variation": statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0}


def microbenchmark(config, backend: str, lane: str) -> dict[str, Any]:
    import torch
    implementation = importlib.import_module("nanochat.mixers.kda")
    if not torch.cuda.is_available(): return {"status": "invalid", "reason": "CUDA is required"}
    if backend not in {"reference", "fla_triton", "project_cuda"}: return {"status": "invalid", "reason": f"unknown backend {backend}"}
    rows=[]
    try:
        if backend == "project_cuda": implementation.prepare_kda_backend("project_cuda")
        elif backend == "fla_triton": implementation.prepare_kda_backend("fla_triton")
        if lane == "bootstrap":
            limit, timed = config.bootstrap.maximum_sequence_length, config.bootstrap.timed_iterations
        elif lane == "migration":
            limit, timed = config.migration.maximum_sequence_length, config.migration.timed_iterations
        elif lane == "optimization":
            limit, timed = config.measurement.sequence_length, config.kernel_gates.timed_iterations
        elif lane == "anchor":
            limit, timed = config.reporting.canonical_length, config.bootstrap.timed_iterations
        else:
            return {"status": "invalid", "reason": f"unknown microbenchmark lane {lane}"}
        lengths=sorted(set(length for length in config.kernel_gates.sequence_lengths if length <= limit) | {1, config.reporting.canonical_length if config.reporting.canonical_length <= limit else limit})
        mode_prefix = "project_" if backend == "project_cuda" else ""
        oracle = _load_oracle(Path.cwd()) if backend == "reference" else None
        for length in lengths:
            values=_inputs(torch, length, config.correctness.production_head_dim, gradients=False)
            if backend == "reference":
                def forward(values=values): implementation.kda(*values, mode="reference", allow_fallback=False)
            else:
                def forward(values=values, length=length): implementation.kda(*values, mode=mode_prefix + ("recurrent" if length == 1 else "chunk"), allow_fallback=False)
            timing=_time_call(torch, forward, config.kernel_gates.warmup_iterations, timed)
            rows.append({"operation": "recurrent_decode" if length == 1 else "chunk_forward", "length": length, **timing})
            if length > 1:
                def forward_backward(length=length):
                    grad_values=_inputs(torch,length,config.correctness.production_head_dim,gradients=True)
                    mode="reference" if backend == "reference" else mode_prefix+"chunk"
                    output,_=implementation.kda(*grad_values,mode=mode,allow_fallback=False); output.float().square().mean().backward()
                timing=_time_call(torch,forward_backward,max(1,config.kernel_gates.warmup_iterations//2),timed)
                rows.append({"operation":"chunk_forward_backward","length":length,**timing})

            channels=128; conv_x=torch.randn(1,length,channels,device="cuda",dtype=torch.bfloat16)
            conv_weight=torch.randn(channels,4,device="cuda",dtype=torch.bfloat16)
            if backend == "reference":
                def conv_forward(conv_x=conv_x,conv_weight=conv_weight): oracle.causal_depthwise_conv(conv_x,conv_weight)
            else:
                conv=implementation.ShortConvolution(channels,4).cuda()
                with torch.no_grad(): conv.weight.copy_(conv_weight.float().unsqueeze(1))
                selected="project_cuda" if backend == "project_cuda" else "fla_triton"
                def conv_forward(conv=conv,conv_x=conv_x,selected=selected): conv(conv_x,output_final_state=False,backend=selected)
            timing=_time_call(torch,conv_forward,config.kernel_gates.warmup_iterations,timed)
            rows.append({"operation":"causal_convolution_forward","length":length,**timing})
            if length > 1:
                if backend == "reference":
                    def conv_backward(length=length):
                        x=torch.randn(1,length,channels,device="cuda",dtype=torch.bfloat16,requires_grad=True)
                        weight=torch.randn(channels,4,device="cuda",dtype=torch.bfloat16,requires_grad=True)
                        output,_=oracle.causal_depthwise_conv(x,weight); output.float().square().mean().backward()
                else:
                    def conv_backward(length=length,selected=selected):
                        module=implementation.ShortConvolution(channels,4).cuda(); x=torch.randn(1,length,channels,device="cuda",dtype=torch.bfloat16,requires_grad=True)
                        output,_=module(x,output_final_state=False,backend=selected); output.float().square().mean().backward()
                timing=_time_call(torch,conv_backward,max(1,config.kernel_gates.warmup_iterations//2),timed)
                rows.append({"operation":"causal_convolution_forward_backward","length":length,**timing})

        profile=[]
        if backend != "reference":
            profile_length=max(lengths)
            values=_inputs(torch,profile_length,config.correctness.production_head_dim,gradients=True)
            activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
                output,_=implementation.kda(*values,mode=mode_prefix+"chunk",allow_fallback=False); output.float().square().mean().backward()
            for item in sorted(prof.key_averages(),key=lambda row:row.self_device_time_total,reverse=True)[:config.kernel_gates.profile_rows]:
                profile.append({"name":item.key,"calls":int(item.count),"self_device_time_us":float(item.self_device_time_total)})
            if config.kernel_gates.require_profile and not profile: raise RuntimeError("kernel profile has no events")
        return {"status":"complete","backend":backend,"lane":lane,"profile_mode":"bounded_key_averages_no_trace" if profile else "bounded_cuda_events_no_profile","microbenchmarks":rows,"profile":profile}
    except Exception as error:
        return {"status":"invalid","reason":f"{type(error).__name__}: {error}","backend":backend,"lane":lane,"microbenchmarks":rows}

def main(argv=None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("command", choices=("runtime-audit", "sanitizer-smoke", "profile-audit", "microbenchmark"))
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True); parser.add_argument("--backend", choices=("reference", "fla_triton", "project_cuda"), default="project_cuda"); parser.add_argument("--lane", choices=("bootstrap", "migration", "optimization", "anchor"), default="optimization"); parser.add_argument("--implementation-root", default=None)
    args=parser.parse_args(argv)
    if args.implementation_root:
        import nanochat
        implementation_package=str((Path(args.implementation_root).resolve()/"nanochat"))
        if implementation_package not in nanochat.__path__: nanochat.__path__.insert(0,implementation_package)
    config_path=Path(args.config)
    config=(cuda_campaign_config_from_dict(json.loads(config_path.read_text())) if config_path.suffix == ".json" else load_cuda_campaign_config(config_path))
    root=Path.cwd().resolve()
    if args.command == "runtime-audit": result = runtime_audit(root, config, args.lane)
    elif args.command == "sanitizer-smoke": result = sanitizer_smoke(config, args.lane)
    elif args.command == "profile-audit": result = profile_audit(root, config, args.lane)
    else: result = microbenchmark(config, args.backend, args.lane)
    _write(args.output, result)
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
