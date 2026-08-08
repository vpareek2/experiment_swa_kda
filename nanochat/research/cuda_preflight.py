"""Native SM121 CUDA toolchain preflight using a real registered hello op."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from nanochat.research.cuda_build import build_cuda_extension

_HELLO_CUDA = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

__global__ void nanochat_sm121_hello_kernel(
    const float* input, float* output, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index] + 1.0f;
  }
}

at::Tensor nanochat_sm121_hello_cuda(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "nanochat SM121 hello expects a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float,
              "nanochat SM121 hello expects float32");
  TORCH_CHECK(input.is_contiguous(), "nanochat SM121 hello expects contiguous input");
  auto output = at::empty_like(input);
  const int64_t count = input.numel();
  if (count != 0) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    nanochat_sm121_hello_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.const_data_ptr<float>(), output.mutable_data_ptr<float>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

TORCH_LIBRARY(nanochat_cuda_preflight, m) {
  m.def("hello(Tensor input) -> Tensor");
}
TORCH_LIBRARY_IMPL(nanochat_cuda_preflight, CUDA, m) {
  m.impl("hello", &nanochat_sm121_hello_cuda);
}
"""


def _build_hello(cache_dir: Path | None, verbose: bool = False) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nanochat-sm121-preflight-source-") as temporary:
        source = Path(temporary) / "hello_sm121.cu"
        source.write_text(_HELLO_CUDA, encoding="utf-8")
        return build_cuda_extension(
            [source],
            name="nanochat_sm121_preflight",
            cache_dir=cache_dir,
            extra_cuda_cflags=("-lineinfo",),
            verbose=verbose,
        )


def _profile_child(cache_dir: Path | None) -> None:
    import torch

    _build_hello(cache_dir)
    value = torch.arange(4097, device="cuda", dtype=torch.float32)
    torch.ops.nanochat_cuda_preflight.hello(value)
    torch.cuda.synchronize()


def nsys_profile_command(command: Sequence[str], report_prefix: str | Path) -> list[str]:
    """Construct the fixed, low-overhead Nsight Systems CUDA profile command."""
    if not command:
        raise ValueError("a command to profile is required")
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("nsys is required for independent CUDA kernel symbol evidence")
    return [
        nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
        "--force-overwrite=true", "--output", str(Path(report_prefix)),
        *map(str, command),
    ]


def read_nsys_kernel_symbols(report_path: str | Path, *, timeout: float = 180.0) -> list[str]:
    """Return GPU kernel names from an Nsight ``.nsys-rep`` artifact."""
    if timeout <= 0:
        raise ValueError("Nsight stats timeout must be positive")
    report = Path(report_path)
    if not report.is_file():
        raise FileNotFoundError(report)
    nsys = shutil.which("nsys")
    if nsys is None:
        raise RuntimeError("nsys is required to inspect CUDA kernel symbol evidence")
    stats = subprocess.run(
        [nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(report)],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if stats.returncode != 0:
        raise RuntimeError(
            "nsys could not read the CUDA profile: " + (stats.stdout + stats.stderr).strip()
        )
    symbols: list[str] = []
    name_index: int | None = None
    for row in csv.reader(stats.stdout.splitlines()):
        normalized = [column.strip().strip('"') for column in row]
        if "Name" in normalized:
            name_index = normalized.index("Name")
            continue
        if name_index is not None and len(row) > name_index and row[name_index].strip():
            symbols.append(row[name_index].strip())
    if not symbols:
        raise RuntimeError("Nsight profile contains no CUDA GPU kernel symbols")
    return sorted(dict.fromkeys(symbols))


def capture_nsys_cuda_symbols(
    command: Sequence[str],
    *,
    expected_symbols: Sequence[str],
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 180.0,
) -> dict[str, list[str]]:
    """Profile ``command`` and require each requested GPU kernel symbol."""
    requested = tuple(dict.fromkeys(expected_symbols))
    if not command or not requested or any(not symbol for symbol in requested):
        raise ValueError("a command and non-empty expected CUDA symbols are required")
    if timeout <= 0:
        raise ValueError("Nsight profiling timeout must be positive")
    with tempfile.TemporaryDirectory(prefix="nanochat-sm121-nsys-") as temporary:
        report_prefix = Path(temporary) / "profile"
        profiled = subprocess.run(
            nsys_profile_command(command, report_prefix),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        if profiled.returncode != 0:
            raise RuntimeError(
                "nsys failed to profile CUDA kernels: "
                + (profiled.stdout + profiled.stderr).strip()
            )
        symbols = read_nsys_kernel_symbols(
            report_prefix.with_suffix(".nsys-rep"), timeout=timeout
        )
        matches = {symbol: [name for name in symbols if symbol in name] for symbol in requested}
        missing = [symbol for symbol, found in matches.items() if not found]
        if missing:
            raise RuntimeError(f"nsys did not observe required CUDA kernels: {missing}")
        return matches


def _nsys_kernel_symbols(cache_dir: Path | None) -> list[str]:
    command = [sys.executable, "-m", "nanochat.research.cuda_preflight", "--_profile-child"]
    if cache_dir is not None:
        command += ["--cache-dir", str(cache_dir)]
    return capture_nsys_cuda_symbols(
        command,
        expected_symbols=("nanochat_sm121_hello_kernel",),
    )["nanochat_sm121_hello_kernel"]


def sanitizer_zero_summary(tool:str,evidence:str)->bool:
    if tool=="racecheck":
        return "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in evidence
    return "ERROR SUMMARY: 0 errors" in evidence


def run_cuda_preflight_sanitizers(*,cache_dir:str|Path|None=None,timeout:float=300.0)->dict[str,Any]:
    executable=shutil.which("compute-sanitizer")
    if executable is None: raise RuntimeError("compute-sanitizer is required for CUDA preflight")
    results={}
    for tool in ("memcheck","racecheck","synccheck","initcheck"):
        command=[executable,"--tool",tool,"--error-exitcode=99",sys.executable,"-m","nanochat.research.cuda_preflight","--_profile-child"]
        if cache_dir is not None: command += ["--cache-dir",str(cache_dir)]
        completed=subprocess.run(command,text=True,capture_output=True,check=False,timeout=timeout)
        evidence=completed.stdout+completed.stderr
        passed=completed.returncode==0 and sanitizer_zero_summary(tool,evidence)
        results[tool]={"status":"complete" if passed else "invalid","returncode":completed.returncode,"zero_error_summary":sanitizer_zero_summary(tool,evidence)}
        if not passed: raise RuntimeError(f"CUDA preflight {tool} failed: {evidence[-4000:]}")
    return results


def run_cuda_preflight(
    *, cache_dir: str | Path | None = None, verbose: bool = False
) -> dict[str, Any]:
    """Compile, execute, and profile a visible SM121 CUDA kernel.

    Raises on any unavailable toolchain component or missing evidence; a
    successful return is therefore a usable supervisor preflight receipt.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native CUDA preflight")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"SM121 GPU required, found compute capability {capability[0]}.{capability[1]}")

    configured_cache = cache_dir if cache_dir is not None else os.environ.get("TORCH_EXTENSIONS_DIR")
    cache_root = Path(configured_cache).expanduser().resolve() if configured_cache else None
    build = _build_hello(cache_root, verbose=verbose)

    expected = torch.linspace(-2.0, 2.0, 4097, device="cuda", dtype=torch.float32) + 1.0
    input_tensor = expected - 1.0
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        actual = torch.ops.nanochat_cuda_preflight.hello(input_tensor)
        torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    symbols = sorted({event.key for event in trace.key_averages()})
    observed = [symbol for symbol in symbols if "nanochat_sm121_hello_kernel" in symbol]
    profiler_backend = "torch.profiler"
    if not observed:
        # Some CUDA/PyTorch combinations can execute SM121 code before Kineto
        # understands that architecture. Nsight Systems is the independent
        # profiler fallback; it launches a clean child and reports actual GPU
        # kernel activities rather than trusting a declared symbol.
        observed = _nsys_kernel_symbols(cache_root)
        profiler_backend = "nsys"
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "nanochat_cuda_preflight::hello", "CUDA"
    ):
        raise RuntimeError("hello operator has no registered CUDA implementation")

    return {
        "status": "complete",
        "device": torch.cuda.get_device_name(),
        "compute_capability": "12.1",
        "operator": "nanochat_cuda_preflight::hello",
        "kernel_symbol": "nanochat_sm121_hello_kernel",
        "profiler_backend": profiler_backend,
        "profiler_symbols": observed,
        "output_parity": True,
        "build": build,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sanitizers", action="store_true", help="run all four Compute Sanitizer tools")
    parser.add_argument("--_profile-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._profile_child:
        _profile_child(args.cache_dir)
        return 0
    result = run_cuda_preflight(cache_dir=args.cache_dir, verbose=args.verbose)
    if args.sanitizers: result["sanitizers"]=run_cuda_preflight_sanitizers(cache_dir=args.cache_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
