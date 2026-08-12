# NVIDIA DGX Spark / GB10 architecture notes

Last researched: 2026-08-12

## Purpose

This note summarizes the public architecture of NVIDIA DGX Spark and its GB10
Grace Blackwell Superchip, with emphasis on CUDA kernel design for this
repository's KDA workload. It separates published facts, locally observed
device properties, and engineering inferences.

NVIDIA does not currently appear to publish a GB10 silicon whitepaper with the
microarchitectural depth available for datacenter Blackwell. The best public
primary sources are NVIDIA's DGX Spark hardware and porting guides, CUDA's
compute-capability and PTX documentation, CUTLASS documentation, and NVIDIA's
own DGX Spark tuning articles. Consequently, details such as GB10 die layout,
transistor count, GPU L2 size, SM subpartition layout, per-datatype dense Tensor
Core peaks, and internal fabric bandwidth should be treated as undisclosed.
They must not be copied from B200 or RTX 5090 specifications.

## Executive summary

GB10 is not a reduced-clock B200. It belongs to a different Blackwell CUDA
family:

- DGX Spark's integrated GPU is compute capability **12.1**, with **48 SMs**,
  **6,144 CUDA cores**, fifth-generation Tensor Cores, and fourth-generation RT
  Cores.
- The GPU and 20-core Arm CPU share **128 GB LPDDR5X** through a dynamic,
  hardware-coherent unified-memory architecture. The published peak bandwidth
  is **273 GB/s**, shared by the system rather than dedicated HBM bandwidth for
  the GPU alone.
- SM 12.x exposes at most **100 KiB shared memory per SM**, **99 KiB opt-in
  shared memory per block**, **64K 32-bit registers per SM**, and **48 resident
  warps / 1,536 resident threads per SM**.
- SM 12.x supports TMA, thread-block clusters, distributed shared memory,
  hardware `memcpy_async`, BF16, FP8, FP6, and FP4 Tensor Core inputs. It does
  **not** expose the SM100/SM110 Tensor Memory and `tcgen05` programming model.
- NVIDIA's own cross-platform example tunes the same FMHA source to a 64x64
  tile with occupancy 2 on 48-SM Spark, versus a 256x128 tile with occupancy 1
  on 148-SM B200. That is unusually direct evidence that Spark wants a
  different program shape, not merely a recompiled datacenter kernel.

For KDA, the likely opportunity is therefore a GB10-specific dataflow that
keeps useful same-group state near its consumer, minimizes LPDDR5X round trips,
uses smaller tiles with enough blocks to occupy 48 SMs, and stays below sharp
register/shared-memory residency cliffs. The opportunity is not access to
B200's Tensor Memory instructions; GB10 does not provide them.

## System organization

```text
                     GB10 Grace Blackwell SoC
    +--------------------------------------------------------+
    |                                                        |
    |  10x Cortex-X925          10x Cortex-A725              |
    |  performance cores        efficiency cores             |
    |          \                    /                         |
    |           +---- coherent system fabric ----+           |
    |                                             |           |
    |                            Blackwell GPU, SM 12.1       |
    |                            48 SMs / 6,144 CUDA cores    |
    |                            5th-gen Tensor Cores         |
    |                                             |           |
    +-------------------------+-------------------+-----------+
                              |
                  128 GB LPDDR5X unified memory
                 256-bit interface, 273 GB/s peak

    System I/O: ConnectX-7, 10 GbE, Wi-Fi 7, NVMe, USB-C, HDMI
```

The drawing is conceptual. NVIDIA publishes the coherent UMA behavior but not
the GB10 internal fabric topology or bandwidth in the reviewed material.

## Published and observed specifications

| Property | Value | Status |
| --- | ---: | --- |
| GPU compute capability | 12.1 (`sm_121`) | NVIDIA + local |
| GPU SM count | 48 | NVIDIA + local |
| CUDA cores | 6,144 | NVIDIA |
| CUDA cores per SM | 128 | Derived from the two rows above |
| Tensor Cores | Fifth generation | NVIDIA |
| RT Cores | Fourth generation | NVIDIA |
| CPU | 10 Cortex-X925 + 10 Cortex-A725, Arm v9.2 | NVIDIA + local |
| Unified memory | 128 GB LPDDR5X | NVIDIA |
| Local CUDA-visible capacity | 130,663,165,952 bytes (121.69 GiB) | Local, 2026-08-12 |
| Memory interface | 256 bit, 16 channels, LPDDR5X-8533 | NVIDIA |
| Peak memory bandwidth | 273 GB/s | NVIDIA theoretical specification |
| SM count / published example frequency | 48 / 2.14 GHz | NVIDIA tuning blog workload |
| Local reported maximum SM clock | 3.003 GHz | Local `nvidia-smi`, not a sustained-clock claim |
| Registers per SM | 65,536 x 32-bit | CUDA 12.x + local |
| Maximum registers per thread | 255 x 32-bit | CUDA 12.x |
| Shared memory per SM | 100 KiB | CUDA 12.x + local |
| Default / opt-in shared memory per block | 48 KiB / 99 KiB | CUDA 12.x + local |
| Maximum resident warps / threads per SM | 48 / 1,536 | CUDA 12.x + local |
| Maximum resident blocks per SM | 24 | CUDA 12.x architectural limit |
| Warp size / maximum threads per block | 32 / 1,024 | CUDA 12.x |
| GB10 SoC TDP | 140 W | NVIDIA |
| External system power supply | 240 W | NVIDIA |
| Advertised AI peak | 1 PFLOP FP4 with sparsity | NVIDIA; not a BF16 training peak |

The local observations came from PyTorch device properties, `lscpu`, and
`nvidia-smi`. They describe this repository's host and should be captured again
for any conclusion-bearing benchmark because firmware, clocks, thermals, and
software versions can change.

## CPU complex

The CPU is a heterogeneous 20-core Arm v9.2 design, not the 72-core Arm
Neoverse V2 Grace CPU used in datacenter Grace products. NVIDIA documents two
clusters, each containing five Cortex-X925 performance cores and five
Cortex-A725 efficiency cores:

- Each X925 has 2 MiB L2; each A725 has 512 KiB L2.
- The performance-core cluster has 16 MiB L3 and the efficiency-core cluster
  has 8 MiB L3.
- The local host reports 20 single-threaded cores, 25 MiB aggregate L2 and two
  L3 instances totaling 24 MiB.
- The Arm memory model is weaker than x86's. Correct multithreaded host code
  must use C/C++ atomics and barriers rather than relying on x86 ordering.

For GPU kernels, the practical consequence is mostly launch and orchestration
behavior. A many-kernel CUDA pipeline is driven by an Arm host with asymmetric
cores, not a large server CPU. Launch-count reductions can help, but only when
they do not destroy GPU locality or enlarge kernels enough to lose residency.
The CUDA KDA campaign has already observed both sides of this tradeoff.

## Unified memory is an architectural advantage and a bandwidth constraint

The CPU and integrated GPU share one physical LPDDR5X pool without a fixed GPU
carve-out. CUDA describes Grace Blackwell-class systems as providing full
Unified Memory with hardware coherence and Address Translation Services. This
allows the GPU to access pageable host allocations and permits CPU/GPU sharing
without the explicit PCIe copies required by a conventional discrete GPU.

This should not be confused with infinite-speed memory:

- The published **273 GB/s is the peak bandwidth of the shared memory
  interface**. CPU traffic, GPU traffic, the OS page cache, and other engines
  all draw on the same physical memory system.
- B200's published comparison point is roughly **8 TB/s HBM3e**, about 29 times
  Spark's peak bandwidth. Algorithms and tile sizes optimized around that HBM
  system need not be optimal on GB10.
- `nvidia-smi` does not report a normal discrete framebuffer on this UMA
  platform. `cudaMemGetInfo` also omits potentially reclaimable page-cache and
  swap-backed memory, so it is not a complete measure of allocatable capacity.
- Hardware coherence removes mandatory copies; it does not make repeated
  global materialization of KDA intermediates free. Avoiding a byte of
  LPDDR5X traffic may be more valuable on Spark than on an HBM datacenter GPU.

Large model capacity is the main system-level benefit. Kernel optimization is
still governed by locality, bandwidth, page behavior, and contention.

## GPU execution resources

### Occupancy cliffs are unusually important

SM 12.x has a 128 KiB unified data cache, of which at most 100 KiB can be
configured as shared memory. That is materially less shared-memory capacity
than the 228 KiB class available on SM 9.0/10.x datacenter parts.

Simple residency arithmetic illustrates the constraint:

- A block using more than 50 KiB shared memory cannot have two such blocks
  resident on a 100 KiB SM.
- A 256-thread block using 128 registers per thread consumes 32,768 registers;
  two blocks consume the complete 65,536-register SM pool before allocation
  granularity and other constraints are considered.
- A 256-thread, 59 KiB-shared block may therefore be one-block-per-SM even if
  it has enough threads and registers for more occupancy.

These are ceilings, not performance predictions. One block per SM can be
correct for a large, efficient kernel. But on a 48-SM device, oversized tiles
also reduce the number of independent CTAs available to fill the machine.

### GB10 supports TMA, but not Tensor Memory

CUDA's compute-capability table lists the following for SM 12.x:

- Tensor Memory Accelerator (TMA)
- thread-block clusters and distributed shared memory
- hardware-accelerated `memcpy_async` and split arrive/wait barriers
- L2 cache residency management
- Tensor Core inputs including TF32, BF16, FP16, FP8, FP6, FP4, INT8, and INT4

TMA and Tensor Memory are different features. The PTX target notes restrict
`tcgen05.alloc`, Tensor Memory loads/stores, and the corresponding fifth-gen
MMA programming model to the SM100/SM110 families; SM120/SM121 is not listed.
Compiling for `sm_121a` enables GB10's architecture-specific features but does
not manufacture SM100 Tensor Memory hardware.

The useful low-level matrix path for the current BF16 KDA kernels is therefore
the SM120/SM121 MMA/CUTLASS family, including the classic warp-level mechanisms
already used by the project. CUTLASS supports DGX Spark from CUDA 13.0 and now
contains SM120/SM121-specific kernels and smaller tile shapes. Datacenter
`sm_100a` kernels are not binary-compatible with SM121.

### Compilation targets

CUDA 13 recognizes baseline, family, and architecture-specific GB10 targets:

- `sm_121`: baseline 12.1 feature set
- `sm_120f` / `sm_121f`: family-specific features; `120f` is compatible with
  both compute capability 12.0 and 12.1
- `sm_121a`: architecture-specific 12.1 features, runnable only on 12.1

The project's use of CUDA 13.1 `ptxas` targeting `sm_121a` is therefore the
right mechanism for GB10 specialization. Every instruction still needs its
own PTX target-ISA support check.

## GB10 versus datacenter Blackwell

| Characteristic | DGX Spark / GB10 | B200 comparison used by NVIDIA |
| --- | ---: | ---: |
| CUDA architecture | SM 12.1 | SM 10.0 |
| SM count | 48 | 148 |
| Memory | Unified LPDDR5X | Dedicated HBM3e |
| Peak memory bandwidth | 273 GB/s | About 8 TB/s |
| Shared memory ceiling per SM | 100 KiB | 228 KiB class for SM 10.x |
| Tensor Memory / `tcgen05` | No | Yes |
| NVIDIA FMHA example tile | 64x64, occupancy 2 | 256x128, occupancy 1 |

The frequency values in NVIDIA's comparison were workload/platform snapshots,
not fixed architectural frequencies, so they are omitted from the contrast.

The table explains why “FLA targets broad Blackwell” cuts both ways. Triton
already specializes code generation for the running architecture, so FLA is
not simply executing a B200 binary on Spark. However, a broadly useful operator
may retain program shapes, intermediate contracts, and scheduling choices that
are sensible across many GPUs. A project-owned GB10 kernel can specialize those
algorithmic choices more aggressively—provided it respects GB10's smaller
shared-memory budget and much lower global-memory bandwidth.

## Implications for KDA optimization

### Directions supported by the architecture

1. **Design around bytes, not only FLOPs.** Measure every forward/backward
   intermediate written to LPDDR5X. Fusing A/T/U/W production or retaining a
   compact value until its same-group consumer can be more valuable than
   eliminating a few arithmetic instructions.

2. **Preserve immediate producer-consumer locality.** The architecture favors
   reducing global round trips, but the campaign showed that deferring work
   across KDA groups can evict useful data and lose more than launch batching
   saves. A new program shape should remain group-local unless profiling proves
   otherwise.

3. **Use smaller tiles and enough CTAs.** NVIDIA's own 64x64/occupancy-2 Spark
   example is a strong prior. KDA's exact dimensions still need measurement,
   but B200-sized tiles should not be the default assumption.

4. **Budget registers and shared memory together.** Report registers/thread,
   shared bytes/block, stack/local bytes, active CTAs/SM, and achieved warps for
   every candidate. A 2% instruction reduction that crosses a residency cliff
   is unlikely to survive end-to-end timing.

5. **Keep BF16 Tensor Cores busy through supported MMA paths.** GB10 supports
   BF16 Tensor Core inputs. The retained project's BF16 WMMA transformations
   are aligned with the available hardware; a `tcgen05`/TMEM port is not.

6. **Treat TMA as a measured option, not a conclusion.** TMA exists on SM121
   and may reduce address-generation or copy overhead for regular tiles. The
   earlier FLA TMA experiment was neutral at the protected KDA shape, so a new
   use needs a specific data-movement hypothesis and isolated evidence.

7. **Account for host and thermal state.** Record system software, clocks,
   temperature, throttling/power counters, and ordering of matched runs. The
   compact 140 W SoC shares power and memory resources across CPU, GPU, and
   system activity; drift can be comparable to sub-percent kernel gains.

### Directions not supported by current evidence

- Assuming that datacenter Blackwell's TMEM or `tcgen05` is available on GB10.
- Enlarging shared-memory tiles without calculating the 100 KiB/SM residency
  result.
- Treating the advertised sparse FP4 petaFLOP as BF16 training throughput.
- Treating unified memory as dedicated GPU HBM or using B200 bandwidth roofs.
- Inferring a trainer win from launch count, isolated convolution time, or
  microkernel time alone.
- Copying RTX 5090 or B200 cache sizes, instruction availability, or throughput
  numbers into the GB10 model because all three carry the Blackwell name.

## Recommended research before another kernel campaign

The missing information is best filled empirically on the exact machine:

1. Build a GB10 roofline sheet for the production KDA shapes:
   device read/write/copy bandwidth, BF16 MMA throughput, FP32 throughput,
   shared-memory bandwidth/bank conflicts, and launch latency.
2. Compare scalar/vector loads, `cp.async`, and TMA for the exact 64-token KDA
   tiles and alignments rather than a generic GEMM.
3. Profile attempt342 and the pinned FLA reference in one interleaved session.
   Attribute bytes, duration, occupancy, registers, shared memory, and stalls by
   forward A/T/U/W production, state scan, complete/local VJP, convolution, and
   framework-common work.
4. Produce a candidate dataflow diagram and lifetime table before coding. For
   each intermediate, record precision, bytes, producer, first/last consumer,
   whether it crosses a group boundary, and proposed storage level.
5. Set a new fixed objective large enough to exceed machine drift, then require
   microbenchmark, matched-operator, and ordered trainer gates. Do not optimize
   against a moving contemporaneous reference alone.

This work would turn the plausible “GB10-specific kernels should beat a broad
library” thesis into a falsifiable resource and dataflow hypothesis.

## Primary sources

- [NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark Porting Guide: System Overview](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)
- [NVIDIA DGX Spark Porting Guide: Optimization](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html)
- [CUDA Programming Guide: Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- [PTX ISA: TensorCore fifth-generation instructions and target notes](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Compiler Driver: supported SM 12.1 targets](https://docs.nvidia.com/cuda/archive/13.1.2/cuda-compiler-driver-nvcc/index.html)
- [NVIDIA CUTLASS documentation](https://docs.nvidia.com/cutlass/latest/overview.html)
- [NVIDIA Technical Blog: Scaling Autonomous AI Agents and Workloads with DGX Spark](https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/)
- [NVIDIA DGX Spark product page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)

