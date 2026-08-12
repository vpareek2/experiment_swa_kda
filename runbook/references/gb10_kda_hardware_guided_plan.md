# GB10-guided KDA optimization plan

Status: implementation audit complete on 2026-08-12. The audited parent is
`801165ac4a20c28d526078fc5ecce37c213c705d`, which contains integrated
attempt342. No candidate source change from this audit is retained.

This document turns the DGX Spark / GB10 architecture notes into an
implementation plan for the exact production KDA training call. Hardware facts
and primary-source links live in the companion
[DGX Spark / GB10 architecture note](dgx_spark_gb10_architecture.md).

## Decision

Attempt342 is already a strongly GB10-specific implementation. It narrowly
cleared the historical FLA-derived trainer target, and a fresh bounded operator
diagnostic can show a larger lead, but its GPU kernel sum remains effectively
tied with pinned FLA. There is no remaining launch, block-size, vectorization,
cache hint, TMA, or precision micro-tune with a credible end-to-end three-percent
ceiling.

The immediate recommendations are:

1. Keep attempt342 as the production and comparison baseline.
2. Do not merge the tested transient-`A` removal. It is bitwise exact but saved
   only 0.918% in the frozen production-shape Level-1 operation.
3. Preserve 64-token semantic chunks and eight-chunk reverse groups for the
   next mainline design. A direct FLA C32/C64 A/B on this GB10 made C32 1.50%
   slower and increased peak allocation by 6.18%.
4. Pursue a C64 same-group producer/consumer rewrite, then stack vertical
   convolution/KDA and output-normalization fusion only after the kernel-only
   rewrite proves a material operator saving.
5. Require at least 0.8--0.9 ms per production KDA invocation before spending a
   trainer run. Smaller changes cannot create a comfortable trainer lead.

The key shift is from tuning individual launches to deleting boundaries. The
remaining opportunity is in values that are produced, published to global
memory, and read again by an adjacent project-owned operation. Whether those
reads hit L2 or unified LPDDR is currently unmeasured.

## Define “comfortably beat FLA” before coding

The accepted attempt342 trainer median is 43,840 tok/s. The fixed historical
FLA-derived target is 43,680 tok/s, so the retained margin is only 0.366%.
At a 32,768-token optimizer update:

| Target | Throughput | Update time | Saving from attempt342 |
|---|---:|---:|---:|
| Attempt342 | 43,840 tok/s | 747.445 ms | — |
| 3% over fixed FLA | 44,990.4 tok/s | 728.333 ms | 19.112 ms |
| 3% over attempt342 | 45,155.2 tok/s | 725.675 ms | 21.770 ms |

The trainer executes 24 production KDA calls per update: six KDA layers over
four gradient-accumulation microsteps. The two targets therefore require about
0.796 ms and 0.907 ms per KDA call respectively.

For the next campaign, use all of these gates:

- Point estimate at least 3% faster than a contemporaneous, matched FLA 0.5.2
  trainer.
- Point estimate at least 2% faster than attempt342, with every stable candidate
  run above the fixed 43,680 tok/s target.
- Production forward+backward operator at most 3.45 ms, versus the fresh
  attempt342 Level-1 median of 4.343 ms. This is the early proxy for the
  0.8--0.9 ms per-call budget.
- Trainer peak allocation no more than 1.03 times attempt342.
- No architecture-quality claim until the separate matched discovery and
  promotion protocols run.

An isolated operator result is not enough. In one fresh interleaved diagnostic,
project CUDA measured 4.408 ms versus FLA at 4.970 ms, an 11.3% project lead.
The corresponding Nsight Systems captures summed to 4.069 and 4.057 ms of GPU
kernel time per call respectively. The difference is queueing, launch, allocator,
and system-state sensitive; only the matched trainer gate establishes a
comfortable system win.

## Current implementation audit

### Dispatch and specialization

`nanochat/mixers/kda.py` provides the protected semantic dispatcher and custom
autograd boundaries. The project CUDA training specialization is entered only
for the exact call:

```text
B=2, T=4096, H=3, K=128, V=128, BF16
lower_bound=-5, scale=1/sqrt(128)
no initial/final state, gradients enabled
```

Every other chunk shape, state-bearing call, inference call, and unsupported
contract retains the correctness-qualified generic path. Recurrent T=1 decode
uses its own native kernel. This exact-shape gate is useful: a new implementation
can target one immutable launch geometry without weakening fallback coverage.

The custom forward returns a visible 6 MiB BF16 output backed by a larger hidden
sidecar. Autograd saves the original seven inputs and the output backing, then
passes the sidecar to the native backward.

### Forward dataflow

Fresh SM121 resource dumps and a ten-call Nsight Systems profile give the
following representative production phases:

| Phase | Per-call time | Launch and resource shape | Main constraint |
|---|---:|---|---|
| Preprocess + A/M build + T solve | 0.699 ms | 384 CTAs × 1024 threads, REG40, 96,768 B dynamic shared | One CTA/SM; 8 waves over 48 SMs |
| U/W products | 0.156 ms | 6,144 CTAs × 128 threads, REG48, 7,680 B shared | Re-reads T for many product owners |
| H/Z recurrent state | 0.337 ms | 24 CTAs × 64 threads, REG254, 41,984 B shared | Dependency-carrying; only 24 SMs receive work |
| Output `qγH + AZ` | 0.178 ms | 384 CTAs × 128 threads, REG96, 33,792 B shared | Parallel, but begins after the state scan |

The fused factory holds FP16 qbar/khat and FP32 prefix state in shared memory,
builds stable 16×16 A/M products with BF16 tensor-core MMA, solves the 64-row
unit-lower system, and publishes A/T plus retained operands. Its 96,768-byte
(94.5 KiB) dynamic allocation plus 1,024 B static shared totals 95.5 KiB. That
sits just below GB10's 99-KiB per-block opt-in limit and prevents another factory
CTA from co-residing on the same SM.

The state kernel is already a GB10-specific register program. It keeps a
128×32 state strip in registers, stages W and restored E panels asynchronously,
and uses `ldmatrix`/`mma.sync`. Splitting state from output prevents the 24
dependency owners from serializing 384 independent output chunks.

### Backward dataflow

Backward walks eight reverse groups, each containing eight C64 chunks per
recurrence. The exact group order is a correctness and cache-locality boundary.

| Phase | Per-call time | Launch and resource shape | Main constraint |
|---|---:|---|---|
| Group U + qg/kg pack | 0.338 ms | 8 launches; 3,072 × 128-thread CTAs per group launch | Group-local vector work; eight launches cover the sequence |
| H/Z reconstruction + reverse dH | 0.399 ms | 8 × 48 CTAs, 160 threads, REG220, 54 KiB dynamic shared | Exactly one owner per SM |
| Complete local VJP | 0.791 ms | 8 × 48 CTAs, 256 threads, REG132, 29,696 B shared | 33,792 registers/CTA; one CTA/SM |
| Four colored stable-pair phases | 0.483 ms | 32 launches, REG48, 39,936 B shared, 16 B stack | Local triangular dependency colors |
| Finalize | 0.143 ms | 8 launches | Smaller but still material |

The complete VJP uses four owner and four helper warps. It keeps the local
adjoint live while helpers form dP and key products, then transforms dM and
publishes dA. The colored kernels apply the stable pair pullback in four exact
dependency colors. This is why simply fusing all backward work into a single
CTA or delaying one group has repeatedly lost locality and performance.

The fresh profile observed 101 project KDA kernel launches per core call. A
saved full-trainer attribution observed approximately 109 project launches
versus 33 FLA launches per KDA call, but also showed that launch reduction by
itself is not the missing mechanism.

### Retained and transient memory

The visible output owns this retained sidecar until backward:

| Surface | Size per KDA call |
|---|---:|
| Visible BF16 output | 6.000 MiB |
| FP32 group checkpoints | 2.625 MiB |
| BF16 A and T | 6.000 MiB |
| BF16 W, Q, qbar, khat, P | 30.000 MiB |
| FP16 prefix | 6.000 MiB |
| FP32 beta, q-inverse, k-inverse | 0.281 MiB |
| Total retained output backing | 50.906 MiB |

Forward transient storage adds qgamma (6 MiB), Z backing (12 MiB), Q backing
(12 MiB), M backing (6 MiB), transient A (3 MiB), and U (6 MiB). Including the
retained output, the forward working set is about 95.906 MiB. Backward adds
full incoming-H and Z histories plus one reverse group's local operands and
adjoints.

The 273 GB/s unified LPDDR fabric makes the worst case of unnecessary global
materialization relevant, but the capacity of GB10's GPU L2 is undisclosed and
nearby consumers may still hit cache. Compulsory tensor-core work also dominates
the largest phases. A traffic
optimization must remove a producer/consumer boundary, not merely exchange one
global surface for another retained surface. Nsight Compute counters are needed
before attributing a saving specifically to DRAM traffic.

### Convolution and decode

The width-four causal convolution already has a time-four GB10 specialization.
Saved full-trainer attribution measured the project convolution path faster
than FLA, so another convolution tile sweep is not a priority.

The recurrent T=1 decoder is underoptimized: every value lane redundantly
computes q/k norms and key-wise decay. Shared normalized q/k and decay would be
a straightforward decode-latency project, but it has no effect on the current
training-throughput objective and should be a separate campaign.

## Hardware interpretation

GB10 has 48 SMs, 100 KiB shared memory and 64K 32-bit registers per SM, a
1,536-thread/48-warp SM limit, and 273 GB/s unified LPDDR5X. SM121 provides TMA,
clusters, DSM, asynchronous copies, and conventional `mma.sync`, but not
datacenter Blackwell TMEM or `tcgen05`.

Attempt342 already reflects those facts:

- The forward factory uses almost the complete per-block shared-memory budget.
- The boundary and complete grids launch exactly 48 CTAs, one per SM.
- The state grid uses register-resident value strips and asynchronous panels.
- Direct BF16 U/W publication and compact group-major retained operands reduce
  global-memory traffic.
- The current convolution uses a Spark-sized time tile.

The remaining datacenter-versus-consumer opportunity is therefore not an
unused instruction family. It is a different program shape: smaller resident
tiles, two chunks resident per SM where dependencies permit, and direct
consumption of values while they are still in registers/shared memory.

Nsight Compute performance counters are presently unavailable on the host:
`ncu` returns `ERR_NVGPUCTRPERM`. Static resources and Nsight Systems are enough
to begin a bounded prototype, but an administrator should enable non-admin GPU
performance counters before roofline-driven tuning or claims about DRAM versus
tensor-core saturation.

## The simple implementation that was tested

The fused producer wrote the same rounded BF16 A matrix twice: once to a
recurrence-major 3 MiB transient used by forward output and once to the
group-major retained sidecar used by backward. A one-file candidate removed the
transient allocation/store and taught output to remap its chunk index into the
retained layout.

Results:

- Protected runtime and external profile audits passed with full project
  ownership and no runtime FLA.
- An independent production random-upstream capture was bitwise equal to
  attempt342 for output and all seven gradients.
- Frozen Level 1 measured T4096 forward+backward at 4.343440 ms for attempt342
  and 4.303584 ms for the candidate: a 0.918% improvement.
- Peak allocation remained 143,002,112 bytes because backward, not the removed
  forward allocation, is the peak.
- The required three-percent Level-1 gate failed. No trainer or sanitizer
  campaign was launched, and the patch was not applied to `main`.

This is the correct stop decision. Its ideal bandwidth ceiling was too small
to affect the trainer objective even though it was bitwise exact in the sampled
production comparison.

## Directions that are closed or too small

Do not restart these axes without new profile evidence:

- CUDA graphs, pointer-keyed replay, launch-only fusion, and programmatic
  dependent launch.
- Adjacent reverse-group batching or deferred complete VJP. Delaying even one
  group increased complete time materially through lost locality.
- Whole-group retained U/W/P/Q, retained Q/Z swaps, consumer recomputation, or
  wider global lifetimes.
- One giant forward/backward CTA, output-owned VJP variants, reduced-register
  occupancy changes without new work ownership, or shared factor caches.
- Cache carveout hints, host-only compiler flags, memset folding, allocation
  rearrangement, and vector-publication microvariants.
- Reduced-precision normalized/factor surfaces, ratio-factorized exponential
  paths, or FP16/BF16 exponent caches that have already crossed correctness
  envelopes.
- Cluster/DSM `ldmatrix`, which produced an illegal instruction on the
  production path, and TMA conversion without a measured producer/consumer
  saving.
- More convolution time/dweight tile sweeps.

Changing the semantic chunk size to 32 is also not the default next step. Pinned
FLA supports C32 and C64, so a directional A/B was run at the exact production
shape. C32 measured 4.911248 ms versus C64 at 4.838816 ms and raised peak
allocation from 271,571,968 to 288,349,184 bytes. A project-specific paired-C32
design could in theory avoid some of FLA's doubled boundary cost, but it now
needs an unusually strong forward-only proof before earning a full rewrite.

## Proposed design: C64 same-group superpipeline

The next kernel-only design should preserve C64 state/checkpoint semantics but
tile the local algebra at C16 inside each chunk. The purpose is to retain the
current numerical and cache boundaries while lowering per-CTA residency and
consuming T blocks before a global-memory round trip.

### Forward factory

Replace the 1024-thread, 95.5-KiB-total monolithic factory with a 512-thread streaming
C64 factory targeting at most 50 KiB total shared memory and at most 64
registers/thread. Two independent chunk CTAs should then fit on each GB10 SM.
This does not raise warp occupancy: two 512-thread CTAs still provide 1,024
threads and 32 warps, just like the current single 1,024-thread CTA. The intended
benefit is two resident chunks and independent-CTA/barrier latency hiding.

The factory should:

1. Preserve the current FP32 norm, sigmoid, prefix, and safe exponential
   expressions.
2. Process the 64×128 qbar/khat/prefix data as C16 row/key panels instead of
   retaining every full-chunk surface simultaneously in shared memory.
3. Build four diagonal 16×16 M blocks and six causal off-diagonal blocks with
   `mma.sync`/`ldmatrix`.
4. Apply a block unit-lower solve. Keep diagonal solve state in FP32 and use
   MMA for off-diagonal transforms. Publish final T with the same BF16 rounding
   boundary as attempt342.
5. As soon as a solved T row block is complete, multiply it by P and Q and
   publish U/W. This deletes the standalone U/W grid and avoids repeated full-T
   reads.
6. Publish A/T/W/Q/qbar/khat/P directly in the existing group-major backward
   order. Preserve C64 checkpoints and the current state/output kernels for the
   first prototype.

This is not the previously rejected implicit-M solve. The required product is
still explicit T; only its construction and consumption are blocked and
streamed. The candidate must be abandoned if it needs more than 50 KiB shared,
more than 64 registers/thread, or any stack/local spill.

Hard forward gate: fused factory plus U/W at most 0.60 ms, versus approximately
0.855 ms in the fresh profile. The standalone U/W phase is only about 0.156 ms,
so deleting it cannot meet this gate by itself; the rewrite must also cut roughly
0.10 ms, or about 14%, from the current 0.699-ms factory while making U/W nearly
free. A weaker result cannot fund the end-to-end goal.

### Backward local-VJP pipeline

The backward rewrite must attack complete plus colored work together, currently
about 1.274 ms per call, while preserving immediate boundary-to-complete
locality.

Design the dependency graph around C16 tiles inside one C64 chunk:

- Keep eight 512-token reverse groups and process the current group completely
  before launching the prior group.
- Split each chunk into two material roles only where both roles create useful
  grids. Do not lower registers on a 48-CTA grid and call that occupancy; two
  resident CTAs require at least 96 independent CTAs.
- Let value-product owners form dP/dQ/dR/dE/dW tiles, and let local-pair owners
  consume those tiles immediately in causal color order.
- Keep dH and H group histories in the existing compact BF16 boundaries.
- Use global scratch only for producer-complete tiles with more than one
  consumer. Do not retain whole-group operands merely to remove a launch.
- Size roles by the actual two-CTA constraints: twice the threads per CTA times
  allocated registers per thread must not exceed 65,536 registers, subject to
  allocation granularity; twice total shared per CTA must not exceed 102,400 B;
  and the combined thread/warp counts must fit 1,536 threads and 48 warps. The
  simple 128-register and 48-KiB limits apply only to roles of at most 256
  threads; a 512-thread role at 128 registers admits only one CTA.

Before CUDA implementation, write an explicit tile-lifetime table naming the
producer, first consumer, last consumer, storage type, rounding point, and
owner CTA for dP, dQ, dR, dE, dW, dA, dM, dqbar, dkhat, dbeta, and dprefix.
This is the main logical-risk item; prior giant-CTA and adjacent-group attempts
failed because those lifetimes were broadened implicitly.

Hard backward gate: complete plus colored at most 0.80 ms and no regression
larger than 0.03 ms in group pack or boundary recurrence. The combined
kernel-only path should reach at most 3.65 ms before any protected ABI expansion.

## Vertical fusion needed for the final margin

Even successful forward/backward local rewrites may stop short of 0.8--0.9 ms.
The remaining margin should come from model-specific boundaries that FLA's
standalone public KDA API cannot remove.

### Fuse width-four q/k/v convolution into KDA preprocessing

`KimiDeltaAttention.forward` currently materializes three BF16 convolution
outputs before KDA. At B2/T4096/D384 those tensors total 18 MiB. Forward alone
creates up to 36 MiB of global write-plus-read traffic, although the L2/DRAM
split is unmeasured. Backward also materializes KDA dq/dk/dv before three
convolution backward calls.

A fused training operator should accept the three projected inputs and
convolution weights, reproduce the exact width-four BF16 rounding and SiLU in
the KDA factory, and return gradients directly to the three projection outputs
and convolution weights. Current native KDA backward still consumes `v`
directly, while `q` and `k` are recoverable from retained normalized data.
Eliminating all three convolution outputs therefore requires recomputing `v`
before or inside complete VJP, or retaining it. Benchmark recomputed `v` against
retention instead of assuming all three surfaces are free to drop.

Required properties:

- Preserve width-four halo semantics at sequence and reverse-group boundaries.
- Preserve the existing convolution product-round, preactivation-round, and
  SiLU-round order.
- Do not add q/k/v retained sidecars that merely move the same 18 MiB boundary.
- Fuse KDA finalize into convolution backward so dq/dk/dv are not full global
  handoff tensors.

This changes the protected native ABI and component accounting. It requires a
separately declared campaign and explicit authorization to expand the protected
harness; it must not be smuggled into the existing five-component ownership
program.

### Fuse output RMSNorm and sigmoid gate

The output CTA already owns a 64×128 FP32 result tile immediately before BF16
publication. A future fused operator can apply per-token RMSNorm, learned norm
weight, and sigmoid output gate at that boundary, then return the tensor consumed
by `o_proj`.

The backward must preserve the current BF16 mixed-value rounding seen by
RMSNorm. Either retain that 6 MiB BF16 mixed surface in the hidden output
backing or prove a producer-complete recomputation. Do not silently normalize
the pre-round FP32 accumulator.

Vertical-fusion gate: at least 0.20 ms additional saving per production KDA
block with trainer peak still within 1.03 times attempt342. If convolution and
normalization fusion together cannot clear that threshold, stop rather than
expanding more model boundaries.

## Implementation order and stop rules

| Milestone | Scope | Required evidence to continue |
|---|---|---|
| 0. Counter access and baseline | No source edits | Repeated matched attempt342/FLA operator captures; enable Nsight Compute counters if possible |
| 1. Streaming C64 factory + U/W | `chunk_wy_forward.cu` or a new forward CUDA unit | Exact/tolerant production output, ≤50 KiB shared, ≤64 regs, no spill, phase ≤0.60 ms |
| 2. C16 local backward roles | `chunk_wy_backward.cu` or a new backward CUDA unit | Random-dO gradients pass; complete+colored ≤0.80 ms; core call ≤3.65 ms |
| 3. Kernel-only frozen Level 1 | Existing protected operator ABI | T4096 forward+backward improves materially, memory ≤1.03×, no important-shape regression |
| 4. Fused block ABI pilot | `nanochat/mixers/kda.py`, native fused CUDA units, declared harness expansion | Conv/KDA + norm/gate adds ≥0.20 ms saving; exact halo/rounding checks pass |
| 5. Final operator gate | Exact B2/H3/T4096 production call | Median ≤3.45 ms, selective-PTX A/B passes, all four sanitizers pass |
| 6. Trainer | Clean commits only | Matched paired median ≥3% over FLA and ≥2% over attempt342; memory ≤1.03× |

Do not launch a trainer before milestone 5. Do not compose a subthreshold
candidate simply because two isolated medians are positive; rerun the combined
operator and profile consumer locality first.

## Correctness and provenance gates

Every retained candidate must pass:

- Independent oracle forward and all seven gradients at the protected
  tolerances, including random upstream gradients rather than only a squared
  loss.
- For the expanded fused-block ABI, additionally validate gradients for all
  three projected inputs, all three convolution weights, the norm weight, the
  output gate, and every other newly accepted parameter.
- Explicit-zero, extreme-gate, finite-gradient, boundary-length, fallback,
  state-reset, recurrent-decode, and convolution-halo cases.
- Production comparison against attempt342, recording which tensors are
  bitwise equal and every nonzero maximum delta.
- Runtime ownership and external kernel-symbol profile audits with no runtime
  FLA and no reference/oracle import from candidate source.
- Selective-PTX enabled/disabled A/B with a measured benefit.
- Memcheck, racecheck, synccheck, and initcheck before conclusion-bearing
  performance.
- Full repository tests, clean-commit provenance, protected-file hashes,
  resolved configuration, environment, raw samples, and runbook update.

The generic and recurrent paths must remain unchanged until a separate decode
objective is declared. No private confirmation seed or artifact may be used to
choose any of these designs.

## Current recommendation

Start with milestone 1, not another launch-count or tile-size sweep. Its result
answers the key hardware question cheaply: can two resident C64 factories and
producer-local U/W beat the current 95.5-KiB-total one-CTA program without paying the
old global-intermediate penalty?

If that mechanism cannot save roughly 0.25 ms, stop the internal-kernel campaign
and move directly to the explicitly authorized fused-block ABI. If it succeeds,
use it as the foundation for the backward tile-lifetime rewrite and vertical
fusion. That stack is the first remaining plan whose summed mechanism budget
can plausibly produce a comfortable trainer lead over FLA.
