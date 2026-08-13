# Numerical integrity

The campaign optimizes KDA implementation, not the training objective. A result
is eligible only if the model still computes and trains the same KDA.

## Correctness basis

- `tests/kda_oracle.py` implements an independent recurrence and gradient oracle.
- Operator and layer tests cover forward values, random-upstream gradients,
  initial/final recurrent state, chunk/decode equivalence, boundary lengths,
  reset behavior, and backend routing.
- Training-sensitive candidates are checked across gradient accumulation and
  optimizer updates, including loss sequences and final model tensors where
  applicable.
- Native kernels are gated by CUDA memcheck, racecheck, synccheck, and initcheck.
- Backend provenance is audited: the retained path resolves to project CUDA,
  declares every component as project-owned, and runs without an FLA fallback.

Tolerance-based oracle checks establish mathematical correctness. Some later
implementation comparisons additionally achieved bitwise equality to the
retained parent. “Exact” in release prose means the complete KDA training
computation is preserved; it does not promise bitwise identity across arbitrary
hardware, toolchains, or backends.

## Disallowed shortcuts

The following invalidate a throughput candidate:

- freezing KDA or projection parameters;
- detaching a parameter-dependent value from autograd;
- returning only hidden-state/input gradients while omitting parameter gradients;
- substituting a local, value-only, straight-through, or other surrogate backward;
- changing recurrence, gating, normalization, convolution, or optimizer math;
- silently routing to FLA, PyTorch, or a naive reference path;
- timing a different batch/model/precision or excluding work inside the claimed
  training step.

The campaign did explore a value-only surrogate as a diagnostic upper-bound. It
reached roughly 48k–50k tok/s by freezing parameters and avoiding their exact
backward work. It is preserved as a rejected experiment because it would change
parameter updates and therefore training quality. It is not KDA optimization and
is excluded from every release result.

## Comparator versus oracle

Pinned FLA measures relative performance under the same trainer workload. It is
not assumed to be ground truth. Numerical correctness is established against the
independent oracle and the retained exact implementation, so a shared issue or
different reduction order in FLA cannot define correctness by agreement alone.

## Evidence limitations

The final three-pair confirmation reran throughput, backend resolution, losses,
and memory; it did not repeat every expensive sanitizer and optimizer gate.
Those gates apply because the measured source is byte-equivalent under model,
CUDA, config, trainer, tests, and dependency files to the previously audited
exact implementation. The release JSON records both source commits and that
equivalence relationship.
