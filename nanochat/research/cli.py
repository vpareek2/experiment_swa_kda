from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from nanochat.research.artifacts import atomic_write_json
from nanochat.research.config import ConfigError, apply_candidate, load_config
from nanochat.research.cuda_config import CudaCampaignConfigError, load_cuda_campaign_config
from nanochat.research.cuda_supervisor import (calibrate_anchors as cuda_calibrate, campaign_report as cuda_report, initialize as cuda_initialize, intake as cuda_intake, recover_interrupted as cuda_recover, render_campaign_report as render_cuda_report, retain as cuda_retain, run_attempt as cuda_run_attempt, summary as cuda_summary, verify_release as cuda_verify_release)
from nanochat.research.decision import aggregate_objectives, calibrate_objectives
from nanochat.research.probe import run_memory_probe
from nanochat.research.protected import initialize_supervisor, verify_protected
from nanochat.research.runner import calibrate_memory_probe, doctor, prepare_data, render_report, run_experiment
from nanochat.research.systems import run_system_benchmark
from nanochat.research.speed_supervisor import initialize as speed_initialize, intake as speed_intake, run_attempt as speed_run_attempt, summary as speed_summary
from nanochat.research.supervisor import format_command, sandbox_command, sign_result, verify_candidate


def _json(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _load_summaries(paths):
    return [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research", description="SWA/KDA research and evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_parser = sub.add_parser("doctor", help="validate the local research environment")
    doctor_parser.add_argument("--config", default="configs/research/discovery.toml")

    prepare_parser = sub.add_parser("prepare", help="show or execute frozen data/tokenizer preparation")
    prepare_parser.add_argument("--execute", action="store_true")

    probe_parser = sub.add_parser("probe", help="run only the controlled memory probe")
    probe_parser.add_argument("--config", default="configs/research/discovery.toml")
    probe_parser.add_argument("--output")
    probe_parser.add_argument("--calibrate", action="store_true", help="validate and register probe v2")
    probe_parser.add_argument("--seeds", help="comma-separated calibration seeds (at least three)")
    probe_parser.add_argument("--artifact-root")

    run_parser = sub.add_parser("run", help="run training, memory evaluation, and frontier classification")
    run_parser.add_argument("--config", default="configs/research/discovery.toml")
    run_parser.add_argument("--candidate", help="architecture-only candidate TOML applied to the frozen protocol")
    run_parser.add_argument("--artifact-root")
    run_parser.add_argument("--skip-training", action="store_true")
    run_parser.add_argument("--skip-probe", action="store_true")
    run_parser.add_argument("--seeds", help="comma-separated declared seeds for a sequential campaign")
    run_parser.add_argument("--confirmation-seeds", help=argparse.SUPPRESS)

    systems_parser = sub.add_parser("systems", help="run the frozen bounded systems benchmark")
    systems_parser.add_argument("--config", default="configs/research/systems_4k.toml")
    systems_parser.add_argument("--candidate", help="architecture-only candidate TOML applied to the frozen protocol")
    systems_parser.add_argument("--artifact-root")

    speed = sub.add_parser("speed-supervisor", help="manage bounded KDA training-speed candidates")
    speed_sub = speed.add_subparsers(dest="speed_command", required=True)
    speed_init = speed_sub.add_parser("init", help="create or verify the immutable speed ledger")
    speed_init.add_argument("--config", default="configs/research/kda_training_speed.toml")
    speed_init.add_argument("--ledger")
    speed_intake_parser = speed_sub.add_parser("intake", help="record and validate one committed candidate")
    speed_intake_parser.add_argument("--config", default="configs/research/kda_training_speed.toml")
    speed_intake_parser.add_argument("--base-ref", required=True)
    speed_intake_parser.add_argument("--candidate-ref", required=True)
    speed_intake_parser.add_argument("--idea", required=True)
    speed_intake_parser.add_argument("--ledger")
    speed_run = speed_sub.add_parser("run", help="run fixed tests and A-B-A timing for one accepted candidate")
    speed_run.add_argument("--config", default="configs/research/kda_training_speed.toml")
    speed_run.add_argument("--attempt", type=int, required=True)
    speed_run.add_argument("--ledger")
    speed_view = speed_sub.add_parser("summary", help="render compact evidence for the next candidate-model turn")
    speed_view.add_argument("--config", default="configs/research/kda_training_speed.toml")
    speed_view.add_argument("--attempt", type=int)
    speed_view.add_argument("--ledger")

    cuda = sub.add_parser("cuda-ownership-supervisor", help="manage protected project-owned KDA CUDA migration")
    cuda_sub = cuda.add_subparsers(dest="cuda_command", required=True)
    cuda_init = cuda_sub.add_parser("init", help="create or verify the independent CUDA-ownership ledger")
    cuda_init.add_argument("--config", default="configs/research/kda_cuda_ownership.toml")
    cuda_init.add_argument("--ledger")
    cuda_cal = cuda_sub.add_parser("calibrate", help="freeze Python-reference and FLA operator anchors before intake")
    cuda_cal.add_argument("--config", default="configs/research/kda_cuda_ownership.toml")
    cuda_cal.add_argument("--ledger")
    cuda_intake_parser = cuda_sub.add_parser("intake", help="record one freely chosen committed CUDA hypothesis in the derived lane")
    cuda_intake_parser.add_argument("--config", default="configs/research/kda_cuda_ownership.toml")
    cuda_intake_parser.add_argument("--base-ref", required=True)
    cuda_intake_parser.add_argument("--candidate-ref", required=True)
    cuda_intake_parser.add_argument("--hypothesis", required=True)
    cuda_intake_parser.add_argument("--ledger")
    cuda_run = cuda_sub.add_parser("run", help="run correctness, ownership, kernel, and interleaved performance gates")
    cuda_run.add_argument("--config", default="configs/research/kda_cuda_ownership.toml")
    cuda_run.add_argument("--attempt", type=int, required=True)
    cuda_run.add_argument("--ledger")
    cuda_retain_parser = cuda_sub.add_parser("retain", help="append one human-approved immutable milestone")
    cuda_retain_parser.add_argument("--config", default="configs/research/kda_cuda_ownership.toml")
    cuda_retain_parser.add_argument("--attempt", type=int, required=True)
    cuda_retain_parser.add_argument("--label", required=True)
    cuda_retain_parser.add_argument("--reason", required=True)
    cuda_retain_parser.add_argument("--commit-ref")
    cuda_retain_parser.add_argument("--ledger")
    cuda_release = cuda_sub.add_parser("verify-release", help="run fixed-FLA cumulative promotion for a retained milestone")
    cuda_release.add_argument("--config", default="configs/research/kda_cuda_ownership.toml")
    cuda_release.add_argument("--milestone", type=int, required=True)
    cuda_release.add_argument("--ledger")
    cuda_report_parser = cuda_sub.add_parser("report", help="render the complete naive-to-optimized lineage")
    cuda_report_parser.add_argument("--config", default="configs/research/kda_cuda_ownership.toml")
    cuda_report_parser.add_argument("--ledger")
    cuda_report_parser.add_argument("--format", choices=("json","markdown"), default="json")
    cuda_report_parser.add_argument("--output")
    cuda_report_parser.add_argument("--exclude-invalid", action="store_true")
    cuda_recover_parser = cuda_sub.add_parser("recover", help="terminally invalidate an interrupted testing transition")
    cuda_recover_parser.add_argument("--config", default="configs/research/kda_cuda_ownership.toml")
    cuda_recover_parser.add_argument("--attempt", type=int, required=True)
    cuda_recover_parser.add_argument("--reason", required=True)
    cuda_recover_parser.add_argument("--ledger")
    cuda_view = cuda_sub.add_parser("summary", help="render separate migration and performance evidence")
    cuda_view.add_argument("--config", default="configs/research/kda_cuda_ownership.toml")
    cuda_view.add_argument("--attempt", type=int)
    cuda_view.add_argument("--ledger")

    report_parser = sub.add_parser("report", help="render a comparison from summary JSON files")
    report_parser.add_argument("summaries", nargs="+")
    report_parser.add_argument("--output")

    aggregate_parser = sub.add_parser("aggregate", help="aggregate seed-level summary JSON files")
    aggregate_parser.add_argument("summaries", nargs="+")
    aggregate_parser.add_argument("--config", default="configs/research/discovery.toml")
    aggregate_parser.add_argument("--calibrate", action="store_true", help="calibrate decision floors from baseline seeds")
    aggregate_parser.add_argument("--output")

    supervisor = sub.add_parser("supervisor", help="manage protected evaluation state")
    supervisor_sub = supervisor.add_subparsers(dest="supervisor_command", required=True)
    init_parser = supervisor_sub.add_parser("init")
    init_parser.add_argument("--config", default="configs/research/discovery.toml")
    verify_parser = supervisor_sub.add_parser("verify")
    verify_parser.add_argument("--config", default="configs/research/discovery.toml")
    verify_parser.add_argument("--base-ref")
    verify_parser.add_argument("--candidate-ref")
    sign_parser = supervisor_sub.add_parser("sign")
    sign_parser.add_argument("summary")
    sandbox_parser = supervisor_sub.add_parser("sandbox-command")
    sandbox_parser.add_argument("--image", default="swa-kda-research:latest")
    sandbox_parser.add_argument("--source", default=".")
    sandbox_parser.add_argument("--output", required=True)
    sandbox_parser.add_argument("--cache", help="read-only prepared nanochat cache")
    sandbox_parser.add_argument("--config", default="configs/research/discovery.toml")
    sandbox_parser.add_argument("--candidate")
    sandbox_parser.add_argument("--confirmation", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    try:
        if args.command == "doctor":
            result = doctor(root, load_config(args.config))
            _json(result)
            return 0 if result["valid"] else 1
        if args.command == "prepare":
            _json(prepare_data(args.execute))
            return 0
        if args.command == "probe":
            config = load_config(args.config)
            def progress(event):
                if event["event"] == "calibration":
                    print(f"[research] calibration topology={event['topology']} seed={event['seed']}", flush=True)
                elif event["event"] == "probe_train":
                    print(f"[research] probe stage={event['stage']} step={event['stage_step']}/{event['stage_steps']} "
                          f"answers={event['supervised_answers']} loss={event['loss']:.4f}", flush=True)
                elif event["event"] == "probe_eval":
                    print(f"[research] probe-eval group={event['group']} cell={event['cell']} "
                          f"accuracy={event['accuracy']:.4f}", flush=True)
            if args.calibrate:
                seeds = ([int(value) for value in args.seeds.split(",") if value.strip()]
                         if args.seeds else list(config.memory_probe.calibration_seeds))
                result = calibrate_memory_probe(root, config, args.artifact_root, seeds, progress)
                if args.output:
                    atomic_write_json(args.output, result)
                _json(result)
                return 0 if result["status"] == "valid" else 1
            import torch
            result = run_memory_probe(
                config.memory_probe, config.training.window_pattern, config.run.seed,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                config.training.force_final_full,
                progress,
                config.training.kda_backend,
            )
            if args.output:
                atomic_write_json(args.output, result)
            _json(result)
            return 0
        if args.command == "run":
            config = load_config(args.config)
            if args.candidate:
                config = apply_candidate(args.candidate, config)
            raw_seeds = args.confirmation_seeds or args.seeds
            if raw_seeds:
                seeds = [int(value) for value in raw_seeds.split(",") if value.strip()]
                if not seeds:
                    raise ValueError("at least one seed is required")
                if args.confirmation_seeds:
                    run_name, suite = "confirmation", "confirmation"
                else:
                    run_name, suite = config.run.name, config.run.suite
                results = []
                for seed in seeds:
                    seeded = replace(config, run=replace(config.run, seed=seed, name=run_name, suite=suite))
                    results.append(run_experiment(root, seeded, args.artifact_root, args.skip_training, args.skip_probe))
                result = {"schema_version": 1, "status": "complete" if all(item.get("status") == "complete" for item in results) else "incomplete", "runs": results}
                _json(result)
                return 0 if result["status"] == "complete" else 1
            result = run_experiment(root, config, args.artifact_root, args.skip_training, args.skip_probe)
            print(f"research_status: {result.get('decision', {}).get('status', result.get('status'))}")
            print(f"summary_path: {Path(args.artifact_root or config.run.artifact_root) / result['run_id'] / 'summary.json'}")
            _json(result)
            return 0 if result.get("status") in {"complete", "diagnostic"} else 1
        if args.command == "systems":
            config = load_config(args.config)
            if args.candidate:
                config = apply_candidate(args.candidate, config)
            result = run_system_benchmark(root, config, args.artifact_root)
            _json(result)
            return 0 if result["status"] == "complete" else 1
        if args.command == "speed-supervisor":
            config = load_config(args.config)
            if args.speed_command == "init":
                _json(speed_initialize(root, config, args.ledger))
                return 0
            if args.speed_command == "intake":
                result = speed_intake(root, config, args.base_ref, args.candidate_ref, args.idea, args.ledger)
                _json(result)
                return 0 if result["status"] == "accepted" else 1
            if args.speed_command == "run":
                result = speed_run_attempt(root, config, args.attempt, args.ledger)
                _json(result)
                return 0 if result["status"] == "complete" else 1
            if args.speed_command == "summary":
                _json(speed_summary(root, config, args.attempt, args.ledger))
                return 0
        if args.command == "cuda-ownership-supervisor":
            config = load_cuda_campaign_config(args.config)
            if args.cuda_command == "init":
                _json(cuda_initialize(root, config, args.ledger))
                return 0
            if args.cuda_command == "calibrate":
                _json(cuda_calibrate(root, config, args.ledger))
                return 0
            if args.cuda_command == "intake":
                result = cuda_intake(root, config, args.base_ref, args.candidate_ref, args.hypothesis, args.ledger)
                _json(result)
                return 0 if result["status"] == "accepted" else 1
            if args.cuda_command == "run":
                result = cuda_run_attempt(root, config, args.attempt, args.ledger)
                _json(result)
                return 0 if result["status"] == "complete" else 1
            if args.cuda_command == "retain":
                _json(cuda_retain(root, config, args.attempt, args.label, args.reason, args.commit_ref, args.ledger))
                return 0
            if args.cuda_command == "verify-release":
                result = cuda_verify_release(root, config, args.milestone, args.ledger)
                _json(result)
                return 0 if result["status"] == "complete" else 1
            if args.cuda_command == "report":
                result = cuda_report(root, config, args.ledger, include_invalid=not args.exclude_invalid)
                rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n" if args.format == "json" else render_cuda_report(result)
                if args.output: Path(args.output).write_text(rendered, encoding="utf-8")
                else: print(rendered, end="")
                return 0
            if args.cuda_command == "recover":
                _json(cuda_recover(root, config, args.attempt, args.reason, args.ledger))
                return 0
            if args.cuda_command == "summary":
                _json(cuda_summary(root, config, args.attempt, args.ledger))
                return 0
        if args.command == "report":
            report = render_report(_load_summaries(args.summaries))
            if args.output:
                Path(args.output).write_text(report, encoding="utf-8")
            print(report, end="")
            return 0
        if args.command == "aggregate":
            summaries = _load_summaries(args.summaries)
            result = (calibrate_objectives(summaries, load_config(args.config).decision)
                      if args.calibrate else aggregate_objectives(summaries))
            if args.output:
                atomic_write_json(args.output, result)
            _json(result)
            return 0
        if args.command == "supervisor":
            if args.supervisor_command == "init":
                config = load_config(args.config)
                _json(initialize_supervisor(root, config.protection.protected_paths))
                return 0
            if args.supervisor_command == "verify":
                config = load_config(args.config)
                result = (
                    verify_candidate(root, config, args.base_ref, args.candidate_ref)
                    if args.base_ref and args.candidate_ref else verify_protected(root)
                )
                _json(result)
                return 0 if result["valid"] else 1
            if args.supervisor_command == "sign":
                _json(sign_result(args.summary))
                return 0
            if args.supervisor_command == "sandbox-command":
                command = sandbox_command(
                    args.image, args.source, args.output, args.config, args.confirmation,
                    args.cache, args.candidate,
                )
                print(format_command(command))
                return 0
    except (ArithmeticError, ConfigError, CudaCampaignConfigError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"research error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
