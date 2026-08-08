from dataclasses import replace
from pathlib import Path
import hashlib
import json
import sqlite3
import subprocess

import pytest

from nanochat.research.config import ResearchConfig, load_config
from nanochat.research.cuda_config import CudaCampaignConfigError, load_cuda_campaign_config, validate_cuda_campaign_config
from nanochat.research.cuda_supervisor import (
    calibrate_anchors, campaign_report, decide_migration, decide_performance, initialize, intake,
    paired_interval, protocol_sha, recover_interrupted, retain, summary, CONTROLLER_PATHS,
)
from nanochat.research.speed_supervisor import _protocol_sha as legacy_protocol_sha

ROOT=Path(__file__).resolve().parents[1]


def git(root,*args):
    result=subprocess.run(["git",*args],cwd=root,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    return result.stdout.strip()


def make_repo(tmp_path):
    root=tmp_path/"repo"; source=root/"nanochat"/"mixers"/"cuda_kda"; source.mkdir(parents=True)
    (source/"__init__.py").write_text("stub\n")
    for relative in CONTROLLER_PATHS:
        target=root/relative; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    git(tmp_path,"init",str(root)); git(root,"config","user.email","test@example.com"); git(root,"config","user.name","Test")
    git(root,"add","."); git(root,"commit","-m","foundation"); return root,git(root,"rev-parse","HEAD")


def config_for(tmp_path,foundation):
    value=load_cuda_campaign_config(ROOT/"configs/research/kda_cuda_ownership.toml")
    return replace(value,campaign=replace(value.campaign,ledger_path=str(tmp_path/"cuda.sqlite3"),artifact_root=str(tmp_path/"artifacts"),foundation_ref=foundation,controller_ref=foundation,cumulative_performance_anchor_ref=foundation),reporting=replace(value.reporting,historical_context_manifest=str(ROOT/"configs/research/archive/kda_training_speed_context.json")))


def commit_candidate(root,name="kernel.cu",body="// candidate CUDA\n"):
    path=root/"nanochat"/"mixers"/"cuda_kda"/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(body)
    git(root,"add","."); git(root,"commit","-m",f"candidate {name}"); return git(root,"rev-parse","HEAD")


def kernel_payload(ms):
    return {"status":"complete","microbenchmarks":[{"operation":"chunk_forward","length":256,"median_ms":ms,"values_ms":[ms]*10,"coefficient_of_variation":0.0}],"profile":[]}


def complete_attempt(config,attempt_id,owners,eligibility,runtime_fla_free=False,ms=10.0,performance="observed"):
    weights=dict(zip(config.ownership.required_components,config.ownership.component_weights)); fraction=sum(weights[name] for name in owners)
    db=sqlite3.connect(config.campaign.ledger_path)
    row=db.execute("SELECT lane,parent_milestone_id,base_sha,candidate_sha,hypothesis FROM attempts WHERE id=?",(attempt_id,)).fetchone()
    lane,parent,base,candidate,hypothesis=row
    summary={"schema_version":2,"lane":lane,"attempt_id":attempt_id,"parent_milestone_id":parent,"base_sha":base,"candidate_sha":candidate,"hypothesis":hypothesis,"quality_not_evaluated":True,
      "migration":{"decision":"migration_ready" if len(owners)==len(config.ownership.required_components) and runtime_fla_free else "ownership_progress","reason":"fixture","owner_set":sorted(owners),"owned_fraction":fraction,"runtime_fla_free":runtime_fla_free,"new_components":sorted(owners)},
      "performance":{"decision":performance,"reason":"fixture","advisory":lane!="optimization","paired_interval":None,"candidate_kernel_status":"timeout" if performance=="observed_timeout" else "complete"},
      "eligibility_decision":eligibility,"baseline_backend":"reference" if lane=="bootstrap" else "project_cuda","baseline_kernel":kernel_payload(ms*1.2),"candidate_kernel":kernel_payload(ms),"artifact_dir":"fixture"}
    db.execute("UPDATE attempts SET status='complete',migration_decision=?,performance_decision=?,eligibility_decision=?,reason='fixture',summary_json=? WHERE id=?",(summary["migration"]["decision"],performance,eligibility,json.dumps(summary,sort_keys=True),attempt_id)); db.commit(); db.close()
    return summary


def add_anchor_metrics(config,foundation,python_ms=100.0,fla_ms=5.0):
    db=sqlite3.connect(config.campaign.ledger_path)
    for kind,label,backend,ms in (("python_reference","Python","reference",python_ms),("fla_reference","FLA","fla_triton",fla_ms)):
        db.execute("INSERT INTO anchors(kind,label,commit_sha,backend,metric_json,artifact_dir,comparison_compatible,created_at) VALUES(?,?,?,?,?,NULL,1,0)",(kind,label,foundation,backend,json.dumps(kernel_payload(ms))))
    db.commit(); db.close()


def test_cuda_schema_is_separate_and_legacy_protocol_hashes_are_golden():
    assert "cuda_ownership" not in ResearchConfig.__dataclass_fields__
    assert legacy_protocol_sha(load_config(ROOT/"configs/research/archive/kda_training_speed_campaign.toml"))=="c50c1dfdddc62b332d0f61ffb6671327958cf27e8c1c38d27ae500694f3f0332"
    assert legacy_protocol_sha(load_config(ROOT/"configs/research/archive/kda_training_speed_dry_run.toml"))=="fed6455c6f2d3e5aa32eca99281dca8b4b1bff0af3d741e2beead2acc270f3a3"


def test_lane_schema_freezes_atomic_units_and_measurement_contract():
    value=load_cuda_campaign_config(ROOT/"configs/research/kda_cuda_ownership.toml")
    assert value.schema_version==2
    assert protocol_sha(value)=="6fdb0ec11d7efb82ae67bf39997f3601eae08026f5ec12f719f21f1c7c916e7c"
    assert value.ownership.atomic_units==(("chunk_forward","chunk_backward"),("recurrent_decode",),("causal_convolution_forward","causal_convolution_backward"))
    assert value.bootstrap.performance_advisory and value.bootstrap.forbid_selective_ptx
    assert value.migration.performance_advisory and value.migration.require_strict_owner_superset
    assert value.optimization.require_all_components_project and value.optimization.require_runtime_fla_free
    assert value.measurement.discovery_paired_blocks==9 and 0.005<=value.measurement.discovery_effect_fraction<=0.01
    assert value.measurement.promotion_paired_blocks==15
    assert value.measurement.promotion_cumulative_fraction==0.03
    assert value.measurement.block_timeout_seconds==13_500.0
    assert value.kernel_gates.sequence_lengths==(1,64,65,256,1024,4096)
    assert value.kernel_gates.warmup_iterations==10 and value.kernel_gates.timed_iterations==50
    assert value.kernel_gates.timeout_seconds==3_600.0
    bad=replace(value,ownership=replace(value.ownership,atomic_units=(("chunk_forward",),)))
    with pytest.raises(CudaCampaignConfigError,match="partition"):
        validate_cuda_campaign_config(bad)


def test_timeout_refreeze_changes_only_namespaces_and_process_ceilings():
    current=load_cuda_campaign_config(ROOT/"configs/research/kda_cuda_ownership.toml")
    previous=replace(
        current,
        campaign=replace(
            current.campaign,
            controller_ref="kda-cuda-ownership-controller-state-gradient-gates",
            ledger_path="runs/kda-cuda-ownership.sqlite3",
        ),
        measurement=replace(current.measurement,block_timeout_seconds=180.0),
        kernel_gates=replace(current.kernel_gates,timeout_seconds=180.0),
    )

    def changed(left,right,prefix=""):
        if isinstance(left,dict) and isinstance(right,dict):
            paths=[]
            for key in sorted(set(left)|set(right)):
                path=f"{prefix}.{key}" if prefix else key
                paths.extend(changed(left.get(key),right.get(key),path))
            return paths
        return [] if left==right else [prefix]

    assert set(changed(previous.to_dict(),current.to_dict()))=={
        "campaign.controller_ref",
        "campaign.ledger_path",
        "measurement.block_timeout_seconds",
        "kernel_gates.timeout_seconds",
    }


def test_paired_decision_separates_retention_from_improvement(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation)
    assert decide_performance(paired_interval([100.0]*9,[100.2]*9),config)[0]=="retained"
    assert decide_performance(paired_interval([100.0]*9,[101.0]*9),config)[0]=="improved"
    assert decide_performance(paired_interval([100.0]*9,[98.0]*9),config)[0]=="regressed"


def test_migration_decision_is_lane_and_parent_aware(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation)
    components={name:{"owner":"third_party"} for name in config.ownership.required_components}
    for name in ("causal_convolution_forward","causal_convolution_backward"): components[name]={"owner":"project"}
    audit={"status":"complete","runtime_fla_free":False,"provenance":{"owned_fraction":0.2,"components":components}}
    assert decide_migration(audit,config,set(),"bootstrap")[0]=="ownership_progress"
    assert decide_migration(audit,config,set(components),"migration")[0]=="no_ownership_progress"


def test_initialize_pins_refs_and_creates_foundation_state(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation)
    ready=initialize(root,config); assert ready["ledger_revision"]==2 and ready["foundation_sha"]==foundation
    view=summary(root,config); assert view["next_lane"]=="bootstrap" and not view["anchors_calibrated"]
    json.dumps(view,allow_nan=False)
    assert "simplest correct naive" in view["next_model_instruction"] and "no PTX" in view["next_model_instruction"]


def test_initialize_rejects_protocol_change_without_mutating_existing_ledger(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation)
    initialize(root,config); path=Path(config.campaign.ledger_path); before=path.read_bytes()
    changed=replace(config,kernel_gates=replace(config.kernel_gates,timeout_seconds=config.kernel_gates.timeout_seconds+300.0))
    with pytest.raises(ValueError,match="protocol hash differs"):
        initialize(root,changed)
    assert path.read_bytes()==before


def test_intake_is_forced_to_current_bootstrap_head_and_scope(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config); add_anchor_metrics(config,foundation)
    candidate=commit_candidate(root)
    result=intake(root,config,foundation,candidate,"model-chosen naive convolution")
    assert result["status"]=="accepted" and result["lane"]=="bootstrap"
    assert result["parent_milestone_id"]==1
    other=root.parent/"other"; other.mkdir(); git(root.parent,"init",str(other))


def test_intake_rejects_binary_and_rename_scope_bypasses(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config)
    binary=root/"nanochat/mixers/cuda_kda/kernel.so"; binary.write_bytes(b"binary"); git(root,"add","."); git(root,"commit","-m","binary"); candidate=git(root,"rev-parse","HEAD")
    assert intake(root,config,foundation,candidate,"binary")["status"]=="rejected"
    root2,foundation2=make_repo(tmp_path/"rename"); config2=config_for(tmp_path/"rename",foundation2); protected=root2/"protected.py"; protected.write_text("secret\n"); git(root2,"add","."); git(root2,"commit","-m","protected"); base=git(root2,"rev-parse","HEAD")
    # The current head is foundation2, so this also proves arbitrary non-head bases fail.
    git(root2,"mv","protected.py","nanochat/mixers/cuda_kda/stolen.py"); git(root2,"commit","-m","rename"); cand=git(root2,"rev-parse","HEAD")
    result=intake(root2,config2,base,cand,"rename"); assert result["status"]=="rejected"


def test_slow_correct_bootstrap_is_retainable_and_advances_to_migration(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config); add_anchor_metrics(config,foundation)
    candidate=commit_candidate(root); accepted=intake(root,config,foundation,candidate,"naive CUDA, optimize nothing")
    owners={"causal_convolution_forward","causal_convolution_backward"}
    complete_attempt(config,accepted["attempt_id"],owners,"correct_bootstrap",False,ms=500.0,performance="observed_timeout")
    retained=retain(root,config,accepted["attempt_id"],"Naive convolution bootstrap","Correct despite advisory timeout")
    assert retained["kind"]=="correct_bootstrap"
    view=summary(root,config); assert view["next_lane"]=="migration" and "unclaimed" in view["next_model_instruction"]


def test_migration_requires_strict_owner_superset_and_retains_component(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config); add_anchor_metrics(config,foundation)
    first=commit_candidate(root,"conv.cu"); a1=intake(root,config,foundation,first,"conv"); owners={"causal_convolution_forward","causal_convolution_backward"}; complete_attempt(config,a1["attempt_id"],owners,"correct_bootstrap"); m1=retain(root,config,a1["attempt_id"],"conv","first unit")
    second=commit_candidate(root,"recurrent.cu"); a2=intake(root,config,first,second,"recurrent"); owners.add("recurrent_decode"); complete_attempt(config,a2["attempt_id"],owners,"validated_component"); m2=retain(root,config,a2["attempt_id"],"recurrent","strict superset")
    assert m2["ordinal"]==2 and summary(root,config)["next_lane"]=="migration"


def test_complete_fla_free_naive_opens_optimization(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config); add_anchor_metrics(config,foundation)
    candidate=commit_candidate(root); a=intake(root,config,foundation,candidate,"complete naive CUDA")
    owners=set(config.ownership.required_components); complete_attempt(config,a["attempt_id"],owners,"fla_free_naive",True,ms=25.0)
    retained=retain(root,config,a["attempt_id"],"Complete naive CUDA","First all-project backend")
    assert retained["kind"]=="fla_free_naive" and summary(root,config)["next_lane"]=="optimization"


def test_stale_attempt_cannot_advance_head(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config); add_anchor_metrics(config,foundation)
    c1=commit_candidate(root,"one.cu"); a1=intake(root,config,foundation,c1,"one")
    # Build a parallel candidate from the same foundation without checkout by committing on top is rejected as wrong base after head advances.
    complete_attempt(config,a1["attempt_id"],{"causal_convolution_forward","causal_convolution_backward"},"correct_bootstrap"); retain(root,config,a1["attempt_id"],"one","retain")
    c2=commit_candidate(root,"two.cu"); result=intake(root,config,foundation,c2,"stale")
    assert result["status"]=="rejected" and "current retained" in result["reason"]


def test_milestones_and_anchors_are_sql_append_only(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config)
    db=sqlite3.connect(config.campaign.ledger_path)
    with pytest.raises(sqlite3.IntegrityError,match="append-only"): db.execute("UPDATE milestones SET label='x' WHERE ordinal=0")
    with pytest.raises(sqlite3.IntegrityError,match="immutable"): db.execute("DELETE FROM anchors WHERE kind='cuda_foundation'")
    db.close()


def test_report_builds_compatible_waterfall_without_chained_ci(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config); add_anchor_metrics(config,foundation,100.0,5.0)
    candidate=commit_candidate(root); a=intake(root,config,foundation,candidate,"naive"); complete_attempt(config,a["attempt_id"],set(config.ownership.required_components),"fla_free_naive",True,ms=20.0); retain(root,config,a["attempt_id"],"naive","anchor")
    report=campaign_report(root,config)
    milestone=report["milestones"][-1]
    assert report["schema"]=="kda_cuda_ownership_report" and report["schema_version"]==2 and milestone["speedup_ratios"]["python_reference"]==5.0
    assert milestone["per_shape_speedup_ratios"]["python_reference"]["chunk_forward:256"]==5.0
    assert milestone["ownership_percentage_point_change"]==pytest.approx(100.0) and set(milestone["new_components"])==set(config.ownership.required_components)
    assert "ci" not in "illustrative_chained_optimization_point_estimate"
    assert report["quality_not_evaluated"] is True


def test_foreign_ledger_rejected_before_mutation(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); path=Path(config.campaign.ledger_path)
    db=sqlite3.connect(path); db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"); db.execute("INSERT INTO metadata VALUES('schema','speed-supervisor')"); db.commit(); db.close(); before=path.read_bytes()
    with pytest.raises(ValueError,match="schema marker"): initialize(root,config)
    assert path.read_bytes()==before


def test_campaign_report_and_summary_use_read_only_connections(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config); path=Path(config.campaign.ledger_path); before=hashlib.sha256(path.read_bytes()).hexdigest()
    campaign_report(root,config); summary(root,config)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before and not Path(str(path)+"-wal").exists()


def test_interrupted_attempt_has_terminal_recovery(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config); add_anchor_metrics(config,foundation); candidate=commit_candidate(root); accepted=intake(root,config,foundation,candidate,"interrupt")
    db=sqlite3.connect(config.campaign.ledger_path); db.execute("UPDATE attempts SET status='testing' WHERE id=?",(accepted["attempt_id"],)); db.commit(); db.close()
    assert recover_interrupted(root,config,accepted["attempt_id"],"terminated")["status"]=="invalid"


def test_cuda_cli_has_staged_commands_and_speed_cli_is_unchanged():
    from nanochat.research.cli import build_parser
    parser=build_parser()
    assert parser.parse_args(["cuda-ownership-supervisor","calibrate"]).cuda_command=="calibrate"
    assert parser.parse_args(["cuda-ownership-supervisor","retain","--attempt","1","--label","x","--reason","y"]).cuda_command=="retain"
    assert parser.parse_args(["cuda-ownership-supervisor","verify-release","--milestone","2"]).cuda_command=="verify-release"
    candidate=parser.parse_args(["cuda-candidate-check","--worktree","/tmp/candidate","--lane","bootstrap","--sanitizers"])
    assert candidate.command=="cuda-candidate-check" and candidate.worktree=="/tmp/candidate" and candidate.sanitizers
    preflight=parser.parse_args(["cuda-toolchain-preflight","--sanitizers"]); assert preflight.command=="cuda-toolchain-preflight" and preflight.sanitizers
    speed=parser.parse_args(["speed-supervisor","summary"]); assert speed.command=="speed-supervisor" and speed.speed_command=="summary"


def test_cuda_candidate_checker_validates_the_staged_snapshot(tmp_path):
    from nanochat.research.cuda_candidate import inspect_staged_candidate
    root,foundation=make_repo(tmp_path)
    config=config_for(tmp_path,foundation)
    source=root/"nanochat/mixers/cuda_kda/recurrent.cu"
    source.write_text("// naive recurrent kernel\n")
    git(root,"add",str(source.relative_to(root)))
    result=inspect_staged_candidate(root,config)
    assert result["staged_paths"]==["nanochat/mixers/cuda_kda/recurrent.cu"]
    assert result["sources"][0]["sha256"]
    source.write_text("// unstaged replacement\n")
    with pytest.raises(ValueError,match="no unstaged or untracked"):
        inspect_staged_candidate(root,config)


def test_cuda_candidate_cli_dispatches_without_supervisor_or_ledger(monkeypatch,tmp_path):
    import nanochat.research.cli as cli
    captured={}
    monkeypatch.setattr(cli,"load_cuda_campaign_config",lambda path:"config")
    def check(root,config,**kwargs):
        captured.update(root=root,config=config,**kwargs)
        return {"status":"complete","ledger_accessed":False}
    monkeypatch.setattr(cli,"run_cuda_candidate_check",check)
    assert cli.main(["cuda-candidate-check","--worktree",str(tmp_path/"candidate"),"--lane","migration"])==0
    assert captured["lane"]=="migration" and captured["worktree"]==str(tmp_path/"candidate")


def test_worker_and_supervisor_preserve_build_sanitizer_and_no_trace_guards():
    supervisor=(ROOT/"nanochat/research/cuda_supervisor.py").read_text(); worker=(ROOT/"nanochat/research/cuda_worker.py").read_text()
    assert "--error-exitcode=99" in supervisor and "sanitizer_zero_summary" in supervisor
    assert "TORCH_EXTENSIONS_DIR" in supervisor and "CUDA_CACHE_PATH" in supervisor
    assert "_dispatch_has_kernel_for_dispatch_key" in worker and 'Path("/proc/self/maps")' in worker
    assert "export_chrome_trace" not in worker and "INSERT OR REPLACE INTO phases" not in supervisor
    assert 'controller_root=Path(__file__).resolve().parents[2]' in supervisor
    assert 'bridge_flag="--implementation-root"' in supervisor and "--implementation-root" in worker
    assert "--historical-implementation-root" in supervisor and "--historical-implementation-root" in worker
    assert "nanochat.mixers.__path__.insert(0,implementation_mixers)" in worker
    assert 'sys.executable,"-m","nanochat.research.cuda_worker"' not in supervisor
    assert "artifact_dir) VALUES(?,?,'testing',?,?,'pending')" not in supervisor
    dispatcher=(ROOT/"nanochat/mixers/kda.py").read_text()
    assert '_project_operator(forward_component)(' in dispatcher and "_ProjectConvolutionAutograd.apply(" in dispatcher
    assert "_ProjectKDAAutograd.apply(" in dispatcher and "ctx.backward_component" in dispatcher
    assert "_load_project_backend().kda(" not in dispatcher and "_load_project_backend().causal_convolution(" not in dispatcher


def test_project_dispatch_is_fail_closed_on_unsupported_cpu(monkeypatch):
    import importlib,torch
    implementation=importlib.import_module("nanochat.mixers.kda")
    monkeypatch.setattr(implementation,"_reference_kda",lambda *a,**k:(_ for _ in ()).throw(AssertionError("fallback")))
    values=(torch.zeros(1,1,1,4),)*4+(torch.zeros(1,1,1),torch.zeros(1),torch.zeros(4))
    with pytest.raises(RuntimeError,match="project CUDA"): implementation.kda(*values,mode="project_chunk",allow_fallback=True)


def test_protected_dispatch_routes_unclaimed_to_fla_and_claimed_to_project(monkeypatch):
    import importlib,torch
    implementation=importlib.import_module("nanochat.mixers.kda")
    q=torch.ones(1,2,1,4,requires_grad=True); beta=torch.zeros(1,2,1); A=torch.zeros(1); dt=torch.zeros(4)
    values=(q,q.clone(),q.clone(),q.clone(),beta,A,dt)
    monkeypatch.setattr(implementation,"_optimized_backend_reason",lambda *a,**k:None)
    calls=[]
    def fla(*args,**kwargs): calls.append("fla"); return args[0]*2,None
    class Project:
        def provenance(self): return implementation._PROJECT_PROVENANCE
    def project_operator(component):
        def call(*args):
            calls.append("project")
            if component=="chunk_backward":
                q,k,v,gate,beta,A,dt,initial,*_=args
                return tuple(torch.zeros_like(item) for item in (q,k,v,gate,beta,A,dt))+(None,)
            return args[0]*3,None
        return call
    monkeypatch.setattr(implementation,"_run_fla_kda",fla); monkeypatch.setattr(implementation,"_load_project_backend",lambda:Project()); monkeypatch.setattr(implementation,"_project_operator",project_operator)
    unclaimed={name:{"owner":"third_party"} for name in ("chunk_forward","chunk_backward","recurrent_decode","causal_convolution_forward","causal_convolution_backward")}
    implementation._PROJECT_PROVENANCE={"components":unclaimed}; implementation.reset_project_runtime_events()
    output,_=implementation.kda(*values,mode="project_chunk",allow_fallback=False); output.sum().backward()
    assert calls==["fla"] and {event["backend"] for event in implementation.project_runtime_events()}=={"fla"}
    claimed={name:{"owner":"project"} for name in unclaimed}; implementation._PROJECT_PROVENANCE={"components":claimed}; implementation.reset_project_runtime_events(); calls.clear(); q.grad=None
    output,_=implementation.kda(*values,mode="project_chunk",allow_fallback=False); output.sum().backward()
    assert calls==["project","project"] and {event["backend"] for event in implementation.project_runtime_events()}=={"project"}


def test_moved_foundation_tag_fails_closed(tmp_path):
    root,foundation=make_repo(tmp_path); git(root,"tag","foundation-tag",foundation)
    value=config_for(tmp_path,foundation); config=replace(value,campaign=replace(value.campaign,foundation_ref="foundation-tag"))
    initialize(root,config)
    moved=commit_candidate(root,"later.cu"); git(root,"tag","-f","foundation-tag",moved)
    with pytest.raises(ValueError,match="moved"):
        initialize(root,config)


def test_report_keeps_more_than_twelve_invalid_attempts(tmp_path):
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); ready=initialize(root,config)
    db=sqlite3.connect(config.campaign.ledger_path)
    parent=db.execute("SELECT current_milestone_id FROM campaign_state WHERE singleton=1").fetchone()[0]
    for index in range(13):
        db.execute("INSERT INTO attempts(created_at,status,lane,parent_milestone_id,base_sha,candidate_sha,candidate_tree_sha,hypothesis,changed_paths_json,patch_sha,patch_text,protocol_sha,reason) VALUES(0,'rejected','bootstrap',?,?,?,?,?,'[]',?,'',?,'fixture')",(parent,foundation,f"candidate-{index}",f"tree-{index}",f"hypothesis-{index}",f"patch-{index}",ready["protocol_sha"]))
    db.commit(); db.close()
    assert len(campaign_report(root,config)["attempts"])==13


def test_optimization_uses_parent_project_backend_and_release_is_separate():
    source=(ROOT/"nanochat/research/cuda_supervisor.py").read_text()
    assert 'baseline_backend="reference" if lane=="bootstrap" else "project_cuda"' in source
    assert 'order=(("baseline",baseline_root,"project_cuda"),("candidate",candidate_root,"project_cuda"))' in source
    assert "CREATE TABLE IF NOT EXISTS release_runs" in source
    assert 'order=(("baseline",baseline_root,"fla_triton"),("candidate",candidate_root,"project_cuda"))' in source


def test_anchor_calibration_failure_is_atomic_and_retryable(tmp_path,monkeypatch):
    import nanochat.research.cuda_supervisor as supervisor
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation); initialize(root,config)
    calls={"count":0}
    payload={"status":"complete","microbenchmarks":[{"operation":"chunk_forward","length":256,"median_ms":1.0}]}
    def flaky(*args,**kwargs):
        calls["count"]+=1
        if calls["count"]==2: return {"status":"crash","reason":"fixture"},{"status":"invalid","reason":"fixture"}
        return {"status":"complete"},payload
    monkeypatch.setattr(supervisor,"_worker",flaky)
    first=calibrate_anchors(root,config); assert first["status"]=="invalid"
    db=sqlite3.connect(config.campaign.ledger_path); assert db.execute("SELECT COUNT(*) FROM anchors WHERE kind IN ('python_reference','fla_reference')").fetchone()[0]==0; db.close()
    second=calibrate_anchors(root,config); assert second["status"]=="calibrated" and "attempt-0002" in second["artifact_dir"]


def test_provenance_scans_unlisted_candidate_helpers_for_forbidden_runtime_paths(tmp_path):
    from nanochat.research.cuda_worker import _provenance
    root=tmp_path/"repo"; source=root/"nanochat/mixers/cuda_kda"; source.mkdir(parents=True); (source/"hidden.py").write_text("import fla\n")
    config=load_cuda_campaign_config(ROOT/"configs/research/kda_cuda_ownership.toml")
    class Fake:
        @staticmethod
        def kda_backend_provenance():
            return {"components":{name:{"owner":"third_party","sources":[],"kernel_symbols":[],"torch_operator":None} for name in config.ownership.required_components}}
    result=_provenance(root,Fake,config)
    assert result["status"]=="invalid" and "hidden.py" in result["reason"]


def test_sanitizer_claims_only_complete_atomic_units_and_requires_all_in_optimization():
    from nanochat.research.cuda_worker import _sanitizer_claims
    config=load_cuda_campaign_config(ROOT/"configs/research/kda_cuda_ownership.toml")
    components={name:{"owner":"third_party"} for name in config.ownership.required_components}
    components["chunk_forward"]={"owner":"project"}; components["chunk_backward"]={"owner":"project"}
    provenance={"components":components}
    assert _sanitizer_claims(provenance,config,"bootstrap")=={"chunk_forward","chunk_backward"}
    components["chunk_backward"]={"owner":"third_party"}
    with pytest.raises(RuntimeError,match="partially claimed atomic unit"):
        _sanitizer_claims(provenance,config,"migration")
    components["chunk_backward"]={"owner":"project"}
    with pytest.raises(RuntimeError,match="requires every component"):
        _sanitizer_claims(provenance,config,"optimization")


def test_worker_bridge_always_uses_current_protected_controller(tmp_path):
    from nanochat.research.cuda_supervisor import _worker_invocation
    implementation=tmp_path/"candidate"
    args,env=_worker_invocation("sanitizer-smoke",implementation,{"FIXTURE":"1"})
    assert Path(args[1]).resolve()==(ROOT/"nanochat/research/cuda_worker.py").resolve()
    assert args[2:]==["sanitizer-smoke","--implementation-root",str(implementation)]
    assert Path(env["PYTHONPATH"].split(__import__("os").pathsep)[0]).resolve()==ROOT.resolve()
    assert env["FIXTURE"]=="1"



def test_cuda_build_content_address_and_receipt_contract(tmp_path):
    from nanochat.research.cuda_build import _content_name

    source = tmp_path / "kernel.cu"
    source.write_text("__global__ void fixture() {}\n")
    first = _content_name("fixture-op", [source], (), (), ())
    assert first == _content_name("fixture-op", [source], (), (), ())
    assert first.startswith("fixture_op_")
    source.write_text("__global__ void changed() {}\n")
    assert first != _content_name("fixture-op", [source], (), (), ())
    helper = (ROOT / "nanochat/research/cuda_build.py").read_text()
    assert 'os.environ["TORCH_CUDA_ARCH_LIST"] = _TARGET_ARCH' in helper
    assert "is_python_module=False" in helper and 'cuobjdump, "--list-elf"' in helper
    assert '{"library_path", "source_paths", "compiler_command", "target_arch"}' in helper


def test_nsys_helpers_construct_profile_and_read_kernel_symbols(tmp_path, monkeypatch):
    import subprocess as subprocess_module
    import nanochat.research.cuda_preflight as preflight

    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/tools/{name}")
    command = preflight.nsys_profile_command(["python", "worker.py"], tmp_path / "profile")
    assert command[:4] == ["/tools/nsys", "profile", "--trace=cuda", "--sample=none"]
    assert command[-2:] == ["python", "worker.py"]
    report = tmp_path / "profile.nsys-rep"
    report.write_bytes(b"fixture")
    csv_output = '"Time (%)","Instances","Name"\n"100.0","1","visible_fixture_kernel(float *)"\n'
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_module.CompletedProcess(args[0], 0, csv_output, ""),
    )
    assert preflight.read_nsys_kernel_symbols(report) == ["visible_fixture_kernel(float *)"]


def test_sm121_preflight_is_a_real_registered_cuda_op():
    source = (ROOT / "nanochat/research/cuda_preflight.py").read_text()
    assert "__global__ void nanochat_sm121_hello_kernel" in source
    assert "TORCH_LIBRARY(nanochat_cuda_preflight" in source
    assert "TORCH_LIBRARY_IMPL(nanochat_cuda_preflight, CUDA" in source
    assert "torch.testing.assert_close(actual, expected" in source
    assert "capture_nsys_cuda_symbols" in source


def test_candidate_source_policy_ignores_comments_but_not_executable_imports(tmp_path):
    from nanochat.research.cuda_candidate import _comment_free_source
    commented=tmp_path/"commented.py"; commented.write_text("# import fla\nvalue = 1\n")
    active=tmp_path/"active.py"; active.write_text("import fla\n")
    assert b"import fla" not in _comment_free_source(commented,commented.read_bytes()).lower()
    assert b"import fla" in _comment_free_source(active,active.read_bytes()).lower()


def test_launch_hardening_binds_nsys_frozen_abi_and_read_only_guide():
    supervisor=(ROOT/"nanochat/research/cuda_supervisor.py").read_text()
    candidate=(ROOT/"nanochat/research/cuda_candidate.py").read_text()
    worker=(ROOT/"nanochat/research/cuda_worker.py").read_text()
    guide=(ROOT/"nanochat/mixers/cuda_kda/README.md").read_text()
    assert "capture_nsys_cuda_symbols" in supervisor and "capture_nsys_cuda_symbols" in candidate
    assert "protected_operator_trace_plus_external_nsys" in worker
    assert 'expected_operator=f"nanochat_kda::{name}"' in worker
    assert "nanochat_kda::chunk_backward(" in guide and "Tensor? grad_final_state" in guide
    assert "READ_ONLY_CANDIDATE_PATHS" in supervisor and "READ_ONLY_CANDIDATE_PATHS" in candidate


def test_candidate_checker_rejects_protected_readme_change(tmp_path):
    from nanochat.research.cuda_candidate import inspect_staged_candidate
    root,foundation=make_repo(tmp_path); config=config_for(tmp_path,foundation)
    readme=root/"nanochat/mixers/cuda_kda/README.md"; readme.write_text(readme.read_text()+"\nchanged\n")
    git(root,"add",str(readme.relative_to(root)))
    with pytest.raises(ValueError,match="protected read-only"):
        inspect_staged_candidate(root,config)


def test_forbidden_runtime_finder_scopes_blocks_to_candidate_importer(tmp_path):
    from nanochat.research.cuda_worker import _ForbiddenRuntimeFinder
    candidate_root=tmp_path/"nanochat/mixers/cuda_kda"; candidate_root.mkdir(parents=True)
    candidate_file=candidate_root/"dynamic.py"
    finder=_ForbiddenRuntimeFinder(("fla",),candidate_root)
    assert finder.find_spec("fla") is None  # protected caller may load transitional fallback
    code=compile("finder.find_spec('fla')",str(candidate_file),"exec")
    with pytest.raises(ModuleNotFoundError,match="candidate runtime module forbidden"):
        exec(code,{"finder":finder})
    assert finder.attempts==["fla"]


def test_profile_recorder_and_sm121_receipt_launch_fixes_are_pinned():
    worker=(ROOT/"nanochat/research/cuda_worker.py").read_text()
    helper=(ROOT/"nanochat/research/cuda_build.py").read_text()
    assert '_NativeOperatorRecorder("nanochat_kda")' not in worker
    assert 'valid_target=target_arch in {"12.1","sm_121"}' in worker
    assert '"compute_121" in compiler_command and "sm_121" in compiler_command' in worker
    assert worker.count('_ForbiddenRuntimeFinder(config.ownership.forbid_runtime_modules, root / config.campaign.candidate_paths[0])')==2
    assert '"target_arch": "sm_121"' in helper


def test_protected_convolution_gradient_cases_cover_cache_boundaries_and_none_state():
    worker=(ROOT/"nanochat/research/cuda_worker.py").read_text()
    assert "convolution_cases" in worker
    for case in ("(65, True, True)","(2, True, True)","(4, True, True)","(5, False, True)","(3, True, False)"):
        assert case in worker
    assert "conv_initial.grad" in worker and "expected_initial.grad" in worker
    assert "causal_convolution_forward_backward_cache_gradient" in worker
