"""Protected, staged supervisor for project-owned KDA CUDA autoresearch.

The completed speed campaign has a separate schema, hash domain, ledger, CLI,
and artifact tree.  This supervisor never edits candidates or Git refs.  It
records a correctness-first bootstrap, monotone component migration, then
strict full-system optimization and fixed-anchor release verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Iterator

from nanochat.research.artifacts import environment_provenance
from nanochat.research.cuda_config import KdaCudaCampaignConfig

SCHEMA = "cuda-ownership-supervisor"
LEDGER_REVISION = "2"
CONTROLLER_PATHS = (
    "configs/research/kda_cuda_ownership.toml", "nanochat/research/cuda_config.py",
    "nanochat/research/cuda_supervisor.py", "nanochat/research/cuda_worker.py",
    "nanochat/research/cuda_build.py", "nanochat/research/cuda_preflight.py",
    "nanochat/research/cuda_candidate.py", "nanochat/research/cli.py",
    "nanochat/mixers/kda.py", "nanochat/gpt.py", "scripts/base_train.py",
    "pyproject.toml", "uv.lock", "program_kda_cuda_ownership.md",
    "nanochat/mixers/cuda_kda/README.md",
)
READ_ONLY_CANDIDATE_PATHS={"nanochat/mixers/cuda_kda/README.md"}
FORBIDDEN_GENERATED_SUFFIXES = {".so", ".o", ".a", ".cubin", ".fatbin", ".pyc"}
RETAINABLE = {"correct_bootstrap", "validated_component", "fla_free_naive", "optimization_retained"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def protocol_sha(config: KdaCudaCampaignConfig) -> str:
    return hashlib.sha256(_canonical({"protocol_schema": "kda-cuda-ownership-supervisor/2", "config": config.to_dict()})).hexdigest()


def _git(root: Path, args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _commit(root: Path, ref: str) -> str:
    return _git(root, ["rev-parse", "--verify", f"{ref}^{{commit}}"])


def _tree(root: Path, ref: str, path: str = "nanochat/mixers/cuda_kda") -> str:
    return _git(root, ["rev-parse", f"{ref}:{path}"])


def _clean(root: Path) -> None:
    if _git(root, ["status", "--porcelain"]):
        raise ValueError("CUDA-ownership supervisor requires a clean coordinator worktree")


def _ledger_path(root: Path, config: KdaCudaCampaignConfig, override=None) -> Path:
    path = Path(override or config.campaign.ledger_path)
    return path if path.is_absolute() else root / path


def _validate_existing_schema(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        rows = dict(db.execute("SELECT key,value FROM metadata").fetchall())
    except sqlite3.Error as error:
        raise ValueError("existing ledger is not a CUDA-ownership ledger") from error
    finally:
        if "db" in locals(): db.close()
    if rows.get("schema") != SCHEMA or rows.get("ledger_revision") != LEDGER_REVISION:
        raise ValueError("ledger schema marker is not CUDA-ownership-supervisor revision 2")


def _open(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        _validate_existing_schema(path)
        db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        db.execute("PRAGMA query_only=ON")
        return db
    _validate_existing_schema(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30, isolation_level=None)
    db.execute("PRAGMA foreign_keys=ON"); db.execute("PRAGMA busy_timeout=30000")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS anchors (
      id INTEGER PRIMARY KEY, kind TEXT UNIQUE NOT NULL, label TEXT NOT NULL,
      commit_sha TEXT NOT NULL, backend TEXT, metric_json TEXT, artifact_dir TEXT,
      comparison_compatible INTEGER NOT NULL, created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS milestones (
      id INTEGER PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL, kind TEXT NOT NULL,
      lane TEXT NOT NULL, parent_id INTEGER REFERENCES milestones(id),
      attempt_id INTEGER UNIQUE, commit_sha TEXT UNIQUE NOT NULL, implementation_tree_sha TEXT NOT NULL,
      owner_set_json TEXT NOT NULL, ownership_fraction REAL NOT NULL, runtime_fla_free INTEGER NOT NULL,
      label TEXT NOT NULL, reason TEXT NOT NULL, measurement_json TEXT, created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS campaign_state (
      singleton INTEGER PRIMARY KEY CHECK(singleton=1), current_milestone_id INTEGER NOT NULL REFERENCES milestones(id), updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS attempts (
      id INTEGER PRIMARY KEY, created_at REAL NOT NULL, status TEXT NOT NULL,
      lane TEXT NOT NULL, parent_milestone_id INTEGER NOT NULL REFERENCES milestones(id),
      base_sha TEXT NOT NULL, candidate_sha TEXT NOT NULL, candidate_tree_sha TEXT NOT NULL,
      hypothesis TEXT NOT NULL, changed_paths_json TEXT NOT NULL, patch_sha TEXT NOT NULL,
      patch_text TEXT NOT NULL, protocol_sha TEXT NOT NULL, migration_decision TEXT,
      performance_decision TEXT, eligibility_decision TEXT, milestone_id INTEGER REFERENCES milestones(id),
      reason TEXT, summary_json TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS cuda_attempt_commit_protocol ON attempts(candidate_sha,protocol_sha);
    CREATE TABLE IF NOT EXISTS phases (
      id INTEGER PRIMARY KEY, attempt_id INTEGER NOT NULL REFERENCES attempts(id),
      role TEXT NOT NULL, phase TEXT NOT NULL, ordinal INTEGER NOT NULL, status TEXT NOT NULL,
      started_at REAL NOT NULL, finished_at REAL NOT NULL, returncode INTEGER,
      artifact_path TEXT, metric_json TEXT, reason TEXT,
      UNIQUE(attempt_id,role,phase,ordinal)
    );
    CREATE TABLE IF NOT EXISTS paired_blocks (
      id INTEGER PRIMARY KEY, attempt_id INTEGER NOT NULL REFERENCES attempts(id), ordinal INTEGER NOT NULL,
      execution_order TEXT NOT NULL, baseline_tps REAL, candidate_tps REAL,
      baseline_peak_mb REAL, candidate_peak_mb REAL, baseline_artifact TEXT, candidate_artifact TEXT,
      status TEXT NOT NULL, UNIQUE(attempt_id,ordinal)
    );
    CREATE TABLE IF NOT EXISTS release_runs (
      id INTEGER PRIMARY KEY, milestone_id INTEGER NOT NULL REFERENCES milestones(id), created_at REAL NOT NULL,
      status TEXT NOT NULL, candidate_sha TEXT NOT NULL, fixed_anchor_sha TEXT NOT NULL,
      decision TEXT, reason TEXT, artifact_dir TEXT UNIQUE NOT NULL, summary_json TEXT
    );
    CREATE TABLE IF NOT EXISTS release_blocks (
      id INTEGER PRIMARY KEY, release_id INTEGER NOT NULL REFERENCES release_runs(id), ordinal INTEGER NOT NULL,
      execution_order TEXT NOT NULL, baseline_tps REAL, candidate_tps REAL,
      baseline_peak_mb REAL, candidate_peak_mb REAL, status TEXT NOT NULL,
      UNIQUE(release_id,ordinal)
    );
    CREATE TABLE IF NOT EXISTS release_phases (
      id INTEGER PRIMARY KEY, release_id INTEGER NOT NULL REFERENCES release_runs(id), role TEXT NOT NULL,
      phase TEXT NOT NULL, ordinal INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT, artifact TEXT,
      payload_json TEXT, created_at REAL NOT NULL, UNIQUE(release_id,role,phase,ordinal)
    );
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY, attempt_id INTEGER REFERENCES attempts(id), at REAL NOT NULL,
      event TEXT NOT NULL, detail_json TEXT NOT NULL
    );
    CREATE TRIGGER IF NOT EXISTS milestones_no_update BEFORE UPDATE ON milestones BEGIN SELECT RAISE(ABORT,'milestones are append-only'); END;
    CREATE TRIGGER IF NOT EXISTS milestones_no_delete BEFORE DELETE ON milestones BEGIN SELECT RAISE(ABORT,'milestones are append-only'); END;
    CREATE TRIGGER IF NOT EXISTS anchors_no_update BEFORE UPDATE ON anchors BEGIN SELECT RAISE(ABORT,'anchors are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS anchors_no_delete BEFORE DELETE ON anchors BEGIN SELECT RAISE(ABORT,'anchors are immutable'); END;
    """)
    return db


def _controller_hashes(repo:Path,controller:str)->dict[str,str]:
    hashes={}
    for relative in CONTROLLER_PATHS:
        result=subprocess.run(["git","show",f"{controller}:{relative}"],cwd=repo,capture_output=True)
        if result.returncode!=0:
            raise ValueError(f"protected controller file is missing from pinned controller: {relative}")
        target=repo/relative
        if not target.is_file() or target.read_bytes()!=result.stdout:
            raise ValueError(f"protected controller file differs from pinned controller: {relative}")
        hashes[relative]=hashlib.sha256(result.stdout).hexdigest()
    return hashes


def _verify_pinned_refs(repo: Path, config: KdaCudaCampaignConfig, stored: dict[str,str]) -> tuple[str,str,str]:
    foundation = _commit(repo, config.campaign.foundation_ref)
    controller = _commit(repo, config.campaign.controller_ref)
    cumulative = _commit(repo, config.campaign.cumulative_performance_anchor_ref)
    if stored.get("foundation_sha") not in {None, foundation}:
        raise ValueError("foundation ref moved after ledger initialization")
    if stored.get("controller_sha") not in {None, controller}:
        raise ValueError("controller ref moved after ledger initialization")
    if stored.get("cumulative_performance_anchor_sha") not in {None, cumulative}:
        raise ValueError("cumulative performance anchor ref moved after ledger initialization")
    hashes=_controller_hashes(repo,controller)
    encoded=json.dumps(hashes,sort_keys=True)
    if stored.get("controller_hashes_json") not in {None,encoded}:
        raise ValueError("protected controller hashes differ from initialized ledger")
    return foundation, controller, cumulative


def initialize(root: str | Path, config: KdaCudaCampaignConfig, ledger=None) -> dict[str,Any]:
    repo=Path(root).resolve(); path=_ledger_path(repo,config,ledger); sha=protocol_sha(config)
    existing={}
    if path.exists() and path.stat().st_size:
        db=_open(path,readonly=True)
        try: existing=dict(db.execute("SELECT key,value FROM metadata").fetchall())
        finally: db.close()
    foundation,controller,cumulative=_verify_pinned_refs(repo,config,existing)
    db=_open(path)
    try:
        if existing.get("protocol_sha") not in {None,sha}: raise ValueError("ledger protocol hash differs from frozen CUDA config")
        db.execute("BEGIN IMMEDIATE")
        entries={"schema":SCHEMA,"ledger_revision":LEDGER_REVISION,"protocol_sha":sha,
                 "config_json":_canonical(config.to_dict()).decode(),"foundation_sha":foundation,"controller_sha":controller,
                 "controller_hashes_json":json.dumps(_controller_hashes(repo,controller),sort_keys=True),
                 "cumulative_performance_anchor_sha":cumulative}
        for key,value in entries.items(): db.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES(?,?)",(key,value))
        root_row=db.execute("SELECT id FROM milestones WHERE ordinal=0").fetchone()
        if root_row is None:
            cursor=db.execute("INSERT INTO milestones(ordinal,kind,lane,parent_id,attempt_id,commit_sha,implementation_tree_sha,owner_set_json,ownership_fraction,runtime_fla_free,label,reason,measurement_json,created_at) VALUES(0,'foundation','foundation',NULL,NULL,?,?,?,0,0,'Protected CUDA foundation','Campaign root; no candidate backend retained',NULL,?)",
                              (foundation,_tree(repo,foundation),json.dumps([]),time.time()))
            root_id=int(cursor.lastrowid)
            db.execute("INSERT INTO campaign_state(singleton,current_milestone_id,updated_at) VALUES(1,?,?)",(root_id,time.time()))
            db.execute("INSERT INTO anchors(kind,label,commit_sha,backend,metric_json,artifact_dir,comparison_compatible,created_at) VALUES('cuda_foundation','Protected CUDA campaign foundation',?,NULL,NULL,NULL,0,?)",(foundation,time.time()))
            db.execute("INSERT INTO anchors(kind,label,commit_sha,backend,metric_json,artifact_dir,comparison_compatible,created_at) VALUES('cumulative_performance','Fixed retained FLA performance anchor',?,'fla_triton',NULL,NULL,1,?)",(cumulative,time.time()))
        if db.execute("SELECT COUNT(*) FROM campaign_state WHERE singleton=1").fetchone()[0]!=1: raise RuntimeError("incomplete CUDA campaign initialization")
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction: db.execute("ROLLBACK")
        raise
    finally: db.close()
    return {"status":"ready","ledger":str(path),"protocol_sha":sha,"foundation_sha":foundation,"controller_sha":controller,
            "cumulative_performance_anchor_sha":cumulative,"schema":SCHEMA,"ledger_revision":2}


def _allowed(path: str, roots: tuple[str,...]) -> bool:
    return any(path == root.rstrip("/") or path.startswith(root) for root in roots)


def _event(db,attempt,event,detail):
    db.execute("INSERT INTO events(attempt_id,at,event,detail_json) VALUES(?,?,?,?)",(attempt,time.time(),event,json.dumps(detail,sort_keys=True)))


def _head(db) -> dict[str,Any]:
    row=db.execute("SELECT m.id,m.ordinal,m.kind,m.lane,m.commit_sha,m.owner_set_json,m.ownership_fraction,m.runtime_fla_free FROM campaign_state s JOIN milestones m ON m.id=s.current_milestone_id WHERE s.singleton=1").fetchone()
    if not row: raise ValueError("campaign has no retained head")
    return {"id":row[0],"ordinal":row[1],"kind":row[2],"lane":row[3],"commit_sha":row[4],"owner_set":set(json.loads(row[5])),"ownership_fraction":row[6],"runtime_fla_free":bool(row[7])}


def _derived_lane(head: dict[str,Any], config: KdaCudaCampaignConfig) -> str:
    if head["kind"] == "foundation": return "bootstrap"
    if head["owner_set"] != set(config.ownership.required_components): return "migration"
    return "optimization"


def intake(root:str|Path,config:KdaCudaCampaignConfig,base_ref:str,candidate_ref:str,hypothesis:str,ledger=None)->dict[str,Any]:
    repo=Path(root).resolve(); _clean(repo); ready=initialize(repo,config,ledger); db=_open(Path(ready["ledger"]))
    try:
        head=_head(db); lane=_derived_lane(head,config); base,candidate=_commit(repo,base_ref),_commit(repo,candidate_ref); reason=None
        if base != head["commit_sha"]: reason="base ref is not the current retained milestone"
        elif base == candidate: reason="candidate commit must differ from base"
        elif subprocess.run(["git","merge-base","--is-ancestor",base,candidate],cwd=repo).returncode: reason="base must be an ancestor of candidate"
        changed=[line for line in _git(repo,["diff","--no-renames","--name-only",f"{base}..{candidate}"]).splitlines() if line]
        if reason is None and (not changed or not all(_allowed(item,config.campaign.candidate_paths) for item in changed)):
            reason="candidate has no changes or changes paths outside the CUDA/build-source scope"
        if reason is None and any(item in READ_ONLY_CANDIDATE_PATHS for item in changed):
            reason="candidate changed protected read-only onboarding documentation"
        patch=_git(repo,["diff","--binary",f"{base}..{candidate}"])
        if reason is None and len(patch.encode()) > config.campaign.max_patch_bytes: reason="candidate patch exceeds frozen byte cap"
        for item in changed:
            exists=subprocess.run(["git","cat-file","-e",f"{candidate}:{item}"],cwd=repo).returncode==0
            candidate_type=_git(repo,["cat-file","-t",f"{candidate}:{item}"]) if exists else "deleted"
            tree_entry=_git(repo,["ls-tree",candidate,"--",item]).split(None,1)[0] if exists else ""
            suffix=Path(item).suffix.lower(); allowed=set(config.ownership.source_extensions)|{".json"}
            if reason is None and (candidate_type not in {"blob","deleted"} or tree_entry in {"120000","160000"} or suffix in FORBIDDEN_GENERATED_SUFFIXES):
                reason="candidate contains a symlink/tree/submodule or generated binary artifact"
            elif reason is None and candidate_type!="deleted" and suffix not in allowed:
                reason="candidate source suffix is outside the frozen CUDA/build allowlist"
        existing=db.execute("SELECT id,status FROM attempts WHERE candidate_sha=? AND protocol_sha=?",(candidate,ready["protocol_sha"])).fetchone()
        if existing: return {"attempt_id":existing[0],"status":"already_recorded","reason":f"candidate already {existing[1]}"}
        used=db.execute("SELECT COUNT(*) FROM attempts WHERE protocol_sha=? AND status!='rejected'",(ready["protocol_sha"],)).fetchone()[0]
        if reason is None and used>=config.campaign.max_attempts: reason="frozen maximum attempt budget exhausted"
        status="accepted" if reason is None else "rejected"; patch_sha=hashlib.sha256(patch.encode()).hexdigest()
        cursor=db.execute("INSERT INTO attempts(created_at,status,lane,parent_milestone_id,base_sha,candidate_sha,candidate_tree_sha,hypothesis,changed_paths_json,patch_sha,patch_text,protocol_sha,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(),status,lane,head["id"],base,candidate,_tree(repo,candidate),hypothesis,json.dumps(changed),patch_sha,patch,ready["protocol_sha"],reason))
        attempt=int(cursor.lastrowid); _event(db,attempt,"intake",{"status":status,"lane":lane,"reason":reason,"changed_paths":changed})
        return {"attempt_id":attempt,"status":status,"lane":lane,"reason":reason,"parent_milestone_id":head["id"],"base_sha":base,"candidate_sha":candidate,"changed_paths":changed,"patch_sha256":patch_sha}
    finally: db.close()


def paired_interval(baseline:list[float],candidate:list[float])->dict[str,float]:
    if len(baseline)!=len(candidate) or len(baseline)<7 or any(x<=0 for x in baseline+candidate): raise ValueError("paired positive measurements require at least seven interleaved blocks")
    logs=[math.log(c/b) for b,c in zip(baseline,candidate)]; mean=statistics.mean(logs); n=len(logs); se=statistics.stdev(logs)/math.sqrt(n)
    criticals={6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,21:2.080,22:2.074,23:2.069,24:2.064,25:2.060,26:2.056,27:2.052,28:2.048,29:2.045}
    critical=criticals.get(n-1,1.96)
    return {"blocks":n,"geometric_relative_change":math.exp(mean)-1,"mean_log_ratio":mean,
            "ci95_low":math.exp(mean-critical*se)-1,"ci95_high":math.exp(mean+critical*se)-1,
            "baseline_median":statistics.median(baseline),"candidate_median":statistics.median(candidate)}


def _owner_set(audit:dict[str,Any])->set[str]:
    return {name for name,item in audit.get("provenance",{}).get("components",{}).items() if item.get("owner")=="project"}


def decide_migration(audit:dict[str,Any],config:KdaCudaCampaignConfig,parent_owner_set:set[str]|None=None,lane:str="migration")->tuple[str,str]:
    if audit.get("status")!="complete": return "invalid","runtime ownership/correctness audit failed"
    owners=_owner_set(audit); parent=parent_owner_set or set(); required=set(config.ownership.required_components)
    if lane=="bootstrap":
        if audit.get("provenance",{}).get("owned_fraction",0.0)>=config.bootstrap.minimum_owned_fraction: return "ownership_progress","correct native CUDA bootstrap established"
        return "not_migrated","bootstrap owns no weighted native CUDA unit"
    if lane=="migration":
        if not owners>parent: return "no_ownership_progress","candidate owner set is not a strict superset of its retained parent"
        if owners==required and audit.get("runtime_fla_free"): return "migration_ready","all KDA units are project-owned and FLA-free"
        return "ownership_progress","project-owned CUDA coverage strictly increased"
    if owners==required and audit.get("runtime_fla_free"): return "migration_ready","optimization backend remains fully project-owned and FLA-free"
    return "invalid","optimization lost project ownership or used FLA"


def decide_performance(interval:dict[str,float],config:KdaCudaCampaignConfig,*,peak_memory_regression:float=0.0,kernel_latency_regression:float=0.0,baseline_drift:float=0.0)->tuple[str,str]:
    m=config.measurement
    if baseline_drift>m.max_baseline_drift_fraction: return "retest","baseline drift exceeded the frozen one-percent limit"
    if peak_memory_regression>m.max_peak_memory_regression_fraction: return "regressed","peak allocated memory exceeded the retention limit"
    if kernel_latency_regression>m.max_kernel_latency_regression_fraction: return "regressed","a required KDA kernel lane exceeded the latency limit"
    if interval["ci95_high"] < -m.retention_margin_fraction: return "regressed","95% paired interval is below the retention floor"
    if interval["ci95_low"] >= m.discovery_effect_fraction: return "improved","95% paired interval exceeds the discovery effect"
    if interval["ci95_low"] >= -m.retention_margin_fraction: return "retained","95% paired interval excludes a material regression"
    return "retest","paired interval cannot establish performance retention"


def decide_promotion(migration:str,performance:str,interval:dict[str,float],config:KdaCudaCampaignConfig,promotion:bool=True)->tuple[str,str]:
    if not promotion: return "not_requested","release verification was not requested"
    if migration!="migration_ready": return "blocked","CUDA migration decision is not ready"
    if performance not in {"improved","retained"}: return "blocked","release resource/performance retention gates failed"
    if interval["ci95_low"]>=config.measurement.promotion_cumulative_fraction: return "promotion_ready","fixed-anchor lower confidence bound reaches three percent"
    return "not_promoted","fixed-anchor cumulative three-percent gate was not reached"


def _bounded(command:list[str],cwd:Path,log:Path,timeout:float,env_extra=None)->dict[str,Any]:
    log.parent.mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env.update(env_extra or {}); started=time.monotonic()
    with log.open("w",encoding="utf-8") as handle:
        try: process=subprocess.Popen(command,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        except OSError as error: return {"status":"launch_error","reason":str(error),"seconds":time.monotonic()-started}
        try:
            code=process.wait(timeout=timeout)
            result={"status":"complete" if code==0 else "crash","returncode":code,"seconds":time.monotonic()-started}
            if code: result["reason"]=f"process exited with code {code}"
            return result
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL)
            return {"status":"timeout","seconds":time.monotonic()-started,"reason":f"timeout after {timeout}s"}


@contextmanager
def _worktree(repo:Path,commit:str)->Iterator[Path]:
    directory=Path(tempfile.mkdtemp(prefix="kda-cuda-ownership-"))
    try:
        _git(repo,["worktree","add","--detach",str(directory),commit]); yield directory
    finally:
        subprocess.run(["git","worktree","remove","--force",str(directory)],cwd=repo,capture_output=True,text=True)
        if directory.exists(): import shutil; shutil.rmtree(directory,ignore_errors=True)


def _phase(db,attempt,role,phase,ordinal,result,artifact,metric=None):
    now=time.time(); db.execute("INSERT INTO phases(attempt_id,role,phase,ordinal,status,started_at,finished_at,returncode,artifact_path,metric_json,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (attempt,role,phase,ordinal,result.get("status","invalid"),now-result.get("seconds",0),now,result.get("returncode"),str(artifact),json.dumps(metric,sort_keys=True) if metric is not None else None,result.get("reason")))


def _read_json(path:Path)->dict[str,Any]:
    try: return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError) as error: return {"status":"invalid","reason":str(error)}


def _resolved_artifacts(repo:Path,config:KdaCudaCampaignConfig,artifact:Path,candidate:str)->tuple[Path,dict[str,str]]:
    resolved=artifact/"resolved-config.json"; resolved.write_text(json.dumps(config.to_dict(),indent=2,sort_keys=True)+"\n")
    environment=environment_provenance()
    for tool in ("nvcc","compute-sanitizer"):
        executable=__import__("shutil").which(tool); probe=subprocess.run([executable,"--version"],text=True,capture_output=True) if executable else None
        environment[f"{tool}_path"]=executable; environment[f"{tool}_version"]=(probe.stdout+probe.stderr).strip() if probe else None
    (artifact/"environment.json").write_text(json.dumps(environment,indent=2,sort_keys=True)+"\n")
    env={"TORCH_EXTENSIONS_DIR":str((artifact/"extension-cache"/candidate).resolve()),"CUDA_CACHE_PATH":str((artifact/"cuda-cache"/candidate).resolve())}
    return resolved,env


def _worker_invocation(command:str,implementation_root:Path,env:dict[str,str],protected_dispatch:bool=True)->tuple[list[str],dict[str,str]]:
    # The coordinator always owns the runner. Candidate runs also pin the
    # protected dispatcher; historical anchor runs intentionally load the
    # retained implementation so their operator timing remains exact.
    controller_root=Path(__file__).resolve().parents[2]
    bridge_flag="--implementation-root" if protected_dispatch else "--historical-implementation-root"
    args=[sys.executable,str(controller_root/"nanochat/research/cuda_worker.py"),command,
          bridge_flag,str(implementation_root)]
    bridged={**env,"PYTHONPATH":str(controller_root)+((os.pathsep+os.environ["PYTHONPATH"]) if os.environ.get("PYTHONPATH") else "")}
    return args,bridged


def _worker(command:str,backend:str,lane:str,root:Path,config_path:Path,output:Path,log:Path,timeout:float,env:dict[str,str],protected_dispatch:bool=True)->tuple[dict[str,Any],dict[str,Any]]:
    args,env=_worker_invocation(command,root,env,protected_dispatch)
    args += ["--backend",backend,"--lane",lane,"--config",str(config_path),"--output",str(output)]
    result=_bounded(args,root,log,timeout,env); return result,_read_json(output)


def _declared_kernel_symbols(audit:dict[str,Any])->list[str]:
    return [symbol for component in audit.get("provenance",{}).get("components",{}).values()
            if component.get("owner")=="project" for symbol in component.get("kernel_symbols",[])]


def _profile_worker(lane:str,implementation_root:Path,config_path:Path,output:Path,log:Path,
                    timeout:float,env:dict[str,str],expected_symbols:list[str])->tuple[dict[str,Any],dict[str,Any]]:
    from nanochat.research.cuda_preflight import capture_nsys_cuda_symbols
    worker,bridged=_worker_invocation("profile-audit",implementation_root,env)
    command=[*worker,"--backend","project_cuda","--lane",lane,"--config",str(config_path),"--output",str(output)]
    started=time.monotonic()
    try:
        evidence=capture_nsys_cuda_symbols(command,expected_symbols=expected_symbols,cwd=implementation_root,env={**os.environ,**bridged},timeout=timeout)
        result={"status":"complete","returncode":0,"seconds":time.monotonic()-started}
        log.write_text(json.dumps({"profiler_backend":"nsys","kernel_evidence":evidence},indent=2,sort_keys=True)+"\n")
    except Exception as error:
        result={"status":"invalid","reason":f"Nsight profile failed: {type(error).__name__}: {error}","seconds":time.monotonic()-started}
        log.write_text(result["reason"]+"\n")
        evidence={}
    payload=_read_json(output)
    if result["status"]=="complete" and payload.get("status")!="complete":
        result={**result,"status":"invalid","reason":payload.get("reason","profile worker returned invalid payload")}
    if result["status"]=="complete":
        payload={**payload,"profiler_backend":"nsys","observed_kernel_symbols":sorted(evidence),"nsys_kernel_evidence":evidence}
        output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return result,payload


def _sanitizer_worker(tool:str,lane:str,implementation_root:Path,config_path:Path,
                      output:Path,log:Path,timeout:float,env:dict[str,str])->tuple[dict[str,Any],dict[str,Any]]:
    worker,env=_worker_invocation("sanitizer-smoke",implementation_root,env)
    command=["compute-sanitizer","--tool",tool,"--error-exitcode=99",*worker,
             "--backend","project_cuda","--lane",lane,"--config",str(config_path),"--output",str(output)]
    result=_bounded(command,implementation_root,log,timeout,env)
    payload=_read_json(output)
    log_text=log.read_text(errors="replace") if log.exists() else ""
    from nanochat.research.cuda_preflight import sanitizer_zero_summary
    if result["status"]=="complete" and not sanitizer_zero_summary(tool,log_text):
        result={**result,"status":"invalid","reason":"sanitizer log lacks zero-error summary"}
    if result["status"]=="complete" and payload.get("status")!="complete":
        result={**result,"status":"invalid","reason":payload.get("reason","sanitizer worker returned invalid payload")}
    return result,payload


def _failure_detail(phase:str,result:dict[str,Any],payload:dict[str,Any]|None=None,artifact:Path|None=None)->dict[str,Any]:
    payload=payload or {}
    reason=payload.get("reason") or result.get("reason") or f"{phase} ended with status {result.get('status','invalid')}"
    detail={"phase":phase,"status":result.get("status","invalid"),"reason":reason}
    if result.get("returncode") is not None: detail["returncode"]=result["returncode"]
    if artifact is not None: detail["artifact"]=str(artifact)
    return detail


def _trainer_command(config:KdaCudaCampaignConfig,backend:str,label:str)->list[str]:
    m=config.measurement; iterations=m.warmup_steps+m.timed_steps
    return [sys.executable,"-m","scripts.base_train","--seed",str(config.campaign.seed),"--depth",str(m.depth),"--head-dim",str(m.head_dim),"--window-pattern","K","--kda-backend",backend,"--no-force-final-full","--max-seq-len",str(m.sequence_length),"--device-batch-size",str(m.device_batch_size),"--total-batch-size",str(m.total_batch_size),"--num-iterations",str(iterations),"--eval-every","-1","--core-metric-every","-1","--sample-every","-1","--save-every","-1","--model-tag",label,"--run","dummy"]


def _training(root:Path,config:KdaCudaCampaignConfig,backend:str,artifact:Path,label:str,cache:Path)->dict[str,Any]:
    env={"TORCH_COMPILE_DISABLE":"1","NANOCHAT_DTYPE":"bfloat16","TORCH_EXTENSIONS_DIR":str(cache),"FLA_FLASH_KDA":"0","FLA_TILELANG":"0"}
    result=_bounded(_trainer_command(config,backend,label),root,artifact,config.measurement.block_timeout_seconds,env)
    if result["status"]!="complete": return result
    steps=[]; training_summary=None
    for line in artifact.read_text(errors="replace").splitlines():
        if line.startswith("RESEARCH_TRAIN_STEP "):
            try: steps.append(json.loads(line[len("RESEARCH_TRAIN_STEP "):]))
            except json.JSONDecodeError: pass
        elif line.startswith("RESEARCH_TRAIN_RESULT "):
            try: training_summary=json.loads(line[len("RESEARCH_TRAIN_RESULT "):])
            except json.JSONDecodeError: pass
    timed=steps[config.measurement.warmup_steps:]
    if len(timed)!=config.measurement.timed_steps: return {**result,"status":"invalid","reason":"missing timed training steps"}
    values=[float(row["tokens_per_second"]) for row in timed]; peak=float(training_summary.get("peak_memory_mb")) if training_summary else float("nan")
    if any(not math.isfinite(v) or v<=0 for v in values) or not math.isfinite(peak) or peak<=0: return {**result,"status":"invalid","reason":"non-finite throughput or peak memory"}
    return {**result,"tokens_per_second":statistics.median(values),"values":values,"peak_memory_mb":peak}


def _kernel_rows(payload:dict[str,Any])->dict[tuple[str,int],dict[str,Any]]:
    return {(row["operation"],int(row["length"])):row for row in payload.get("microbenchmarks",[]) if row.get("median_ms",0)>0}


def calibrate_anchors(root:str|Path,config:KdaCudaCampaignConfig,ledger=None)->dict[str,Any]:
    repo=Path(root).resolve(); _clean(repo); ready=initialize(repo,config,ledger); db=_open(Path(ready["ledger"]))
    try:
        existing={row[0] for row in db.execute("SELECT kind FROM anchors")}
        calibrated={"python_reference","fla_reference"}
        if calibrated<=existing: return {"status":"already_calibrated","ledger":ready["ledger"]}
        if existing & calibrated: raise RuntimeError("partial anchor calibration detected; ledger requires protected recovery")
        if db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]: raise ValueError("anchors must be calibrated before candidate intake")
        foundation=ready["foundation_sha"]; parent=repo/config.campaign.artifact_root/ready["protocol_sha"][:12]/"anchor-calibrations"; parent.mkdir(parents=True,exist_ok=True)
        ordinal=1
        while (parent/f"attempt-{ordinal:04d}").exists(): ordinal+=1
        artifact=parent/f"attempt-{ordinal:04d}"; artifact.mkdir(); resolved,env=_resolved_artifacts(repo,config,artifact,foundation)
        results={}; failure=None
        with _worktree(repo,foundation) as worktree:
            for kind,backend in (("python_reference","reference"),("fla_reference","fla_triton")):
                run,payload=_worker("microbenchmark",backend,"anchor",worktree,resolved,artifact/f"{kind}.json",artifact/f"{kind}.log",config.kernel_gates.timeout_seconds,env)
                results[kind]={"execution":run,"payload":payload}
                if run["status"]!="complete" or payload.get("status")!="complete": failure=f"{kind} anchor failed: {run.get('reason') or payload.get('reason')}"; break
        manifest={"schema_version":1,"foundation_sha":foundation,"protocol_sha":ready["protocol_sha"],"status":"invalid" if failure else "complete","reason":failure,"results":results}
        (artifact/"calibration-summary.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        if failure: return {"status":"invalid","reason":failure,"artifact_dir":str(artifact),"anchors":results}
        db.execute("BEGIN IMMEDIATE")
        try:
            for kind,backend in (("python_reference","reference"),("fla_reference","fla_triton")):
                payload=results[kind]["payload"]
                db.execute("INSERT INTO anchors(kind,label,commit_sha,backend,metric_json,artifact_dir,comparison_compatible,created_at) VALUES(?,?,?,?,?,?,1,?)",
                           (kind,"Sequential PyTorch operator anchor" if kind=="python_reference" else "Retained FLA operator anchor",foundation,backend,json.dumps(payload,sort_keys=True),str(artifact),time.time()))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK"); raise
        return {"status":"calibrated","artifact_dir":str(artifact),"anchors":{kind:value["payload"] for kind,value in results.items()}}
    finally: db.close()

def run_attempt(root:str|Path,config:KdaCudaCampaignConfig,attempt_id:int,ledger=None)->dict[str,Any]:
    repo=Path(root).resolve(); _clean(repo); ready=initialize(repo,config,ledger); db=_open(Path(ready["ledger"])); testing=False
    artifact=repo/config.campaign.artifact_root/ready["protocol_sha"][:12]/f"attempt-{attempt_id:05d}"
    try:
        if db.execute("SELECT COUNT(*) FROM anchors WHERE kind IN ('python_reference','fla_reference')").fetchone()[0]!=2: raise ValueError("calibrate Python and FLA anchors before running candidates")
        row=db.execute("SELECT status,lane,parent_milestone_id,base_sha,candidate_sha,hypothesis FROM attempts WHERE id=?",(attempt_id,)).fetchone()
        if not row: raise ValueError(f"unknown attempt {attempt_id}")
        status,lane,parent_id,base,candidate,hypothesis=row
        if status!="accepted": raise ValueError(f"attempt {attempt_id} is {status}, not accepted")
        head=_head(db)
        if head["id"]!=parent_id or head["commit_sha"]!=base: raise ValueError("attempt parent is stale; current retained head advanced")
        if artifact.exists(): raise ValueError(f"attempt artifact already exists: {artifact}")
        db.execute("UPDATE attempts SET status='testing' WHERE id=?",(attempt_id,)); _event(db,attempt_id,"testing_started",{"lane":lane}); testing=True
        artifact.mkdir(parents=True); resolved,worker_env=_resolved_artifacts(repo,config,artifact,candidate)
        baseline_env={"TORCH_EXTENSIONS_DIR":str((artifact/"extension-cache"/base).resolve()),"CUDA_CACHE_PATH":str((artifact/"cuda-cache"/base).resolve())}
        with _worktree(repo,candidate) as candidate_root, _worktree(repo,base) as baseline_root:
            test=_bounded([sys.executable,"-m","pytest","-q",*config.correctness.tests],candidate_root,artifact/"correctness.log",config.correctness.pytest_timeout_seconds)
            _phase(db,attempt_id,"candidate","correctness",0,test,artifact/"correctness.log")
            first_failure=None
            if test["status"]!="complete":
                first_failure=_failure_detail("correctness",test,artifact=artifact/"correctness.log")

            audit_run={"status":"skipped","reason":"correctness did not complete"}; audit={}
            if first_failure is None:
                audit_run,audit=_worker("runtime-audit","project_cuda",lane,candidate_root,resolved,artifact/"runtime-audit.json",artifact/"runtime-audit.log",config.correctness.runtime_audit_timeout_seconds,worker_env)
                if audit_run["status"]=="complete" and audit.get("status")=="complete" and lane=="bootstrap" and audit.get("provenance",{}).get("selective_ptx"):
                    audit_run={**audit_run,"status":"invalid","reason":"bootstrap forbids selective PTX"}
                _phase(db,attempt_id,"candidate","runtime_audit",0,audit_run,artifact/"runtime-audit.json",audit)
                if audit_run["status"]!="complete" or audit.get("status")!="complete":
                    first_failure=_failure_detail("runtime_audit",audit_run,audit,artifact/"runtime-audit.json")

            profile_run={"status":"skipped","reason":"runtime audit did not complete"}; profile_audit={}
            if first_failure is None:
                profile_run,profile_audit=_profile_worker(lane,candidate_root,resolved,artifact/"ownership-profile.json",artifact/"ownership-profile.log",config.correctness.runtime_audit_timeout_seconds,worker_env,_declared_kernel_symbols(audit))
                if (artifact/"ownership-profile.json").exists() and (artifact/"ownership-profile.json").stat().st_size>config.kernel_gates.profile_max_bytes:
                    profile_run={**profile_run,"status":"invalid","reason":"ownership profile exceeds profile_max_bytes"}
                _phase(db,attempt_id,"candidate","ownership_profile",0,profile_run,artifact/"ownership-profile.json",profile_audit)
                if profile_run["status"]!="complete" or profile_audit.get("status")!="complete":
                    first_failure=_failure_detail("ownership_profile",profile_run,profile_audit,artifact/"ownership-profile.json")

            sanitizer_results=[]
            if first_failure is None:
                for ordinal,tool in enumerate(config.correctness.sanitizer_tools):
                    output=artifact/f"sanitizer-{tool}.json"; log=artifact/f"compute-sanitizer-{tool}.log"
                    result,payload=_sanitizer_worker(tool,lane,candidate_root,resolved,output,log,
                        config.correctness.compute_sanitizer_timeout_seconds,worker_env)
                    sanitizer_results.append(result)
                    _phase(db,attempt_id,"candidate",f"compute_sanitizer_{tool}",ordinal,result,log,payload)
                    if result["status"]!="complete":
                        first_failure=_failure_detail(f"compute_sanitizer_{tool}",result,payload,log)
                        break
            correctness_ok=first_failure is None
            sanitizers_ok=first_failure is None and len(sanitizer_results)==len(config.correctness.sanitizer_tools)
            if first_failure is not None:
                failure_summary={"schema_version":2,"lane":lane,"objective":"staged_project_owned_cuda_kda",
                    "quality_not_evaluated":True,"attempt_id":attempt_id,"parent_milestone_id":parent_id,
                    "base_sha":base,"candidate_sha":candidate,"hypothesis":hypothesis,
                    "first_failure":first_failure,"artifact_dir":str(artifact)}
                return _finish(db,attempt_id,"invalid","invalid","invalid","not_retainable",
                    f"mandatory {first_failure['phase']} gate failed: {first_failure['reason']}",artifact,failure_summary)

            baseline_backend="reference" if lane=="bootstrap" else "project_cuda"
            base_kernel_run,base_kernel=_worker("microbenchmark",baseline_backend,lane,baseline_root if lane!="bootstrap" else candidate_root,resolved,artifact/"baseline-kernel.json",artifact/"baseline-kernel.log",config.kernel_gates.timeout_seconds,baseline_env if lane!="bootstrap" else worker_env)
            candidate_kernel_run,candidate_kernel=_worker("microbenchmark","project_cuda",lane,candidate_root,resolved,artifact/"candidate-kernel.json",artifact/"candidate-kernel.log",config.kernel_gates.timeout_seconds,worker_env)
            _phase(db,attempt_id,"baseline","kernel_profile",0,base_kernel_run,artifact/"baseline-kernel.json",base_kernel)
            _phase(db,attempt_id,"candidate","kernel_profile",0,candidate_kernel_run,artifact/"candidate-kernel.json",candidate_kernel)
            advisory_timeout=candidate_kernel_run["status"]=="timeout" and lane in {"bootstrap","migration"}
            base_rows,candidate_rows=_kernel_rows(base_kernel),_kernel_rows(candidate_kernel)
            kernel_regressions={f"{op}:{length}":candidate_rows[(op,length)]["median_ms"]/row["median_ms"]-1 for (op,length),row in base_rows.items() if (op,length) in candidate_rows}
            max_kernel_regression=max(kernel_regressions.values(),default=0.0)
            candidate_kernel["latency_regression"]={"by_lane":kernel_regressions,"maximum":max_kernel_regression}

            has_selective_ptx=bool(audit.get("provenance",{}).get("selective_ptx")); ptx_ok=not has_selective_ptx
            if has_selective_ptx and lane!="bootstrap":
                disabled_env={**worker_env,"NANOCHAT_DISABLE_SELECTIVE_PTX":"1"}
                disabled_audit_run,disabled_audit=_worker("runtime-audit","project_cuda",lane,candidate_root,resolved,artifact/"runtime-audit-ptx-disabled.json",artifact/"runtime-audit-ptx-disabled.log",config.correctness.runtime_audit_timeout_seconds,disabled_env)
                _phase(db,attempt_id,"candidate","runtime_audit_ptx_disabled",0,disabled_audit_run,artifact/"runtime-audit-ptx-disabled.json",disabled_audit)
                disabled_run,disabled=_worker("microbenchmark","project_cuda",lane,candidate_root,resolved,artifact/"candidate-kernel-ptx-disabled.json",artifact/"candidate-kernel-ptx-disabled.log",config.kernel_gates.timeout_seconds,disabled_env)
                disabled_rows=_kernel_rows(disabled); improvements={f"{op}:{length}":disabled_rows[(op,length)]["median_ms"]/row["median_ms"]-1 for (op,length),row in candidate_rows.items() if (op,length) in disabled_rows}
                ptx_ok=candidate_kernel_run["status"]=="complete" and disabled_audit_run["status"]=="complete" and disabled_audit.get("status")=="complete" and disabled_run["status"]=="complete" and max(improvements.values(),default=-math.inf)>=config.kernel_gates.selective_ptx_min_latency_improvement
                candidate_kernel["selective_ptx_ab"]={"passed":ptx_ok,"improvements":improvements}

            migration,migration_reason=decide_migration(audit,config,head["owner_set"],lane)
            performance="observed"; performance_reason="performance is advisory in this lane"; interval=None; peak_regression=0.0; drift=0.0
            full_training_ok=True
            if lane=="optimization" and correctness_ok and sanitizers_ok and candidate_kernel_run["status"]=="complete" and base_kernel_run["status"]=="complete" and ptx_ok:
                baseline_values=[]; candidate_values=[]; baseline_memory=[]; candidate_memory=[]
                for block in range(config.measurement.discovery_paired_blocks):
                    order=(("baseline",baseline_root,"project_cuda"),("candidate",candidate_root,"project_cuda"))
                    if block%2: order=tuple(reversed(order))
                    results={}
                    for role,worktree,backend in order:
                        log=artifact/f"block-{block:02d}-{role}.log"; result=_training(worktree,config,backend,log,f"cuda-{attempt_id}-{block}-{role}",artifact/"training-cache"/role)
                        results[role]=result; _phase(db,attempt_id,role,"interleaved_training",block,result,log,{"tokens_per_second":result.get("tokens_per_second"),"order":[item[0] for item in order]})
                        if result["status"]!="complete": break
                    complete=len(results)==2 and all(item["status"]=="complete" for item in results.values())
                    db.execute("INSERT INTO paired_blocks(attempt_id,ordinal,execution_order,baseline_tps,candidate_tps,baseline_peak_mb,candidate_peak_mb,baseline_artifact,candidate_artifact,status) VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (attempt_id,block,json.dumps([item[0] for item in order]),results.get("baseline",{}).get("tokens_per_second"),results.get("candidate",{}).get("tokens_per_second"),results.get("baseline",{}).get("peak_memory_mb"),results.get("candidate",{}).get("peak_memory_mb"),str(artifact/f"block-{block:02d}-baseline.log"),str(artifact/f"block-{block:02d}-candidate.log"),"complete" if complete else "invalid"))
                    if not complete: full_training_ok=False; break
                    baseline_values.append(results["baseline"]["tokens_per_second"]); candidate_values.append(results["candidate"]["tokens_per_second"]); baseline_memory.append(results["baseline"]["peak_memory_mb"]); candidate_memory.append(results["candidate"]["peak_memory_mb"])
                if full_training_ok:
                    interval=paired_interval(baseline_values,candidate_values); peak_regression=statistics.median(candidate_memory)/statistics.median(baseline_memory)-1; midpoint=len(baseline_values)//2; drift=abs(statistics.median(baseline_values[midpoint:])/statistics.median(baseline_values[:midpoint])-1)
                    performance,performance_reason=decide_performance(interval,config,peak_memory_regression=peak_regression,kernel_latency_regression=max_kernel_regression,baseline_drift=drift)

        hard_ok=correctness_ok and sanitizers_ok and ptx_ok
        if lane=="optimization": hard_ok=hard_ok and full_training_ok and base_kernel_run["status"]=="complete" and candidate_kernel_run["status"]=="complete"
        else: hard_ok=hard_ok and (candidate_kernel_run["status"]=="complete" or advisory_timeout)
        if not hard_ok:
            return _finish(db,attempt_id,"invalid","invalid","invalid","not_retainable","protected correctness/safety/evidence gate failed",artifact,None)
        owners=sorted(_owner_set(audit)); required=set(config.ownership.required_components)
        if lane=="bootstrap": eligibility="correct_bootstrap" if migration=="ownership_progress" else "not_retainable"
        elif lane=="migration": eligibility="fla_free_naive" if set(owners)==required and audit.get("runtime_fla_free") else ("validated_component" if migration=="ownership_progress" else "not_retainable")
        else: eligibility="optimization_retained" if migration=="migration_ready" and performance=="improved" else "not_retainable"
        summary={"schema_version":2,"lane":lane,"objective":"staged_project_owned_cuda_kda","quality_not_evaluated":True,"attempt_id":attempt_id,"parent_milestone_id":parent_id,"base_sha":base,"candidate_sha":candidate,"hypothesis":hypothesis,
                 "migration":{"decision":migration,"reason":migration_reason,"owner_set":owners,"owned_fraction":audit.get("provenance",{}).get("owned_fraction",0.0),"runtime_fla_free":audit.get("runtime_fla_free",False),"new_components":sorted(set(owners)-head["owner_set"])},
                 "performance":{"decision":performance,"reason":performance_reason,"advisory":lane in {"bootstrap","migration"},"paired_interval":interval,"peak_memory_regression":peak_regression,"kernel_latency_regression":max_kernel_regression,"baseline_drift":drift,"candidate_kernel_status":candidate_kernel_run["status"],"censored_timeout_seconds":config.kernel_gates.timeout_seconds if candidate_kernel_run["status"]=="timeout" else None},
                 "eligibility_decision":eligibility,"baseline_backend":baseline_backend,"baseline_kernel":base_kernel,"candidate_kernel":candidate_kernel,"artifact_dir":str(artifact)}
        return _finish(db,attempt_id,"complete",migration,performance,eligibility,"all mandatory lane gates completed",artifact,summary)
    except Exception as error:
        if testing: return _finish(db,attempt_id,"invalid","invalid","invalid","not_retainable",f"supervisor exception: {type(error).__name__}: {error}",artifact,None)
        raise
    finally: db.close()


def _finish(db,attempt,status,migration,performance,eligibility,reason,artifact,summary):
    db.execute("UPDATE attempts SET status=?,migration_decision=?,performance_decision=?,eligibility_decision=?,reason=?,summary_json=? WHERE id=?",(status,migration,performance,eligibility,reason,json.dumps(summary,sort_keys=True) if summary else None,attempt)); _event(db,attempt,"terminal",{"status":status,"reason":reason,"eligibility":eligibility})
    return {"attempt_id":attempt,"status":status,"migration_decision":migration,"performance_decision":performance,"eligibility_decision":eligibility,"reason":reason,"summary":summary,"artifact_dir":str(artifact)}


def retain(root:str|Path,config:KdaCudaCampaignConfig,attempt_id:int,label:str,reason:str,commit_ref:str|None=None,ledger=None)->dict[str,Any]:
    repo=Path(root).resolve(); _clean(repo); ready=initialize(repo,config,ledger); db=_open(Path(ready["ledger"]))
    try:
        row=db.execute("SELECT status,lane,parent_milestone_id,candidate_sha,candidate_tree_sha,eligibility_decision,summary_json,milestone_id FROM attempts WHERE id=?",(attempt_id,)).fetchone()
        if not row: raise ValueError(f"unknown attempt {attempt_id}")
        status,lane,parent_id,candidate,candidate_tree,eligibility,summary_json,milestone_id=row
        if milestone_id:
            milestone=db.execute("SELECT id,ordinal,kind,commit_sha FROM milestones WHERE id=?",(milestone_id,)).fetchone()
            return {"status":"already_retained","milestone_id":milestone[0],"ordinal":milestone[1],"kind":milestone[2],"commit_sha":milestone[3]}
        if status!="complete" or eligibility not in RETAINABLE or not summary_json: raise ValueError("attempt is not milestone-eligible")
        head=_head(db)
        if head["id"]!=parent_id: raise ValueError("stale attempt cannot advance the retained milestone head")
        retained_commit=_commit(repo,commit_ref or candidate)
        if retained_commit!=candidate: raise ValueError("retained milestone must be the exact measured candidate commit")
        if _tree(repo,retained_commit)!=candidate_tree: raise ValueError("retained commit changes the measured cuda_kda implementation tree")
        summary=json.loads(summary_json); owners=set(summary["migration"]["owner_set"]); required=set(config.ownership.required_components)
        if lane=="migration" and not owners>head["owner_set"]: raise ValueError("migration milestone is not a strict ownership superset")
        if lane=="optimization" and owners!=required: raise ValueError("optimization milestone lost full project ownership")
        kind="fla_free_naive" if owners==required and summary["migration"]["runtime_fla_free"] and not any(row[0]=="fla_free_naive" for row in db.execute("SELECT kind FROM milestones")) else eligibility
        ordinal=head["ordinal"]+1
        db.execute("BEGIN IMMEDIATE")
        try:
            if _head(db)["id"]!=parent_id: raise ValueError("stale attempt cannot advance the retained milestone head")
            cursor=db.execute("INSERT INTO milestones(ordinal,kind,lane,parent_id,attempt_id,commit_sha,implementation_tree_sha,owner_set_json,ownership_fraction,runtime_fla_free,label,reason,measurement_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (ordinal,kind,lane,head["id"],attempt_id,retained_commit,candidate_tree,json.dumps(sorted(owners)),summary["migration"]["owned_fraction"],int(summary["migration"]["runtime_fla_free"]),label,reason,summary_json,time.time()))
            mid=int(cursor.lastrowid); db.execute("UPDATE campaign_state SET current_milestone_id=?,updated_at=? WHERE singleton=1",(mid,time.time())); db.execute("UPDATE attempts SET milestone_id=?,status='retained' WHERE id=?",(mid,attempt_id)); _event(db,attempt_id,"retained",{"milestone_id":mid,"ordinal":ordinal,"kind":kind})
            anchor_kinds=[]
            if head["kind"]=="foundation": anchor_kinds.append("first_bootstrap")
            if kind=="fla_free_naive": anchor_kinds.append("fla_free_naive")
            for anchor_kind in anchor_kinds:
                if db.execute("SELECT 1 FROM anchors WHERE kind=?",(anchor_kind,)).fetchone() is None:
                    db.execute("INSERT INTO anchors(kind,label,commit_sha,backend,metric_json,artifact_dir,comparison_compatible,created_at) VALUES(?,?,?,?,?,?,1,?)",
                      (anchor_kind,"First retained native CUDA bootstrap" if anchor_kind=="first_bootstrap" else "First complete FLA-free naive CUDA backend",retained_commit,"project_cuda",json.dumps(summary.get("candidate_kernel"),sort_keys=True),summary.get("artifact_dir"),time.time()))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK"); raise
        return {"status":"retained","milestone_id":mid,"ordinal":ordinal,"kind":kind,"lane":lane,"commit_sha":retained_commit,"owner_set":sorted(owners)}
    finally: db.close()


def _release_phase(db:sqlite3.Connection,release_id:int,role:str,phase:str,ordinal:int,result:dict[str,Any],artifact:Path,payload:dict[str,Any]|None=None)->None:
    db.execute("INSERT INTO release_phases(release_id,role,phase,ordinal,status,reason,artifact,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
      (release_id,role,phase,ordinal,result.get("status","invalid"),result.get("reason"),str(artifact),json.dumps(payload,sort_keys=True) if payload is not None else None,time.time()))


def verify_release(root:str|Path,config:KdaCudaCampaignConfig,milestone_id:int,ledger=None)->dict[str,Any]:
    repo=Path(root).resolve(); _clean(repo); ready=initialize(repo,config,ledger); db=_open(Path(ready["ledger"])); release_id=None
    try:
        row=db.execute("SELECT id,commit_sha,owner_set_json,runtime_fla_free FROM milestones WHERE id=?",(milestone_id,)).fetchone()
        if not row: raise ValueError(f"unknown milestone {milestone_id}")
        candidate=row[1]; owners=set(json.loads(row[2])); required=set(config.ownership.required_components)
        if owners!=required or not row[3]: raise ValueError("release verification requires a complete FLA-free project milestone")
        anchor=ready["cumulative_performance_anchor_sha"]
        provisional=f"pending-{os.getpid()}-{time.time_ns()}"; cursor=db.execute("INSERT INTO release_runs(milestone_id,created_at,status,candidate_sha,fixed_anchor_sha,artifact_dir) VALUES(?,?,'testing',?,?,?)",(milestone_id,time.time(),candidate,anchor,provisional)); release_id=int(cursor.lastrowid)
        artifact=repo/config.campaign.artifact_root/ready["protocol_sha"][:12]/f"release-{milestone_id:05d}-{release_id:05d}"
        if artifact.exists(): raise ValueError(f"release artifact unexpectedly exists: {artifact}")
        artifact.mkdir(parents=True); resolved,candidate_env=_resolved_artifacts(repo,config,artifact,candidate)
        baseline_env={"TORCH_EXTENSIONS_DIR":str((artifact/"extension-cache"/anchor).resolve()),"CUDA_CACHE_PATH":str((artifact/"cuda-cache"/anchor).resolve())}
        db.execute("UPDATE release_runs SET artifact_dir=? WHERE id=?",(str(artifact),release_id))
        baseline_values=[]; candidate_values=[]; baseline_memory=[]; candidate_memory=[]; complete=True; max_kernel_regression=math.inf
        with _worktree(repo,candidate) as candidate_root,_worktree(repo,anchor) as baseline_root:
            test=_bounded([sys.executable,"-m","pytest","-q",*config.correctness.tests],candidate_root,artifact/"correctness.log",config.correctness.pytest_timeout_seconds)
            _release_phase(db,release_id,"candidate","correctness",0,test,artifact/"correctness.log")
            first_failure=None
            if test["status"]!="complete": first_failure=_failure_detail("correctness",test,artifact=artifact/"correctness.log")
            audit_run={"status":"skipped"}; audit={}
            if first_failure is None:
                audit_run,audit=_worker("runtime-audit","project_cuda","optimization",candidate_root,resolved,artifact/"runtime-audit.json",artifact/"runtime-audit.log",config.correctness.runtime_audit_timeout_seconds,candidate_env)
                if audit_run["status"]=="complete" and audit.get("status")=="complete" and not audit.get("runtime_fla_free"):
                    audit_run={**audit_run,"status":"invalid","reason":"release runtime audit was not FLA-free"}
                _release_phase(db,release_id,"candidate","runtime_audit",0,audit_run,artifact/"runtime-audit.json",audit)
                if audit_run["status"]!="complete" or audit.get("status")!="complete":
                    first_failure=_failure_detail("runtime_audit",audit_run,audit,artifact/"runtime-audit.json")
            profile_run={"status":"skipped"}; profile={}
            if first_failure is None:
                profile_run,profile=_profile_worker("optimization",candidate_root,resolved,artifact/"ownership-profile.json",artifact/"ownership-profile.log",config.correctness.runtime_audit_timeout_seconds,candidate_env,_declared_kernel_symbols(audit))
                if (artifact/"ownership-profile.json").exists() and (artifact/"ownership-profile.json").stat().st_size>config.kernel_gates.profile_max_bytes: profile_run={**profile_run,"status":"invalid","reason":"ownership profile exceeds profile_max_bytes"}
                _release_phase(db,release_id,"candidate","ownership_profile",0,profile_run,artifact/"ownership-profile.json",profile)
                if profile_run["status"]!="complete" or profile.get("status")!="complete":
                    first_failure=_failure_detail("ownership_profile",profile_run,profile,artifact/"ownership-profile.json")
            sanitizer_results=[]
            if first_failure is None:
                for ordinal,tool in enumerate(config.correctness.sanitizer_tools):
                    output=artifact/f"sanitizer-{tool}.json"; log=artifact/f"compute-sanitizer-{tool}.log"
                    result,payload=_sanitizer_worker(tool,"optimization",candidate_root,resolved,output,log,
                        config.correctness.compute_sanitizer_timeout_seconds,candidate_env)
                    sanitizer_results.append(result); _release_phase(db,release_id,"candidate",f"compute_sanitizer_{tool}",ordinal,result,log,payload)
                    if result["status"]!="complete":
                        first_failure=_failure_detail(f"compute_sanitizer_{tool}",result,payload,log); break
            preflight=first_failure is None and len(sanitizer_results)==len(config.correctness.sanitizer_tools)
            if not preflight:
                if first_failure is None:
                    first_failure={"phase":"compute_sanitizer","status":"invalid","reason":"not every mandatory sanitizer completed"}
                reason=f"mandatory {first_failure['phase']} gate failed: {first_failure['reason']}"
                summary={"schema_version":2,"milestone_id":milestone_id,"candidate_sha":candidate,
                    "fixed_anchor_sha":anchor,"decision":"invalid","reason":reason,
                    "first_failure":first_failure,"artifact_dir":str(artifact),"quality_not_evaluated":True}
                db.execute("UPDATE release_runs SET status='invalid',decision='invalid',reason=?,summary_json=? WHERE id=?",
                    (reason,json.dumps(summary,sort_keys=True),release_id))
                return {"release_id":release_id,"status":"invalid","decision":"invalid","reason":reason,"summary":summary}
            base_kernel_run,base_kernel=_worker("microbenchmark","fla_triton","optimization",baseline_root,resolved,artifact/"fixed-anchor-kernel.json",artifact/"fixed-anchor-kernel.log",config.kernel_gates.timeout_seconds,baseline_env,protected_dispatch=False)
            candidate_kernel_run,candidate_kernel=_worker("microbenchmark","project_cuda","optimization",candidate_root,resolved,artifact/"candidate-kernel.json",artifact/"candidate-kernel.log",config.kernel_gates.timeout_seconds,candidate_env)
            _release_phase(db,release_id,"baseline","kernel_profile",0,base_kernel_run,artifact/"fixed-anchor-kernel.json",base_kernel); _release_phase(db,release_id,"candidate","kernel_profile",0,candidate_kernel_run,artifact/"candidate-kernel.json",candidate_kernel)
            base_rows,candidate_rows=_kernel_rows(base_kernel),_kernel_rows(candidate_kernel)
            regressions=[candidate_rows[key]["median_ms"]/value["median_ms"]-1 for key,value in base_rows.items() if key in candidate_rows]
            if base_kernel_run["status"]=="complete" and candidate_kernel_run["status"]=="complete" and regressions: max_kernel_regression=max(regressions)
            else: preflight=False
            if audit.get("provenance",{}).get("selective_ptx"):
                disabled_env={**candidate_env,"NANOCHAT_DISABLE_SELECTIVE_PTX":"1"}
                disabled_audit_run,disabled_audit=_worker("runtime-audit","project_cuda","optimization",candidate_root,resolved,artifact/"runtime-audit-ptx-disabled.json",artifact/"runtime-audit-ptx-disabled.log",config.correctness.runtime_audit_timeout_seconds,disabled_env)
                _release_phase(db,release_id,"candidate","runtime_audit_ptx_disabled",0,disabled_audit_run,artifact/"runtime-audit-ptx-disabled.json",disabled_audit)
                disabled_run,disabled=_worker("microbenchmark","project_cuda","optimization",candidate_root,resolved,artifact/"candidate-kernel-ptx-disabled.json",artifact/"candidate-kernel-ptx-disabled.log",config.kernel_gates.timeout_seconds,disabled_env)
                _release_phase(db,release_id,"candidate","kernel_profile_ptx_disabled",0,disabled_run,artifact/"candidate-kernel-ptx-disabled.json",disabled)
                disabled_rows=_kernel_rows(disabled); enabled_rows=_kernel_rows(candidate_kernel)
                improvements=[disabled_rows[key]["median_ms"]/value["median_ms"]-1 for key,value in enabled_rows.items() if key in disabled_rows]
                preflight=preflight and disabled_audit_run["status"]=="complete" and disabled_audit.get("status")=="complete" and disabled_run["status"]=="complete" and max(improvements,default=-math.inf)>=config.kernel_gates.selective_ptx_min_latency_improvement
            if preflight:
                for block in range(config.measurement.promotion_paired_blocks):
                    order=(("baseline",baseline_root,"fla_triton"),("candidate",candidate_root,"project_cuda"))
                    if block%2: order=tuple(reversed(order))
                    results={}
                    for role,worktree,backend in order:
                        result=_training(worktree,config,backend,artifact/f"block-{block:02d}-{role}.log",f"release-{milestone_id}-{release_id}-{block}-{role}",artifact/"cache"/role); results[role]=result
                        _release_phase(db,release_id,role,"interleaved_training",block,result,artifact/f"block-{block:02d}-{role}.log",{"tokens_per_second":result.get("tokens_per_second")})
                        if result["status"]!="complete": break
                    ok=len(results)==2 and all(item["status"]=="complete" for item in results.values()); db.execute("INSERT INTO release_blocks(release_id,ordinal,execution_order,baseline_tps,candidate_tps,baseline_peak_mb,candidate_peak_mb,status) VALUES(?,?,?,?,?,?,?,?)",(release_id,block,json.dumps([x[0] for x in order]),results.get("baseline",{}).get("tokens_per_second"),results.get("candidate",{}).get("tokens_per_second"),results.get("baseline",{}).get("peak_memory_mb"),results.get("candidate",{}).get("peak_memory_mb"),"complete" if ok else "invalid"))
                    if not ok: complete=False; break
                    baseline_values.append(results["baseline"]["tokens_per_second"]); candidate_values.append(results["candidate"]["tokens_per_second"]); baseline_memory.append(results["baseline"]["peak_memory_mb"]); candidate_memory.append(results["candidate"]["peak_memory_mb"])
            else: complete=False
        if not complete:
            decision,reason,status="invalid","release correctness/safety/kernel/training evidence failed","invalid"; summary=None
        else:
            interval=paired_interval(baseline_values,candidate_values); peak=statistics.median(candidate_memory)/statistics.median(baseline_memory)-1; midpoint=len(baseline_values)//2; drift=abs(statistics.median(baseline_values[midpoint:])/statistics.median(baseline_values[:midpoint])-1)
            performance,performance_reason=decide_performance(interval,config,peak_memory_regression=peak,kernel_latency_regression=max_kernel_regression,baseline_drift=drift); decision,reason=decide_promotion("migration_ready",performance,interval,config,True); status="complete"
            summary={"schema_version":2,"milestone_id":milestone_id,"candidate_sha":candidate,"fixed_anchor_sha":anchor,"paired_interval":interval,"peak_memory_regression":peak,"kernel_latency_regression":max_kernel_regression,"baseline_drift":drift,"performance_decision":performance,"performance_reason":performance_reason,"decision":decision,"reason":reason,"artifact_dir":str(artifact),"quality_not_evaluated":True}
        db.execute("UPDATE release_runs SET status=?,decision=?,reason=?,summary_json=? WHERE id=?",(status,decision,reason,json.dumps(summary,sort_keys=True) if summary else None,release_id))
        return {"release_id":release_id,"status":status,"decision":decision,"reason":reason,"summary":summary}
    except Exception as error:
        if release_id is not None:
            reason=f"release supervisor exception: {type(error).__name__}: {error}"; db.execute("UPDATE release_runs SET status='invalid',decision='invalid',reason=? WHERE id=?",(reason,release_id)); return {"release_id":release_id,"status":"invalid","decision":"invalid","reason":reason,"summary":None}
        raise
    finally: db.close()

def recover_interrupted(root:str|Path,config:KdaCudaCampaignConfig,attempt_id:int,reason:str,ledger=None)->dict[str,Any]:
    repo=Path(root).resolve(); _clean(repo); ready=initialize(repo,config,ledger); db=_open(Path(ready["ledger"]))
    try:
        row=db.execute("SELECT status FROM attempts WHERE id=?",(attempt_id,)).fetchone()
        if not row: raise ValueError(f"unknown attempt {attempt_id}")
        if row[0]!="testing": raise ValueError(f"attempt {attempt_id} is {row[0]}, not interrupted testing")
        detail=f"interrupted supervisor recovery: {reason}"; db.execute("UPDATE attempts SET status='invalid',migration_decision='invalid',performance_decision='invalid',eligibility_decision='not_retainable',reason=? WHERE id=?",(detail,attempt_id)); _event(db,attempt_id,"recovered_invalid",{"reason":detail})
        return {"attempt_id":attempt_id,"status":"invalid","reason":detail,"protocol_sha":ready["protocol_sha"]}
    finally: db.close()


def _metric_latency(payload:dict[str,Any]|None,config:KdaCudaCampaignConfig)->float|None:
    if not payload: return None
    for row in payload.get("microbenchmarks",[]):
        if row.get("operation")==config.reporting.canonical_operation and int(row.get("length",-1))==config.reporting.canonical_length:
            value=float(row.get("median_ms",0)); return value if value>0 else None
    return None


def _metric_map(payload:dict[str,Any]|None)->dict[str,float]:
    if not payload: return {}
    rows=payload.get("microbenchmarks",[]) if isinstance(payload,dict) else []
    return {f"{row['operation']}:{int(row['length'])}":float(row["median_ms"]) for row in rows if isinstance(row,dict) and row.get("median_ms") is not None}


def _compatible_speedups(candidate:dict[str,float],baseline:dict[str,float])->dict[str,float]:
    return {key:baseline[key]/value for key,value in candidate.items() if key in baseline and value>0 and baseline[key]>0}


def _historical_context(repo:Path,config:KdaCudaCampaignConfig)->dict[str,Any]:
    path=Path(config.reporting.historical_context_manifest)
    if not path.is_absolute(): path=repo/path
    encoded=path.read_bytes()
    if hashlib.sha256(encoded).hexdigest()!=config.reporting.historical_context_sha256: raise ValueError("historical speed context manifest hash mismatch")
    payload=json.loads(encoded)
    if payload.get("schema")!="kda_speed_historical_context" or payload.get("comparison_compatible") is not False: raise ValueError("invalid historical context manifest")
    return payload


def campaign_report(root:str|Path,config:KdaCudaCampaignConfig,ledger=None,*,include_invalid:bool=True)->dict[str,Any]:
    repo=Path(root).resolve(); path=_ledger_path(repo,config,ledger); db=_open(path,readonly=True)
    try:
        metadata=dict(db.execute("SELECT key,value FROM metadata").fetchall())
        if metadata.get("protocol_sha")!=protocol_sha(config): raise ValueError("ledger protocol hash differs from frozen CUDA config")
        anchor_rows=db.execute("SELECT kind,label,commit_sha,backend,metric_json,artifact_dir,comparison_compatible FROM anchors ORDER BY id").fetchall()
        anchors={row[0]:{"label":row[1],"commit_sha":row[2],"backend":row[3],"metrics":json.loads(row[4]) if row[4] else None,"artifact_dir":row[5],"comparison_compatible":bool(row[6])} for row in anchor_rows}
        rows=db.execute("SELECT id,ordinal,kind,lane,parent_id,attempt_id,commit_sha,owner_set_json,ownership_fraction,runtime_fla_free,label,reason,measurement_json FROM milestones ORDER BY ordinal").fetchall()
        anchor_maps={name:_metric_map(anchor.get("metrics")) for name,anchor in anchors.items()}
        milestones=[]; previous_latency=None; previous_map={}; previous_ownership=0.0; first_bootstrap_latency=_metric_latency(anchors.get("first_bootstrap",{}).get("metrics"),config); naive_latency=_metric_latency(anchors.get("fla_free_naive",{}).get("metrics"),config); python_latency=_metric_latency(anchors.get("python_reference",{}).get("metrics"),config); fla_latency=_metric_latency(anchors.get("fla_reference",{}).get("metrics"),config); chained_log=0.0
        for row in rows:
            measurement=json.loads(row[12]) if row[12] else None; latency=_metric_latency(measurement.get("candidate_kernel") if measurement else None,config)
            ratios={}
            for name,anchor_latency in (("python_reference",python_latency),("fla_reference",fla_latency),("first_bootstrap",first_bootstrap_latency),("fla_free_naive",naive_latency),("direct_parent",previous_latency)):
                if latency and anchor_latency: ratios[name]=anchor_latency/latency
            direct=(measurement or {}).get("performance",{}).get("paired_interval")
            if direct and row[3]=="optimization": chained_log+=float(direct.get("mean_log_ratio",0.0))
            candidate_map=_metric_map(measurement.get("candidate_kernel") if measurement else None)
            per_shape={name:_compatible_speedups(candidate_map,baseline_map) for name,baseline_map in {**anchor_maps,"direct_parent":previous_map}.items() if baseline_map}
            owner_set=json.loads(row[7]); new_components=sorted(set(owner_set)-set(milestones[-1]["owner_set"] if milestones else []))
            milestones.append({"milestone_id":row[0],"ordinal":row[1],"kind":row[2],"lane":row[3],"parent_id":row[4],"attempt_id":row[5],"commit_sha":row[6],"owner_set":owner_set,"new_components":new_components,"ownership_fraction":row[8],"ownership_percentage_point_change":100.0*(row[8]-previous_ownership),"runtime_fla_free":bool(row[9]),"label":row[10],"reason":row[11],"canonical_latency_ms":latency,"speedup_ratios":ratios,"per_shape_speedup_ratios":per_shape,"direct_paired_interval":direct,"illustrative_chained_optimization_point_estimate":math.exp(chained_log)-1 if chained_log else 0.0,"measurement":measurement})
            if latency: previous_latency=latency
            if candidate_map: previous_map=candidate_map
            previous_ownership=row[8]
        where="" if include_invalid else "WHERE status IN ('complete','retained')"
        attempts=[]
        for row in db.execute(f"SELECT id,status,lane,parent_milestone_id,candidate_sha,hypothesis,migration_decision,performance_decision,eligibility_decision,reason,summary_json FROM attempts {where} ORDER BY id"):
            phases=[{"role":phase[0],"phase":phase[1],"ordinal":phase[2],"status":phase[3],"reason":phase[4],"artifact":phase[5],"metric":json.loads(phase[6]) if phase[6] else None} for phase in db.execute("SELECT role,phase,ordinal,status,reason,artifact_path,metric_json FROM phases WHERE attempt_id=? ORDER BY id",(row[0],))]
            blocks=[{"ordinal":block[0],"execution_order":json.loads(block[1]),"baseline_tps":block[2],"candidate_tps":block[3],"baseline_peak_mb":block[4],"candidate_peak_mb":block[5],"status":block[6]} for block in db.execute("SELECT ordinal,execution_order,baseline_tps,candidate_tps,baseline_peak_mb,candidate_peak_mb,status FROM paired_blocks WHERE attempt_id=? ORDER BY ordinal",(row[0],))]
            attempts.append({"attempt_id":row[0],"status":row[1],"lane":row[2],"parent_milestone_id":row[3],"candidate_sha":row[4],"hypothesis":row[5],"migration_decision":row[6],"performance_decision":row[7],"eligibility_decision":row[8],"reason":row[9],"summary":json.loads(row[10]) if row[10] else None,"phases":phases,"paired_blocks":blocks})
        releases=[]
        for row in db.execute("SELECT id,milestone_id,status,decision,reason,summary_json,artifact_dir FROM release_runs ORDER BY id"):
            phases=[{"role":phase[0],"phase":phase[1],"ordinal":phase[2],"status":phase[3],"reason":phase[4],"artifact":phase[5],"payload":json.loads(phase[6]) if phase[6] else None} for phase in db.execute("SELECT role,phase,ordinal,status,reason,artifact,payload_json FROM release_phases WHERE release_id=? ORDER BY id",(row[0],))]
            blocks=[{"ordinal":block[0],"execution_order":json.loads(block[1]),"baseline_tps":block[2],"candidate_tps":block[3],"baseline_peak_mb":block[4],"candidate_peak_mb":block[5],"status":block[6]} for block in db.execute("SELECT ordinal,execution_order,baseline_tps,candidate_tps,baseline_peak_mb,candidate_peak_mb,status FROM release_blocks WHERE release_id=? ORDER BY ordinal",(row[0],))]
            releases.append({"release_id":row[0],"milestone_id":row[1],"status":row[2],"decision":row[3],"reason":row[4],"summary":json.loads(row[5]) if row[5] else None,"artifact_dir":row[6],"phases":phases,"paired_blocks":blocks})
        head=db.execute("SELECT current_milestone_id FROM campaign_state WHERE singleton=1").fetchone()[0]
        return {"schema":"kda_cuda_ownership_report","schema_version":2,"quality_not_evaluated":True,"protocol_sha":metadata["protocol_sha"],"historical_context":_historical_context(repo,config),"provenance":{"foundation_sha":metadata["foundation_sha"],"controller_sha":metadata["controller_sha"],"cumulative_performance_anchor_sha":metadata["cumulative_performance_anchor_sha"],"canonical_metric":{"operation":config.reporting.canonical_operation,"length":config.reporting.canonical_length,"unit":"milliseconds_lower_is_better"}},"anchors":anchors,"current_milestone_id":head,"milestones":milestones,"attempts":attempts,"release_runs":releases,"warnings":["Illustrative chained point estimates are not gates or confidence intervals.","Historical speed-campaign evidence is not arithmetically combined with this protocol."]}
    finally: db.close()


def render_campaign_report(report:dict[str,Any])->str:
    historical=report["historical_context"]
    lines=["# KDA CUDA-Ownership Autoresearch Report","","> Backend systems evidence only; language-model quality was not evaluated.","",f"Protocol: `{report['protocol_sha']}`","","## Historical speed context (not comparison-compatible)","",f"- Completed attempts: {historical['completed_attempts']}",f"- Terminal recorded throughput: approximately {historical['terminal_recorded_training_tokens_per_second_approx']} tok/s",f"- Protocol: `{historical['campaign_protocol_sha256']}`","","## Retained lineage","","| # | Kind | Lane | Ownership | Δ pp | FLA-free | Canonical ms | vs Python | vs FLA | vs parent | chained opt. point | New components |","|---:|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---|"]
    for item in report["milestones"]:
        ratios=item["speedup_ratios"]; lines.append(f"| {item['ordinal']} | {item['kind']} | {item['lane']} | {item['ownership_fraction']:.1%} | {item['ownership_percentage_point_change']:.1f} | {'yes' if item['runtime_fla_free'] else 'no'} | {item['canonical_latency_ms'] if item['canonical_latency_ms'] is not None else 'n/a'} | {ratios.get('python_reference','n/a')} | {ratios.get('fla_reference','n/a')} | {ratios.get('direct_parent','n/a')} | {item['illustrative_chained_optimization_point_estimate']:.4f} | {', '.join(item['new_components']) or 'none'} |")
        if item["per_shape_speedup_ratios"]:
            lines.append("")
            lines.append(f"Compatible per-shape ratios for milestone {item['ordinal']}: `{json.dumps(item['per_shape_speedup_ratios'],sort_keys=True)}`")
    lines += ["","## Attempts","", "| Attempt | Status | Lane | Eligibility | Hypothesis |", "|---:|---|---|---|---|"]
    for item in report["attempts"]: lines.append(f"| {item['attempt_id']} | {item['status']} | {item['lane']} | {item['eligibility_decision'] or 'n/a'} | {item['hypothesis']} |")
    lines += ["","## Fixed-anchor release runs","","| Release | Milestone | Status | Decision | Reason |","|---:|---:|---|---|---|"]
    for release in report["release_runs"]: lines.append(f"| {release['release_id']} | {release['milestone_id']} | {release['status']} | {release['decision'] or 'n/a'} | {release['reason'] or 'n/a'} |")
    lines += ["","## Warnings",""]+[f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines)+"\n"


def summary(root:str|Path,config:KdaCudaCampaignConfig,attempt_id=None,ledger=None)->dict[str,Any]:
    repo=Path(root).resolve(); path=_ledger_path(repo,config,ledger)
    if not path.exists(): raise FileNotFoundError(path)
    db=_open(path,readonly=True)
    try:
        metadata=dict(db.execute("SELECT key,value FROM metadata").fetchall())
        if metadata.get("protocol_sha")!=protocol_sha(config): raise ValueError("ledger protocol hash differs from frozen CUDA config")
        _verify_pinned_refs(repo,config,metadata)
        ready={"ledger":str(path),"protocol_sha":metadata["protocol_sha"]}
        head=_head(db); lane=_derived_lane(head,config); calibrated=db.execute("SELECT COUNT(*) FROM anchors WHERE kind IN ('python_reference','fla_reference')").fetchone()[0]==2
        where,values=("WHERE id=?",(attempt_id,)) if attempt_id else ("",())
        attempts=[{"attempt_id":r[0],"status":r[1],"lane":r[2],"parent_milestone_id":r[3],"base_sha":r[4],"candidate_sha":r[5],"hypothesis":r[6],"changed_paths":json.loads(r[7]),"patch_sha256":r[8],"migration_decision":r[9],"performance_decision":r[10],"eligibility_decision":r[11],"milestone_id":r[12],"reason":r[13],"measurement":json.loads(r[14]) if r[14] else None} for r in db.execute(f"SELECT id,status,lane,parent_milestone_id,base_sha,candidate_sha,hypothesis,changed_paths_json,patch_sha,migration_decision,performance_decision,eligibility_decision,milestone_id,reason,summary_json FROM attempts {where} ORDER BY id DESC LIMIT 12",values)]
        instruction={"bootstrap":"Write the simplest correct naive project-owned CUDA implementation for at least one atomic unit. Optimize nothing and use no PTX.","migration":"Choose any hypothesis that migrates at least one new atomic KDA unit to project-owned CUDA; FLA is allowed only for explicitly unclaimed units.","optimization":"Choose any profile-supported performance hypothesis for the complete FLA-free project CUDA backend."}[lane]
        serializable_head={**head,"owner_set":sorted(head["owner_set"])}
        return {"schema_version":2,"objective":"naive_to_optimized_project_owned_cuda_kda","quality_not_evaluated":True,"ledger":ready["ledger"],"protocol_sha":ready["protocol_sha"],"anchors_calibrated":calibrated,"current_milestone":serializable_head,"next_lane":lane,"attempts":attempts,"next_model_instruction":instruction}
    finally: db.close()
