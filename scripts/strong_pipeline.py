"""Serial approved strong-model workflow. Never starts a weak model."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone

REPO=Path(__file__).absolute().parents[1]
REPORT=REPO/"reports/strong_c0c1_20261030"
BASELINE=REPO/"runs/strong_baseline_20261030"
BASECFG=REPO/"configs/strong_baseline_4060.json"
BASESHA="0fe257c5bfe115a77ac7deac830af479e4c2b0ad6ff69e609901b24c7980fa9b"

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for attempt in range(20):
        try:os.replace(tmp,p);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.1)

def now():
    return datetime.now(timezone.utc).isoformat()

def event(name,**data):
    path=REPORT/"events.json"
    values=read(path) if path.exists() else []
    values.append(dict(sequence=len(values)+1,node=name,utc=now(),**data))
    write(path,values)
    print(json.dumps(values[-1],ensure_ascii=False),flush=True)

def status(stage,**data):
    write(REPORT/"pipeline_status.json",dict(stage=stage,pipeline_pid=os.getpid(),updated_utc=now(),**data))

def stop_requested():
    return (REPORT/"STOP").exists()

def invoke(node,args,*,training_output=None):
    if stop_requested():raise InterruptedError("User requested pipeline stop")
    log=REPORT/(node+".log")
    status(node,log=str(log))
    with log.open("x",encoding="utf-8") as stream:
        process=subprocess.Popen([sys.executable,"-B",*map(str,args)],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        while process.poll() is None:
            if stop_requested() and training_output:
                Path(training_output,"STOP").write_text(now(),encoding="utf-8")
            status(node,child_pid=process.pid,log=str(log),stop_requested=stop_requested())
            time.sleep(5)
    if process.returncode:
        raise RuntimeError(f"{node} failed with exit {process.returncode}; see {log}")
    if stop_requested():raise InterruptedError("User requested pipeline stop")
    return process.returncode

def attach_small_cache(output):
    dst=Path(output)/"small";dst.mkdir(parents=True,exist_ok=False)
    for name in ("original_predictions.json","hflip_predictions.json","prediction_metadata.json"):
        source=BASELINE/"small"/name
        try:os.link(source,dst/name)
        except OSError:shutil.copy2(source,dst/name)

def freeze_candidate(run,step):
    run=Path(run)
    # Preserve each evaluated EMA artifact even when the live run later
    # atomically replaces candidate.pt at a subsequent update boundary.
    snapshot=run/"snapshots"/f"EMA_update_{step:06d}.pt"
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if snapshot.exists():raise FileExistsError("An evaluated candidate snapshot is immutable")
    try:os.link(run/"candidate.pt",snapshot)
    except OSError:shutil.copy2(run/"candidate.pt",snapshot)
    from apple_strong.training import sha
    expected=(run/"candidate.pt.sha256").read_text(encoding="ascii").strip()
    assert sha(snapshot)==expected
    snapshot.with_suffix(".pt.sha256").write_text(expected,encoding="ascii")
    return snapshot

def eval_candidate(arm,step):
    run=REPO/f"runs/strong_{arm}_20261030"
    output=REPO/f"runs/strong_{arm}_eval_{step:06d}_20261030"
    if output.exists():raise FileExistsError("Use a fresh candidate evaluation directory")
    snapshot=freeze_candidate(run,step)
    invoke(f"{arm}_{step}_base_validation",[REPO/"scripts/reproduce_strong_baseline.py","--config",BASECFG,
        "--model","base","--checkpoint",snapshot,"--output",output])
    attach_small_cache(output)
    invoke(f"{arm}_{step}_four_view_fusion",[REPO/"scripts/reproduce_strong_baseline.py","--config",BASECFG,
        "--model","fuse","--output",output])
    result=read(output/"fusion_summary.json")
    event(f"{arm}_{step}_full_evaluation_complete",arm=arm,step=step,at_recall90=result["at_recall90"],
          directory=str(output))
    return output

def check_pressure(arm):
    from apple_strong.training import BASE_SHA,IGNORE_SHA,sha
    directory=REPO/f"runs/strong_{arm}_pressure_20261030"
    result=read(directory/"training_summary.json")
    assert result["phase"]=="smoke_complete" and result["smoke"]
    assert result["arm"]==arm and result["identity"]["arm"]==arm
    assert result["updates"]==dict(attempted=2,successful=2,skipped=0)
    assert result["identity"]["parent_sha256"]==BASE_SHA and result["identity"]["ignore_sha256"]==IGNORE_SHA
    assert result["backbone_FPN_frozen"] and not result["weak_model_used"] and result["source_before_after_identical"]
    for key in ("own_roundtrip_verified","own_state_reapplied_verified","EMA_candidate_roundtrip_verified"):
        assert result["checkpoint"][key],key
    assert len(result["history"])==2
    branches={"rpn","classifier","box_regression","visible","hidden","amodal"}
    signals=0
    for ordinal,row in enumerate(result["history"],start=1):
        assert row["updates"]==dict(attempted=ordinal,successful=ordinal,skipped=0)
        assert row["finite_gradients"] and row["frozen_backbone_gradients_absent"]
        assert set(row["parameter_changes"])==branches and set(row["branch_gradient_abs_sum"])==branches
        assert all(v>0 for v in row["parameter_changes"].values())
        assert all(v>0 for v in row["branch_gradient_abs_sum"].values())
        assert row["cuda_peak_reserved_bytes"]>=row["cuda_peak_allocated_bytes"]>0
        assert len(row["micro_batches"])==4
        for micro in row["micro_batches"]:
            assert micro["source"]["split"]=="train" and micro["source"]["dataset"]=="PApple"
            assert len(micro["views"])==2
            assert [v["view"] for v in micro["views"]]==["original","hflip"]
            assert all(v["GT_unchanged"] for v in micro["views"])
            flipped=micro["views"][1]
            edges=flipped.get("consistency_output_gradient_abs_sum",{})
            if flipped["common_consistency_GT_ids"] and edges and all(v>0 for v in edges.values()):
                signals+=1
    if arm=="C1":assert signals>0,"C1 consistency must have a real nonzero gradient on matched nonempty GT"
    for name,digest in result["identity"]["source_sha256"].items():assert sha(REPO/name)==digest
    return dict(arm=arm,passed=True,updates=result["updates"],C1_nonzero_matched_microbatches=signals,
                allocated_bytes=max(r["cuda_peak_allocated_bytes"] for r in result["history"]),
                reserved_bytes=max(r["cuda_peak_reserved_bytes"] for r in result["history"]),
                summary_sha256=sha(directory/"training_summary.json"))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adopt-base-pid",type=int)
    args=ap.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True)
    import psutil
    lock=REPORT/"pipeline.lock"
    if lock.exists():
        old=read(lock)
        try:
            p=psutil.Process(old["pid"])
            if p.is_running() and abs(p.create_time()-old["created"])<.01:
                raise RuntimeError("A live pipeline already owns the experiment")
        except psutil.NoSuchProcess:pass
    write(lock,dict(pid=os.getpid(),created=psutil.Process().create_time()))
    sys.path.insert(0,str(REPO/"src"))
    try:
        status("baseline_base_reproduction")
        meta_path=BASELINE/"base/prediction_metadata.json"
        if args.adopt_base_pid:
            try:p=psutil.Process(args.adopt_base_pid);created=p.create_time()
            except psutil.NoSuchProcess:p=None
            while True:
                meta=read(meta_path) if meta_path.exists() else {}
                if meta.get("phase")=="complete":break
                if meta.get("phase")=="failed":raise RuntimeError("Adopted Base reproduction failed: "+meta.get("error",""))
                try:alive=p is not None and p.is_running() and abs(p.create_time()-created)<.01
                except psutil.NoSuchProcess:alive=False
                if not alive:raise RuntimeError("Base process ended without complete full evaluation")
                progress=read(BASELINE/"base/progress.json") if (BASELINE/"base/progress.json").exists() else {}
                status("baseline_base_reproduction",child_pid=p.pid,progress=progress.get("images",0),total=814)
                if stop_requested():raise InterruptedError("User requested pipeline stop; adopted evaluation preserved")
                time.sleep(10)
            while p is not None and p.is_running():
                time.sleep(1)
        elif not meta_path.exists():
            invoke("baseline_base_reproduction",[REPO/"scripts/reproduce_strong_baseline.py","--config",BASECFG,"--model","base"])
        assert read(meta_path)["phase"]=="complete"
        event("baseline_Base_full_reproduction_complete",metrics=read(meta_path)["view_metrics"])
        small_meta=BASELINE/"small/prediction_metadata.json"
        if not small_meta.exists():
            invoke("baseline_small_reproduction",[REPO/"scripts/reproduce_strong_baseline.py","--config",BASECFG,"--model","small"])
        assert read(small_meta)["phase"]=="complete"
        event("baseline_Small_full_reproduction_complete",metrics=read(small_meta)["view_metrics"])
        invoke("baseline_four_view_fusion",[REPO/"scripts/reproduce_strong_baseline.py","--config",BASECFG,"--model","fuse"])
        gate=read(REPORT/"baseline_gate.json")
        assert gate["passed"] and gate["base_checkpoint_sha256"]==BASESHA
        from apple_strong.training import sha
        artifacts=("base/original_predictions.json","base/hflip_predictions.json",
                   "small/original_predictions.json","small/hflip_predictions.json",
                   "base/prediction_metadata.json","small/prediction_metadata.json",
                   "fused_predictions.json","base/mask_metrics_at_recall90.json",
                   "base/mask_pair_diagnostics.json","fusion_summary.json")
        gate["artifact_sha256"]={name:sha(BASELINE/name) for name in artifacts}
        write(REPORT/"baseline_gate.json",gate)
        event("baseline_94_43_reproduced",measurements=gate["fusion"])
        pressures=[]
        for arm in ("C0","C1"):
            cfg=REPO/f"configs/strong_{arm.lower()}_pressure_4060.json"
            invoke(f"{arm}_real_GPU_pressure",[REPO/"scripts/train_strong.py","--config",cfg,"--smoke","--max-smoke-updates","2"],
                   training_output=REPO/f"runs/strong_{arm}_pressure_20261030")
            pressures.append(check_pressure(arm))
            event(f"{arm}_real_GPU_pressure_passed",measurements=pressures[-1])
        write(REPORT/"pressure_gate.json",dict(passed=True,results=pressures))
        short_dirs={}
        for arm in ("C0","C1"):
            cfg=REPO/f"configs/strong_{arm.lower()}_4060.json"
            invoke(f"{arm}_train_to_288",[REPO/"scripts/train_strong.py","--config",cfg,"--stop-after-updates","288"],
                   training_output=REPO/f"runs/strong_{arm}_20261030")
            result=read(REPO/f"runs/strong_{arm}_20261030/training_summary.json")
            assert result["phase"]=="awaiting_evaluation" and result["updates"]["successful"]==288
            event(f"{arm}_288_training_complete",updates=result["updates"])
            short_dirs[arm]=eval_candidate(arm,288)
        from apple_strong.acceptance import plan_experiment_gates
        provenance=REPO.parent/"Apple_DualBranch_3080_20261022/assets/papple_val_provenance.json"
        decisions=plan_experiment_gates(BASELINE,short_dirs["C0"],short_dirs["C1"],provenance)
        write(REPORT/"short_comparison.json",decisions)
        for arm in ("C0","C1"):write(REPORT/f"{arm}_288_continuation_gate.json",decisions["arms"][arm])
        event("short_C0_C1_comparison_complete",decisions=decisions)
        if not decisions["experiment_continue"]:
            status("complete_no_promotion",reason="Neither short arm passed the predeclared accuracy/quality gate")
            event("experiment_stopped_no_promotion",original_model_preserved=True)
            return 0
        long_dirs={}
        for arm in ("C0","C1"):
            invoke(f"{arm}_resume_to_1152",[REPO/"scripts/train_strong.py","--config",REPO/f"configs/strong_{arm.lower()}_4060.json",
                    "--resume","--continue-after-gate",REPORT/f"{arm}_288_continuation_gate.json"],
                   training_output=REPO/f"runs/strong_{arm}_20261030")
            result=read(REPO/f"runs/strong_{arm}_20261030/training_summary.json")
            assert result["phase"]=="complete" and result["updates"]["successful"]==1152
            event(f"{arm}_1152_training_complete",updates=result["updates"])
            long_dirs[arm]=eval_candidate(arm,1152)
        final=plan_experiment_gates(BASELINE,long_dirs["C0"],long_dirs["C1"],provenance)
        write(REPORT/"final_comparison.json",final)
        event("full_C0_C1_comparison_complete",decisions=final,original_model_preserved=True)
        status("complete_base_experiments",final_report=str(REPORT/"final_comparison.json"),
               next_step="Report measured results; any proposed promotion needs repetition before modifying Small")
        return 0
    except InterruptedError as error:
        status("paused_by_user",reason=str(error));event("pipeline_paused",reason=str(error));return 130
    except BaseException as error:
        status("failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        event("pipeline_failed",error_type=type(error).__name__,error=str(error))
        traceback.print_exc();return 1
    finally:
        if lock.exists():lock.unlink()

if __name__=="__main__":
    raise SystemExit(main())
