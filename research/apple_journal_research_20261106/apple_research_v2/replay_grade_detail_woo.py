"""Recover native-320 checkpoints and select them against predeclared Woo development criteria."""
from pathlib import Path
import shutil
import torch

import afruit_grade_model as grade
import afruit_grade_train as io
import select_grade_repair as selection
import train_grade_detail_patch as original


OUT=original.OUT.with_name("border_native320_woo_ref")
TRAJECTORY=OUT/"trajectory"
STEPS=tuple(range(40,241,40))


def good_changes(candidate,baseline,panels):
    changed={}
    for name,(records,*_) in panels.items():
        records_changed=[]
        for row,old,new in zip(records,baseline["predictions"][name],candidate["predictions"][name],strict=True):
            if row["grade"]!=0 or (old["predicted"]==0)==(new["predicted"]==0):
                continue
            records_changed.append(dict(path=row["path"],group=row["group_id"],sha256=row["sha256"],
                         previous=old["predicted"],current=new["predicted"],
                         change="lost" if old["predicted"]==0 else "recovered"))
        changed[name]=records_changed
    return changed


def run():
    if (OUT/"selection.json").is_file():
        raise FileExistsError("Native-320 Woo-reference selection already exists")
    if selection.policy_digest(selection.POLICY)!=selection.POLICY_SHA:
        raise ValueError("Changed common development policy")
    published=io.read(original.OUT/"history.json")
    reference_protocol=io.read(original.OUT/"protocol.json")
    if tuple(r["step"] for r in published)!=STEPS:
        raise ValueError("Unknown original trajectory")
    frozen=io.read(original.OUT/"selection.json")
    old_source=original.OUT/"original_training_source.py"
    if grade.file_sha(old_source)!=reference_protocol["script"]:
        raise ValueError("Original training source archive differs from its frozen protocol")
    policy=dict(version=1,starts_from=grade.file_sha(original.PARENT/"weights.pt"),seed=original.SEED,steps=240,
                recorded_updates=list(STEPS),canonical_policy=selection.POLICY_SHA,
                source_original_sha256=grade.file_sha(old_source),source_replay_sha256=grade.file_sha(original.__file__),
                guard_manifest_sha256=grade.file_sha(selection.guard.INPUT),
                source_history_sha256=grade.file_sha(original.OUT/"history.json"),
                read_only_development="Same 57 AF full/auto photos and same guard v2 val only. No historical panels.",
                fidelity="Verify all six losses and full cached development confusion counts; verify every tensor of the sole prior saved checkpoint. The same seed is replayed from border, not resumed/past the previous endpoint.",
                select="Apply common Woo panel safety and mild gates and predeclared rank to the fresh development logits of all 6 checkpoints. Chronologically first on complete metric tie; document clinically relevant row changes against both parents.")
    lock=OUT/"replay_protocol.json"
    if lock.exists() and io.read(lock)!=policy:raise ValueError("Replay policy changed")
    io.atomic(lock,policy)
    def save_step(step,model,record):
        if record!=published[STEPS.index(step)]:
            raise ValueError(f"Replayed loss/predictions differ at step {step}")
        folder=OUT/f"step{step:03d}"
        io.atomic(folder/"weights.pt",model.state_dict(),tensor=True)
        path=folder/"weights.pt"
        io.atomic(folder/"selection.json",dict(architecture=original.ARCH,threshold=1.,input_side=320,
            step=step,weights_sha256=grade.file_sha(path),protocol_sha256=grade.file_sha(lock)))
    if not (TRAJECTORY/"selection.json").exists():
        original.run(TRAJECTORY,save_step)
    if io.read(TRAJECTORY/"history.json")!=published:
        raise ValueError("Recovery trace does not reproduce original six checkpoints")
    old_weights=torch.load(original.OUT/"weights.pt",map_location="cpu",weights_only=True)
    recovered=torch.load(OUT/f"step{frozen['step']:03d}"/"weights.pt",map_location="cpu",weights_only=True)
    if old_weights.keys()!=recovered.keys() or any(not torch.equal(t,recovered[k]) for k,t in old_weights.items()):
        raise ValueError("The originally saved checkpoint was not reproduced exactly")
    panels=selection.input_panels(detail=True)
    woo=selection.evaluate(selection.REFERENCE,panels,"cuda")
    if woo["weight_sha256"]!=io.read(selection.POLICY)["reference_sha256"]:
        raise ValueError("Wrong Woo reference")
    border=selection.evaluate(original.PARENT,panels,"cuda")
    arms=[];audit=[]
    for step in STEPS:
        folder=OUT/f"step{step:03d}"
        arm=selection.evaluate(folder,panels,"cuda")
        selection.compare(arm,woo)
        arms.append(arm)
        audit.append(dict(step=step,directory=arm["directory"],sha256=arm["weight_sha256"],passed=arm["passed"],
             metrics=arm["metrics"],mild_shortfalls=arm["mild_shortfalls"],safety_shortfalls=arm["safety_shortfalls"],
             performance_rank=arm["performance_rank"],good_changes_vs_woo=good_changes(arm,woo,panels),
             good_changes_vs_border=good_changes(arm,border,panels)))
        print("Woo comparison",step,arm["passed"],arm["metrics"]["auto"]["confusion"],flush=True)
    chosen=min(arms,key=selection.candidate_order)
    folder=grade.ROOT/chosen["directory"];selected_step=int(folder.name.removeprefix("step"))
    shutil.copyfile(folder/"weights.pt",OUT/"weights.pt")
    if grade.file_sha(OUT/"weights.pt")!=chosen["weight_sha256"]:
        raise ValueError("Selected bytes changed")
    out=dict(prior_step=frozen["step"],matched_parameter_tensors=len(recovered),exact_training_checkpoints=list(STEPS),
             woo=woo,border=border,arms=arms,audit=audit,selected_step=selected_step,
             development_inputs={name:items[0] for name,items in panels.items()},
             frozen_step_weight_byte_match=grade.file_sha(original.OUT/"weights.pt")==grade.file_sha(OUT/f"step{frozen['step']:03d}"/"weights.pt"))
    io.atomic(OUT/"development_audit.json",out)
    io.atomic(OUT/"selection.json",dict(architecture=original.ARCH,threshold=1.,input_side=320,step=selected_step,
        passed=chosen["passed"],weights_sha256=chosen["weight_sha256"],parent_sha256=policy["starts_from"],
        protocol_sha256=grade.file_sha(lock),metrics=chosen["metrics"],source_history_sha256=policy["source_history_sha256"]))
    print("Selected",selected_step,chosen["weight_sha256"],flush=True)


if __name__=="__main__":
    run()
