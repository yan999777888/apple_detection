"""CPU mechanism tests for strong training, without checkpoint/model construction."""
from copy import deepcopy
from pathlib import Path
import json
import torch
import pytest
from apple_strong import training as t


BASELINE_FILES = (
    "base/original_predictions.json", "base/hflip_predictions.json",
    "small/original_predictions.json", "small/hflip_predictions.json",
    "base/prediction_metadata.json", "small/prediction_metadata.json",
    "fused_predictions.json", "base/mask_metrics_at_recall90.json",
    "base/mask_pair_diagnostics.json", "fusion_summary.json",
)


def baseline_fixture(repository):
    runtime=repository/"strong_runtime";runtime.mkdir()
    source=repository/"scripts"/"reproduce_strong_baseline.py"
    source.parent.mkdir();source.write_text("baseline=1")
    baseline=repository/"runs"/"baseline";baseline.mkdir(parents=True)
    artifacts={}
    for name in BASELINE_FILES:
        artifact=baseline/name;artifact.parent.mkdir(parents=True,exist_ok=True)
        artifact.write_text(json.dumps({"fixture_artifact":name}),encoding="utf-8")
        artifacts[name]=t.sha(artifact)
    value=dict(schema="strong_full_four_view_gate_v1",passed=True,images=814,GT=3449,empty=13,
               base_checkpoint_sha256=t.BASE_SHA,small_checkpoint_sha256=t.SMALL_SHA,
               source_sha256={str(source):t.sha(source)},baseline_directory=str(baseline),
               artifact_sha256=artifacts,
               fusion=dict(images=814,gt_count=3449,empty_gt_images=13,tp=3105,fp=183,fn=344))
    gate=repository/"gate.json";gate.write_text(json.dumps(value),encoding="utf-8")
    return gate,runtime,value,source,baseline


def test_successful_schedule_and_cursor_replay():
    assert t.lr_factor(0,1152,50)==.02
    assert t.lr_factor(49,1152,50)==1
    assert t.lr_factor(1151,1152,50)==pytest.approx(.05)
    first=t.Cursor(13,8)
    [first.next() for _ in range(17)]
    saved=deepcopy(first.state_dict())
    restored=t.Cursor(13,99);restored.load_state_dict(saved)
    assert [first.next() for _ in range(50)]==[restored.next() for _ in range(50)]


def test_initial_or_pre_gate_resume_cannot_skip_288_boundary():
    recipe={"gate_updates":288}
    assert t.update_boundary(recipe,1152,None)==288
    assert t.update_boundary(recipe,1152,None,1152)==288
    assert t.update_boundary(recipe,1152,{"approved":True})==1152
    with pytest.raises(ValueError):t.update_boundary(recipe,1152,None,0)


def test_continuation_survives_new_checkpoints_but_not_changed_authorization(tmp_path):
    p=tmp_path/"continue.json"
    p.write_text(json.dumps(dict(passed=True,arm="C1",candidate_sha256="at288")))
    approved=t.authorize_continuation("C1","at288",None,p)
    # Later last.pt candidates have different hashes; authorization is bound
    # to the accepted 288-step candidate and persists across exact resumes.
    assert t.authorize_continuation("C1","at500",approved,None)==approved
    with pytest.raises(ValueError):t.authorize_continuation("C0","at288",None,p)
    p.write_text(json.dumps(dict(passed=True,arm="C1",candidate_sha256="changed")))
    with pytest.raises(ValueError):t.authorize_continuation("C1","at500",approved,None)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Dropout(.5))
        self.head=torch.nn.Linear(3,1)
    def forward(self,x):return self.head(self.backbone(x))


def test_backbone_freeze_eval_survives_train_and_no_gradient():
    model=Tiny();t.freeze_backbone(model);t.enter_train(model)
    assert not model.backbone.training and model.head.training
    before={n:p.detach().clone() for n,p in model.backbone.named_parameters()}
    model(torch.ones(2,3)).sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(torch.equal(before[n],p) for n,p in model.backbone.named_parameters())
    assert all(p.grad is not None for p in model.head.parameters())


def test_ema_complete_shadow_resume_and_frozen_integrity():
    model=Tiny();t.freeze_backbone(model);ema=t.CpuEma(model,.999)
    with torch.no_grad():model.head.weight.add_(.25)
    ema.update(model)
    saved=deepcopy(ema.state_dict())
    restored=t.CpuEma(model,.999);restored.load_state_dict(saved)
    assert t.equal_state(ema.state_dict(),restored.state_dict())
    ema.verify_frozen(model)
    with torch.no_grad():model.backbone[0].weight.add_(1.)
    with pytest.raises(RuntimeError,match="Frozen"):ema.verify_frozen(model)


def test_atomic_own_checkpoint_preserves_optimizer_rng_ema(tmp_path):
    model=Tiny();t.freeze_backbone(model)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=.001,foreach=False)
    model(torch.ones(2,3)).sum().backward();optimizer.step()
    ema=t.CpuEma(model,.999)
    cursor=t.Cursor(9,77);cursor.next()
    state=dict(model=t.to_cpu(model.state_dict()),optimizer=t.to_cpu(optimizer.state_dict()),
               ema=ema.state_dict(),sampler=cursor.state_dict(),torch_rng=torch.get_rng_state())
    path=tmp_path/"last.pt";digest=t.save_torch_atomic(path,state)
    assert path.with_suffix(".pt.sha256").read_text()==digest==t.sha(path)
    loaded=torch.load(path,map_location="cpu",weights_only=False)
    assert t.equal_state(state,loaded)


def test_original_ignore_identity_and_inventory_required(tmp_path):
    p=tmp_path/"ignore.json"
    p.write_text(json.dumps(dict(schema="apple_train_ignore_candidate_boxes_v1",records={})),encoding="utf-8")
    with pytest.raises(ValueError,match="hash"):t.load_ignore(p,"0"*64,[])
    with pytest.raises(ValueError,match="inventory"):t.load_ignore(p,t.sha(p),[])


def test_baseline_requires_exact_models_scope_sources_and_counts(tmp_path):
    p,runtime,value,src,_baseline=baseline_fixture(tmp_path)
    assert t.verify_baseline(p,runtime)["passed"]
    value["fusion"]["fp"]=184;p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="count"):t.verify_baseline(p,runtime)
    value["fusion"]["fp"]=183;p.write_text(json.dumps(value));src.write_text("x=2")
    with pytest.raises(ValueError,match="source"):t.verify_baseline(p,runtime)


def test_real_gate_layout_uses_repository_relative_sources(tmp_path):
    gate,runtime,value,source,_baseline=baseline_fixture(tmp_path)
    value["source_sha256"]={"scripts/reproduce_strong_baseline.py":t.sha(source)}
    gate.write_text(json.dumps(value))
    assert t.verify_baseline(gate,runtime)["passed"]
    value["source_sha256"]={"../escaped.py":"0"*64};gate.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="escaped repository"):t.verify_baseline(gate,runtime)


@pytest.mark.parametrize("artifact_name",BASELINE_FILES)
def test_passed_baseline_rejects_changed_raw_metadata_fusion_or_mask_artifact(tmp_path,artifact_name):
    gate,runtime,_value,_source,baseline=baseline_fixture(tmp_path)
    (baseline/artifact_name).write_text("changed after baseline passed",encoding="utf-8")
    with pytest.raises(ValueError,match="artifact SHA256 differs"):
        t.verify_baseline(gate,runtime)


def test_passed_baseline_rejects_missing_file_and_incomplete_inventory(tmp_path):
    gate,runtime,value,_source,baseline=baseline_fixture(tmp_path)
    artifact=baseline/"base/hflip_predictions.json"
    artifact.unlink()
    with pytest.raises(ValueError,match="artifact is missing"):
        t.verify_baseline(gate,runtime)
    value["artifact_sha256"].pop("base/hflip_predictions.json")
    gate.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="all ten"):
        t.verify_baseline(gate,runtime)


@pytest.mark.parametrize("replacement",("../escaped.json",r"C:\\outside.json",r"\\rooted.json"))
def test_artifact_paths_cannot_escape_baseline_directory(tmp_path,replacement):
    gate,runtime,value,_source,_baseline=baseline_fixture(tmp_path)
    digest=value["artifact_sha256"].pop("small/original_predictions.json")
    value["artifact_sha256"][replacement]=digest
    gate.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="relative and cannot escape"):
        t.verify_baseline(gate,runtime)


def test_duplicate_normalized_artifact_paths_are_rejected(tmp_path):
    gate,runtime,value,_source,_baseline=baseline_fixture(tmp_path)
    digest=value["artifact_sha256"].pop("small/original_predictions.json")
    value["artifact_sha256"][r"BASE\original_predictions.json"]=digest
    gate.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="Duplicate normalized"):
        t.verify_baseline(gate,runtime)


def test_duplicate_literal_gate_keys_are_rejected_before_hashing(tmp_path):
    gate,runtime,value,_source,_baseline=baseline_fixture(tmp_path)
    encoded=json.dumps(value).replace('"artifact_sha256": {','"artifact_sha256": {"base/original_predictions.json":"'+'0'*64+'",',1)
    gate.write_text(encoded)
    with pytest.raises(ValueError,match="Duplicate key"):
        t.verify_baseline(gate,runtime)


@pytest.mark.parametrize("directory",("runs/baseline",r"C:\\baseline","missing","outside"))
def test_baseline_directory_must_be_actual_F_drive_directory_in_same_repository(tmp_path,directory):
    gate,runtime,value,_source,_baseline=baseline_fixture(tmp_path)
    if directory=="outside":
        p=tmp_path.parent/(tmp_path.name+"_outside");p.mkdir();directory=str(p)
    elif directory=="missing":directory=str(tmp_path/"runs"/"absent")
    value["baseline_directory"]=directory;gate.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="Baseline directory"):
        t.verify_baseline(gate,runtime)


def test_capture_restores_instance_methods_and_selects_true_gt_roi_support():
    # Small mechanics fixture exercises the capture path without replacing real
    # training data/model smoke. Box/prediction heads remain differentiable.
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            class Predictor(torch.nn.Linear):
                def forward(self,value):
                    result=super().forward(value)
                    return result[:,:2],result
            self.box_predictor=Predictor(4,8)
        def select_training_samples(self,p,targets):
            return p,[torch.tensor([0,0,1])],[torch.tensor([1,1,1])],[torch.zeros(3,4)]
        def _roi_predictions(self,*args):
            x=self.box_predictor.weight.sum()*torch.ones(3,1,2,2)
            return dict(visible_logits=x,amodal_logits=x),None
        def _decoded_boxes_all(self,regression,proposals,hw):return proposals+regression[:,0:4]*0
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.roi_heads=Head()
    model=Model();head=model.roi_heads
    boxes=torch.tensor([[0.,0.,2.,2.],[0.,0.,1.,1.],[2.,2.,4.,4.]])
    targets=[dict(instance_ids=["a","b"],boxes=torch.stack((boxes[0],boxes[2])))]
    with t.capture_gt_predictions(model) as cap:
        head.select_training_samples([boxes],targets)
        head.box_predictor(boxes)
        head._roi_predictions(None,None,None,None)
        result=t.select_gt_capture(model,cap,(4,4))
        assert result["instance_ids"]==["a","b"]
        assert torch.equal(result["roi_boxes"],boxes[[0,2]])
        assert result["visible_logits"].requires_grad
    assert "select_training_samples" not in head.__dict__
    assert "_roi_predictions" not in head.__dict__
