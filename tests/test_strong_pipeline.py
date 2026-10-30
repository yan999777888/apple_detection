"""CPU mechanism checks for immutable evaluated weights and real-pressure gates."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
from apple_strong.training import BASE_SHA, IGNORE_SHA, sha

SPEC=importlib.util.spec_from_file_location("_strong_pipeline_under_test",Path(__file__).parents[1]/"scripts/strong_pipeline.py")
pipeline=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_snapshot_survives_atomic_replacement_and_rejects_overwrite(tmp_path):
    run=tmp_path/"run";run.mkdir()
    live=run/"candidate.pt";live.write_bytes(b"first EMA artifact")
    live.with_suffix(".pt.sha256").write_text(sha(live))
    snapshot=pipeline.freeze_candidate(run,288)
    replacement=run/"replacement";replacement.write_bytes(b"later EMA artifact");replacement.replace(live)
    assert snapshot.read_bytes()==b"first EMA artifact"
    with pytest.raises(FileExistsError):pipeline.freeze_candidate(run,288)
    assert snapshot.read_bytes()==b"first EMA artifact"


@pytest.fixture
def pressure(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,"REPO",tmp_path)
    source=tmp_path/"source.py";source.write_text("frozen source")
    branches={n:1. for n in ("rpn","classifier","box_regression","visible","hidden","amodal")}
    micro={"source":{"split":"train","dataset":"PApple"},"views":[
        {"view":"original","GT_unchanged":True,"common_consistency_GT_ids":["1"]},
        {"view":"hflip","GT_unchanged":True,"common_consistency_GT_ids":["1"],
         "consistency_output_gradient_abs_sum":{"boxes":1.,"visible":1.,"amodal":1.}}]}
    history=[{"updates":{"attempted":i,"successful":i,"skipped":0},"finite_gradients":True,
              "frozen_backbone_gradients_absent":True,"parameter_changes":branches,
              "branch_gradient_abs_sum":branches,"cuda_peak_allocated_bytes":10,"cuda_peak_reserved_bytes":20,
              "micro_batches":[deepcopy(micro) for _ in range(4)]} for i in (1,2)]
    data={"phase":"smoke_complete","smoke":True,"arm":"C1","updates":{"attempted":2,"successful":2,"skipped":0},
          "identity":{"arm":"C1","parent_sha256":BASE_SHA,"ignore_sha256":IGNORE_SHA,"source_sha256":{"source.py":sha(source)}},
          "backbone_FPN_frozen":True,"weak_model_used":False,"source_before_after_identical":True,
          "checkpoint":{"own_roundtrip_verified":True,"own_state_reapplied_verified":True,"EMA_candidate_roundtrip_verified":True},"history":history}
    path=tmp_path/"runs/strong_C1_pressure_20261030/training_summary.json";path.parent.mkdir(parents=True)
    def save():path.write_text(json.dumps(data));return data
    return data,save,source


def test_complete_c1_pressure_passes(pressure):
    _,save,_=pressure;save()
    assert pipeline.check_pressure("C1")["C1_nonzero_matched_microbatches"]==8


@pytest.mark.parametrize("fault",["wrong_arm","skipped","changed_frozen","missing_ema","zero_signal","changed_source"])
def test_invalid_pressure_never_authorizes_training(pressure,fault):
    data,save,source=pressure
    if fault=="wrong_arm":data["arm"]="C0"
    elif fault=="skipped":data["updates"]["skipped"]=1
    elif fault=="changed_frozen":data["history"][0]["frozen_backbone_gradients_absent"]=False
    elif fault=="missing_ema":data["checkpoint"]["EMA_candidate_roundtrip_verified"]=False
    elif fault=="zero_signal":
        for row in data["history"]:
            for micro in row["micro_batches"]:micro["views"][1]["consistency_output_gradient_abs_sum"]={"boxes":0.,"visible":0.,"amodal":0.}
    elif fault=="changed_source":source.write_text("changed")
    save()
    with pytest.raises(AssertionError):pipeline.check_pressure("C1")
