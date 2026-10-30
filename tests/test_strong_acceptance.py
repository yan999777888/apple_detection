"""Full-scope CPU JSON fixtures exercise the real acceptance gate, not model accuracy."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import pytest

from apple_strong import acceptance as gate

REAL_REPO = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_fixture_strong_matching",
    REAL_REPO / "reference/local_amodal/scripts/metrics.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def experiment(monkeypatch):
    reports = REAL_REPO / "reports"
    reports.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="strong_acceptance_",dir=reports) as temp:
        root = Path(temp)
        runtime = root / "strong_runtime"
        for name in ("od_model.py","od_inference.py","od_evaluate.py","od_data.py","data.py"):
            path = runtime / "src/apple_seg" / name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("fixture runtime "+name,encoding="utf-8")
        for rel in ("scripts/reproduce_strong_baseline.py","strong_runtime/tmp/flip_tta.py"):
            path = root / rel;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text("fixture producer "+rel,encoding="utf-8")
        mp = runtime / "reference/local_amodal/scripts/metrics.py"
        mp.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REAL_REPO / "reference/local_amodal/scripts/metrics.py",mp)
        monkeypatch.setattr(gate,"REPO",root)
        monkeypatch.setattr(gate,"RUNTIME",runtime)
        checkpoints = {}
        for name in ("base","small","C0","C1"):
            p = root / "weights" / (name+".pt")
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(("bounded identity fixture "+name).encode())
            p.with_suffix(".pt.sha256").write_text(digest(p),encoding="ascii")
            checkpoints[name] = p
        monkeypatch.setattr(gate,"BASE_SHA",digest(checkpoints["base"]))
        monkeypatch.setattr(gate,"SMALL_SHA",digest(checkpoints["small"]))
        images,truth,groups,ordinal = [],[],list(gate.STRATA.items()),0
        group_by_index = [name for name,count in groups for _ in range(count)]
        for index in range(814):
            count = 0 if index >= 801 else (5 if index < 245 else 4)
            name = f"frame_{index:04d}.png"
            rows=[]
            for g in range(count):
                rows.append({"image_id":name,"gt_index":g,"apple_id":f"apple_{ordinal}",
                             "box":[30*g,0,30*g+10,10],"occlusion_stratum":group_by_index[ordinal]})
                ordinal += 1
            truth.extend(rows)
            images.append({"image_id":name,"image_index":index,"gt_count":count,
                           "apple_ids_in_target_order":[r["apple_id"] for r in rows]})
        assert ordinal == 3449
        provenance = {"phase":"complete","passed":True,
            "scope":{"split":"val","images":814,"GT_instances":3449,"empty_GT_images":13},
            "image_metadata":images,"ground_truth_instances":truth}
        provenance_path = root / "provenance.json";write(provenance_path,provenance)
        monkeypatch.setattr(gate,"PROVENANCE_SHA",digest(provenance_path))
        truth_by_image = {r["image_id"]:[] for r in images}
        global_rows=[]
        for row in truth:
            truth_by_image[row["image_id"]].append(row)
            global_rows.append(row)
        inventory = gate._inventory()
        paths={name:root/name for name in ("baseline","C0","C1")}

        def records(tp=3105,fp=183,selected=None):
            selected=set(range(tp)) if selected is None else set(selected)
            result=[{"image_id":r["image_id"],"gt":{"boxes":[g["box"] for g in truth_by_image[r["image_id"]]]},
                     "pred":{"boxes":[],"scores":[]}} for r in images]
            by_name={r["image_id"]:r for r in result}
            for i,g in enumerate(global_rows):
                if i in selected:
                    by_name[g["image_id"]]["pred"]["boxes"].append(g["box"])
                    by_name[g["image_id"]]["pred"]["scores"].append(.8)
            for i in range(fp):
                pred=result[i%814]["pred"]
                pred["boxes"].append([500,500,510,510]);pred["scores"].append(.8)
            return result

        baseline_records=records()
        def best(values):
            eligible=[r for r in metrics.calibrate_threshold(values,.5,True)["curve"] if r["recall"]>=.90]
            return max(eligible,key=lambda r:(r["precision"],r["recall"],r["f1"],r["confidence"])) if eligible else None

        def materialize(name,tp=3105,fp=183,selected=None,mask_drop=0):
            directory=paths[name];candidate=name!="baseline"
            layer=records(tp,fp,selected)
            for model in ("base","small"):
                values=layer if model=="base" else baseline_records
                for view in ("original","hflip"):
                    write(directory/model/(view+"_predictions.json"),values)
                checkpoint=checkpoints[name] if model=="base" and candidate else checkpoints[model]
                point=best(values)
                assert point is not None, "Fixture Base views retain a valid R90 point"
                meta={"phase":"complete","model":model,"images":814,"GT":3449,"empty":13,
                    "candidate_mode":candidate and model=="base","source_sha256":inventory,
                    "from_fresh_weight_inference":True,"source_before_after_identical":True,
                    "FP32":True,"short_side":1024,"max_size":1280,"provenance_sha256":gate.PROVENANCE_SHA,
                    "strict_restore":{"strict":True,"all_tensors_equal_to_checkpoint":True,"missing_keys":[],
                                      "unexpected_keys":[],"tensor_count":1},
                    "checkpoint":str(checkpoint),"checkpoint_sha256":digest(checkpoint),
                    "view_metrics":{v:{"at_recall90":point} for v in ("original","hflip")}}
                write(directory/model/"prediction_metadata.json",meta)
            selected_point=best(layer)
            write(directory/"fused_predictions.json",layer)
            write(directory/"fusion_summary.json",{"phase":"complete","images":814,"GT":3449,"empty":13,
                "at_recall90":selected_point,"at_confidence_0_75":metrics.evaluate_at_threshold(layer,.75,.5),
                "candidate_mode":candidate,"from_fresh_weight_inference":True,"source_sha256":inventory,
                "fusion":gate.FUSION})
            diagnostics=[];drop_remaining=mask_drop;mask_tp=0;prediction_count=0
            for row in layer:
                gt=truth_by_image[row["image_id"]]
                m=metrics.match_image(row["gt"]["boxes"],row["pred"]["boxes"],row["pred"]["scores"],.5,.8)
                assign=[-1]*len(row["pred"]["boxes"])
                for p,g in zip(m["pred_indices"],m["gt_indices"],strict=True):
                    if g>=0:
                        if drop_remaining:drop_remaining-=1
                        else:assign[int(p)]=int(g);mask_tp+=1
                prediction_count+=len(assign)
                diagnostics.append({"image_id":row["image_id"],"gt_count":len(gt),
                    "apple_ids_in_target_order":[g["apple_id"] for g in gt],"scores":row["pred"]["scores"],
                    "visible_mask_gt_indices_by_prediction":assign,"amodal_mask_gt_indices_by_prediction":assign})
            write(directory/"base/mask_pair_diagnostics.json",diagnostics)
            mask=metrics._counts(mask_tp,prediction_count-mask_tp,3449,.8,.5)
            write(directory/"base/mask_metrics_at_recall90.json",{"confidence":.8,"metrics":{"visible":mask,"amodal":mask}})
            if not candidate:
                point=best(layer)
                write(directory/"baseline_gate.json",{"schema":"strong_full_four_view_gate_v1","passed":True,
                    "images":814,"GT":3449,"empty":13,"base_checkpoint_sha256":gate.BASE_SHA,
                    "small_checkpoint_sha256":gate.SMALL_SHA,"source_sha256":inventory,
                    "baseline_directory":str(directory),"fusion":metrics.evaluate_at_threshold(layer,.75,.5),
                    "at_recall90":point})

        materialize("baseline")
        materialize("C0")
        materialize("C1",fp=176)
        yield {"root":root,"paths":paths,"provenance":provenance_path,"records":records,
               "materialize":materialize,"inventory":inventory,"checkpoints":checkpoints,
               "best":best}


def plan(e):
    return gate.plan_experiment_gates(e["paths"]["baseline"],e["paths"]["C0"],e["paths"]["C1"],e["provenance"])


def alter(path,change):
    data=json.loads(path.read_text(encoding="utf-8"))
    change(data);write(path,data)


def test_one_eligible_arm_continues_both_but_control_is_not_promoted(experiment):
    result=plan(experiment)
    assert result["experiment_continue"] is True
    assert result["arms"]["C1"]["candidate_eligible"] is True
    assert result["arms"]["C0"]["candidate_eligible"] is False
    assert result["arms"]["C0"]["passed"] is True
    assert result["arms"]["C0"]["continuation_role"]=="matched_control_only_not_promoted"
    assert result["deployment_promoted"] is False
    assert result["original_baseline_remains_protected"] is True
    assert result["arms"]["C1"]["candidate_sha256"]==digest(experiment["checkpoints"]["C1"])


def test_both_below_threshold_stop(experiment):
    experiment["materialize"]("C1")
    result=plan(experiment)
    assert result["experiment_continue"] is False
    assert not any(r["passed"] or r["candidate_eligible"] for r in result["arms"].values())


def test_precision_one_fp_short_of_gain_fails(experiment):
    experiment["materialize"]("C1",fp=177)
    result=plan(experiment)
    assert result["arms"]["C1"]["candidate_eligible"] is False
    assert result["experiment_continue"] is False


def test_one_tp_below_recall_threshold_is_not_extrapolated(experiment):
    path=experiment["paths"]["C1"]
    values=experiment["records"](tp=3104,fp=170)
    write(path/"fused_predictions.json",values)
    alter(path/"fusion_summary.json",lambda d:d.update(at_recall90=None,at_confidence_0_75=metrics.evaluate_at_threshold(values,.75,.5)))
    result=plan(experiment)
    assert not result["experiment_continue"]
    assert result["arms"]["C1"]["assessment"]["at_recall90"] is None


def test_high_precision_declared_but_recall_below90_is_rejected(experiment):
    path=experiment["paths"]["C1"]/"fusion_summary.json"
    alter(path,lambda d:d["at_recall90"].update(tp=3104,recall=3104/3449))
    with pytest.raises(ValueError,match="count differs"):
        plan(experiment)


def test_heavy_recall_more_than_half_point_drop_rejects(experiment):
    selected=set(range(3105));selected.difference_update(range(8));selected.update(range(3105,3113))
    experiment["materialize"]("C1",fp=176,selected=selected)
    result=plan(experiment)
    assert not result["experiment_continue"]
    assert any("heavy" in r for r in result["arms"]["C1"]["reasons"])


def test_base_mask_f1_drop_rejects(experiment):
    experiment["materialize"]("C1",fp=176,mask_drop=25)
    result=plan(experiment)
    assert not result["experiment_continue"]
    assert any("mask F1" in r for r in result["arms"]["C1"]["reasons"])


@pytest.mark.parametrize("target",["fused_predictions.json","base/original_predictions.json","small/hflip_predictions.json"])
def test_gt_box_change_cannot_pass(experiment,target):
    path=experiment["paths"]["C1"]/target
    alter(path,lambda d:d[0]["gt"]["boxes"][0].__setitem__(0,-1))
    with pytest.raises(ValueError,match="GT boxes differ"):
        plan(experiment)


@pytest.mark.parametrize("value",["","frame_0001.png"])
def test_empty_or_duplicate_frame_id_rejected(experiment,value):
    path=experiment["paths"]["C1"]/"fused_predictions.json"
    alter(path,lambda d:d[0].__setitem__("image_id",value))
    with pytest.raises(ValueError,match="identities"):
        plan(experiment)


def test_frozen_small_weight_cannot_change(experiment):
    path=experiment["paths"]["C1"]/"small/prediction_metadata.json"
    alter(path,lambda d:d.update(checkpoint_sha256="a"*64))
    with pytest.raises(ValueError,match="Wrong delivered"):
        plan(experiment)


def test_candidate_actual_checkpoint_sha_is_checked(experiment):
    experiment["checkpoints"]["C1"].write_bytes(b"modified checkpoint bytes")
    with pytest.raises(ValueError,match="Actual checkpoint"):
        plan(experiment)


def test_candidate_metadata_sidecar_disagreement_is_rejected(experiment):
    path=experiment["paths"]["C1"]/"base/prediction_metadata.json"
    alter(path,lambda d:d.update(checkpoint_sha256="a"*64))
    with pytest.raises(ValueError,match="sidecar"):
        plan(experiment)


def test_small_two_view_cache_must_match_baseline(experiment):
    path=experiment["paths"]["C1"]/"small/hflip_predictions.json"
    # Same semantics but altered bytes still cannot replace frozen Small cache.
    value=json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(value,indent=1),encoding="utf-8")
    with pytest.raises(ValueError,match="Frozen Small cache"):
        plan(experiment)


def test_baseline_passed_flag_and_actual_published_counts_are_both_required(experiment):
    path=experiment["paths"]["baseline"]/"baseline_gate.json"
    alter(path,lambda d:d.update(passed=False))
    with pytest.raises(ValueError,match="has not passed"):
        plan(experiment)


def test_baseline_cannot_pass_with_fewer_correct_detections(experiment):
    experiment["materialize"]("baseline",tp=3106,fp=183)
    with pytest.raises(ValueError,match="Published baseline"):
        plan(experiment)


def test_mutated_producer_or_runtime_source_fails(experiment):
    path=experiment["root"]/"scripts/reproduce_strong_baseline.py"
    path.write_text("changed producer",encoding="utf-8")
    with pytest.raises(ValueError,match="source inventory"):
        plan(experiment)


def test_missing_or_partial_artifact_fails_closed(experiment):
    path=experiment["paths"]["C1"]/"base/mask_pair_diagnostics.json"
    value=json.loads(path.read_text(encoding="utf-8"));write(path,value[:-1])
    with pytest.raises(ValueError,match="scope"):
        plan(experiment)


def test_same_box_threshold_mask_metrics_are_independently_replayed(experiment):
    path=experiment["paths"]["C1"]/"base/mask_metrics_at_recall90.json"
    alter(path,lambda d:d["metrics"]["visible"].update(f1=1.0))
    with pytest.raises(ValueError,match="value differs: f1"):
        plan(experiment)


def test_optional_baseline_artifact_sha_inventory_is_verified(experiment):
    directory=experiment["paths"]["baseline"]
    names=("base/original_predictions.json","base/hflip_predictions.json",
           "small/original_predictions.json","small/hflip_predictions.json",
           "base/prediction_metadata.json","small/prediction_metadata.json",
           "fused_predictions.json","base/mask_metrics_at_recall90.json")
    hashes={name:digest(directory/name) for name in names}
    alter(directory/"baseline_gate.json",lambda d:d.update(artifact_sha256=hashes))
    assert plan(experiment)["experiment_continue"]
    hashes["base/original_predictions.json"]="0"*64
    alter(directory/"baseline_gate.json",lambda d:d.update(artifact_sha256=hashes))
    with pytest.raises(ValueError,match="artifact SHA differs"):
        plan(experiment)
