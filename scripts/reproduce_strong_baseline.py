"""Fresh full-VAL Base/Small original+hflip reproduction, one GPU model at a time.

Never trains. Every view is recomputed from checkpoint and original RGB, not
substituted with delivered predictions. Gate requires the published TP/FP/FN.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback

REPO = Path(__file__).absolute().parents[1]
RUNTIME = REPO / "strong_runtime"
sys.path.insert(0, str(RUNTIME / "src"))
sys.path.insert(0, str(RUNTIME / "tmp"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def source_inventory():
    paths = [Path(__file__), *sorted((RUNTIME / "src/apple_seg").glob("*.py")),
             RUNTIME / "tmp/flip_tta.py", RUNTIME / "reference/local_amodal/scripts/metrics.py"]
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("base", "small", "fuse"), required=True)
    parser.add_argument("--checkpoint", type=Path, help="Candidate only; never produces a baseline gate")
    parser.add_argument("--output", type=Path, help="Fresh output directory override for a candidate")
    parser.add_argument("--skip-mask-diagnostics", action="store_true")
    args = parser.parse_args()
    config_path = args.config.absolute()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    root = (config_path.parent / config["path_base"]).absolute()
    paths = {k: (root / v).absolute() for k,v in config["paths"].items()}
    output = args.output.absolute() if args.output else paths["output"]
    started = time.monotonic()
    import numpy as np
    import torch
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    import flip_tta
    from apple_seg.evaluate import load_papple_contract, _audit_target, _audit_rgb, _greedy_assignment, _mask_pr, _occlusion
    from apple_seg.data import PAppleDataset
    from apple_seg.od_data import ODInputDataset
    from apple_seg.od_training import load_checkpoint
    from apple_seg.od_model import build_od_amodal_detector
    from apple_seg.od_inference import _strict_restore
    from apple_seg.od_evaluate import _validate_prediction, _roi_mask_overlaps
    inventory = source_inventory()
    contract = load_papple_contract(paths["provenance"])
    metrics = flip_tta.metrics
    if args.model == "fuse":
        layers = [json.loads((output / model / f"{view}_predictions.json").read_text(encoding="utf-8"))
                  for model,view in (("base","original"),("base","hflip"),("small","original"),("small","hflip"))]
        metas = [json.loads((output / model / "prediction_metadata.json").read_text(encoding="utf-8")) for model in ("base","small")]
        assert all(m["phase"] == "complete" and m["images"] == 814 and m["GT"] == 3449 for m in metas)
        assert all(m["source_sha256"] == inventory for m in metas), "Reproduction source changed between models"
        for records in layers:
            assert [r["image_id"] for r in records] == [r["image_id"] for r in contract["images"]]
            assert all(r["gt"]["boxes"] == [g["box"] for g in contract["truth"][r["image_id"]]] for r in records)
        fused = flip_tta.fuse_records(layers, **config["fusion"])
        calibrated = metrics.calibrate_threshold(fused,.5,return_curve=True)
        eligible = [r for r in calibrated["curve"] if r["recall"] >= .90]
        at_r90 = max(eligible, key=lambda r:(r["precision"],r["recall"],r["f1"],r["confidence"])) if eligible else None
        fixed = metrics.evaluate_at_threshold(fused,.75,.5)
        result = {"phase":"complete","images":814,"GT":3449,"empty":13,
                  "balanced":calibrated["summary"],"at_recall90":at_r90,"at_confidence_0_75":fixed,
                  "fusion":config["fusion"],"from_fresh_weight_inference":True,"source_sha256":inventory,
                  "candidate_mode":any(m["candidate_mode"] for m in metas),
                  "elapsed_seconds":time.monotonic()-started}
        write(output/"fused_predictions.json",fused)
        write(output/"fusion_summary.json",result)
        if not result["candidate_mode"]:
            passed = (fixed["tp"],fixed["fp"],fixed["fn"]) == (3105,183,344)
            assert metas[0]["checkpoint_sha256"] == config["base_checkpoint_sha256"]
            assert metas[1]["checkpoint_sha256"] == config["small_checkpoint_sha256"]
            gate = {"schema":"strong_full_four_view_gate_v1","passed":passed,"images":814,"GT":3449,"empty":13,
                    "base_checkpoint_sha256":metas[0]["checkpoint_sha256"],"small_checkpoint_sha256":metas[1]["checkpoint_sha256"],
                    "fusion":fixed,"at_recall90":at_r90,"source_sha256":inventory,"new_source_lineage":True,
                    "baseline_directory":str(output),"created_utc":datetime.now(timezone.utc).isoformat()}
            write(paths["gate"],gate)
            if not passed:
                raise RuntimeError("Fresh baseline does not reproduce published TP/FP/FN; no training gate granted")
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 0
    stage = output/args.model
    if stage.exists() and any(stage.iterdir()):
        raise FileExistsError("Use a fresh model output directory")
    stage.mkdir(parents=True,exist_ok=True)
    checkpoint = args.checkpoint.absolute() if args.checkpoint else paths[args.model+"_checkpoint"]
    digest = sha(checkpoint)
    if not args.checkpoint:
        assert digest == config[args.model+"_checkpoint_sha256"]
    meta = {"phase":"initializing","model":args.model,"checkpoint":str(checkpoint),"checkpoint_sha256":digest,
            "candidate_mode":args.checkpoint is not None,"source_sha256":inventory,"images":0,"GT":0,
            "provenance_sha256":contract["provenance_sha256"],"FP32":True,"from_fresh_weight_inference":True,
            "historical_source_audit_claimed":False,"new_source_lineage":True}
    write(stage/"prediction_metadata.json",meta)
    try:
        print(f"STRONG {args.model} strict full checkpoint restore",flush=True)
        payload = load_checkpoint(checkpoint)
        model = build_od_amodal_detector(weights=None,**payload["constructor_config"])
        meta["strict_restore"] = _strict_restore(model,payload["model"])
        meta["weights_are_ema"] = payload.get("weights_are_ema",False)
        assert model.constructor_config == payload["constructor_config"]
        del payload
        model.requires_grad_(False).eval().to("cuda")
        native = PAppleDataset(paths["papple_root"],"val",32,augment=False)
        dataset = ODInputDataset(native,short_sides=(config["short_side"],),max_size=config["max_size"])
        assert len(dataset)==814 and native.annotation_sha256==contract["provenance"]["binding"]["current_annotation_sha256"]
        records={"original":[],"hflip":[]}; diagnostics=[]
        torch.cuda.reset_peak_memory_stats()
        meta.update(phase="inference",short_side=config["short_side"],max_size=config["max_size"],
                    torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                    cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        write(stage/"prediction_metadata.json",meta)
        with torch.inference_mode(),torch.autocast("cuda",enabled=False):
            for index,frozen in enumerate(contract["images"]):
                item=dataset[index]; target=item["target"]
                truth=_audit_target(target,frozen,contract["truth"][frozen["image_id"]])
                rgb_sha=_audit_rgb(Path(target["source"]["rgb_path"]),frozen)
                pixels=item["pixel_values"]
                input_hw=tuple(pixels.shape[-2:]); hw=tuple(target["original_hw"])
                for view in ("original","hflip"):
                    view_pixels=pixels if view=="original" else pixels.flip(-1).contiguous()
                    raw=model(view_pixels[None].to("cuda"),pixel_mask=torch.ones((1,*input_hw),device="cuda",dtype=torch.bool))
                    prediction=_validate_prediction(raw,input_hw,hw,mask_chunk_size=8)
                    boxes=prediction["boxes"]
                    if view=="hflip":
                        b=boxes.clone();boxes[:,0]=hw[1]-b[:,2];boxes[:,2]=hw[1]-b[:,0]
                    records[view].append({"image_id":frozen["image_id"],"gt":{"boxes":truth["boxes"]},
                                          "pred":{"boxes":boxes.tolist(),"scores":prediction["scores"]}})
                    if view=="original" and args.model=="base" and not args.skip_mask_diagnostics:
                        overlaps=_roi_mask_overlaps(prediction,truth,hw)
                        diagnostic={"image_id":frozen["image_id"],"gt_count":len(truth["boxes"]),
                                    "scores":prediction["scores"],"apple_ids_in_target_order":target["instance_ids"]}
                        match=metrics.match_image(truth["boxes"],boxes.numpy(),prediction["scores"],.5,0.)
                        assigned=[-1]*len(boxes)
                        for p,g in zip(match["pred_indices"],match["gt_indices"],strict=True):assigned[int(p)]=int(g)
                        diagnostic["box_gt_indices_by_prediction"]=assigned
                        for branch,matrix in overlaps.items():
                            diagnostic[branch+"_gt_indices_by_prediction"]=_greedy_assignment(matrix,prediction["scores"])
                        diagnostics.append(diagnostic)
                        del overlaps
                    del raw,prediction,view_pixels
                _audit_target(target,frozen,contract["truth"][frozen["image_id"]])
                assert _audit_rgb(Path(target["source"]["rgb_path"]),frozen)==rgb_sha
                if (index+1)%25==0 or index+1==814:
                    meta.update(images=index+1,GT=sum(len(r["gt"]["boxes"]) for r in records["original"]),
                                elapsed_seconds=time.monotonic()-started,updated_utc=datetime.now(timezone.utc).isoformat())
                    write(stage/"progress.json",meta)
                    print(f"STRONG {args.model} {index+1}/814 bothviews elapsed={meta['elapsed_seconds']:.1f}s",flush=True)
                del item,target,truth,pixels
        assert source_inventory()==inventory,"Strong runtime source changed during reproduction"
        view_metrics={}
        for view,layer in records.items():
            assert len(layer)==814 and sum(len(r["gt"]["boxes"]) for r in layer)==3449
            write(stage/f"{view}_predictions.json",layer)
            calibrated=metrics.calibrate_threshold(layer,.5,return_curve=True)
            eligible=[r for r in calibrated["curve"] if r["recall"]>=.90]
            best=max(eligible,key=lambda r:(r["precision"],r["recall"],r["f1"],r["confidence"])) if eligible else None
            view_metrics[view]={"balanced":calibrated["summary"],"at_recall90":best}
        if diagnostics:
            assert len(diagnostics)==814
            conf=view_metrics["original"]["at_recall90"]["confidence"]
            mask_metrics={b:_mask_pr(diagnostics,b+"_mask",conf,3449,metrics) for b in ("visible","amodal")}
            write(stage/"mask_pair_diagnostics.json",diagnostics)
            write(stage/"mask_metrics_at_recall90.json",{"confidence":conf,"metrics":mask_metrics,
                  "groups":_occlusion(diagnostics,contract,conf)["groups"],"scope":"Base original view only; all native ROI masks independently matched"})
        meta.update(phase="complete",images=814,GT=3449,empty=13,source_before_after_identical=True,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                    view_metrics=view_metrics,elapsed_seconds=time.monotonic()-started)
        write(stage/"prediction_metadata.json",meta)
        print(json.dumps(meta,ensure_ascii=False),flush=True)
        return 0
    except BaseException as error:
        meta.update(phase="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        write(stage/"prediction_metadata.json",meta)
        raise


if __name__=="__main__":
    raise SystemExit(main())
