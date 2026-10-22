"""Three fixed real-TRAIN 1024px residual-query GPU updates; no accuracy claim.

Source planning is standard-library-only. Actual numerical execution is entered
only by the pinned owned-child CLI after native/WDDM/concurrency/resource gates.
The partial smoke checkpoint is forbidden as a formal training initialization.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "temp/query_dual_mask_candidate_20261021"
SCRIPTS = ROOT / "scripts"
PAPPLE = ROOT / "data/papple/data"
PYTHON = Path("F:/anaconda/envs/falldet/python.exe")
SEED = 20261022
STAGES = (1024,)
PINS = {
    HERE / "residual_dual_model.py": "bfacdcb32850d6c13a7c932eaf27860c947af7fb56cc6328d52a569f65b8c1d0",
    HERE / "residual_dual_criterion.py": "ce927df75dc05aaa968c2f0c1daee315deadb97d1d2aff10eff2df08510ce392",
    HERE / "residual_papple_data.py": "e43e8e1865d5e5215e4eace12fdf7f9713a35154a558d7e8970bbb3537b0078c",
    HERE / "residual_dual_postprocess.py": "20cf95359a4e7db1f9cd444c602a3fb7a55bbb1a67550e393eee06cbdbade3d7",
    HERE / "residual_dual_evaluate.py": "2b150dd1fec1d5a0ecb0d7c6a7f33ba1df0ca0e66edf53980709ce936c65d407",
    OLD / "query_model_runtime.py": "d3aae8748062bf76aca7639bf0c7e169c74cedde73269cedd30302e81df16d5e",
    OLD / "query_dual_mask.py": "cef59347445af673eceff65cd5d0ed7caa5769683c5e78eb7f4610deb2e48878",
    OLD / "query_dual_criterion.py": "6ceec79c357b0e30864dfd47e0113b574ce7b65eecd24c40ae193bdf9e5b74ee",
    OLD / "query_papple_data.py": "1fbb142f4710454be90833bdacc3ff4841c9339ca1d5332897f743da36be96df",
    OLD / "query_dual_postprocess.py": "d19cedb68be01e68957acff0026b3c50a6c6e4d7fe2bf176b317e37057cfbf9d",
    SCRIPTS / "dataset.py": "4d8fe0f1ac98e3b54dd4f029675ad240b621766140b259a537e99290ace9a0a0",
    SCRIPTS / "runtime.py": "a21eba633ba1d08cfe07b7695186d811f2e69ac560ce441a236ffce7b93fb64c",
    SCRIPTS / "atomic_io.py": "3b5ca14efc337ce3f511f60f1829c14c24e5ab6fc9772714e051bf265e645380",
    SCRIPTS / "train_refinement.py": "94dc54a5212b1e825295f1085bf6c6ce89dffaa10b331220afa08c20b573fc8f",
    SCRIPTS / "refinement_config.py": "20b31460a36b93d5816358c1f40169ed081d78577b095eb7033fc59cddecd364",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    from atomic_io import replace_with_retry
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def metadata(args):
    """Source/annotation planning only; no model/Torch, image decode or writes."""
    if "torch" in sys.modules:
        raise RuntimeError("Metadata preparation must precede Torch import")
    if any(sha(path) != digest for path, digest in PINS.items()):
        raise ValueError("Frozen query/runtime source changed before metadata import")
    from residual_dual_model import local_residual_model_metadata
    from residual_papple_data import ANNOTATION_SHA256
    model = local_residual_model_metadata()
    sources = {str(path): digest for path, digest in PINS.items()}
    sources.update(model["source_sha256_expected"])
    sources[str(Path(__file__).resolve())] = sha(__file__)
    records = {}
    for branch, filename in (("visible", "via_region_data_instance.json"), ("amodal", "via_region_data_amodal.json")):
        path = PAPPLE / "gt_json/train" / filename
        sources[str(path)] = ANNOTATION_SHA256["train"][branch]
        if sha(path) != sources[str(path)]:
            raise ValueError("Real TRAIN polygon bytes changed")
        raw = read(path)
        raw = raw.get("_via_img_metadata", raw)
        rows = {}
        for value in raw.values():
            filename = value["filename"]
            if filename in rows or Path(filename).name != filename:
                raise ValueError("Duplicated or unsafe TRAIN filename")
            regions = value.get("regions", {})
            values = regions.values() if isinstance(regions, dict) else regions
            ids = {str(r["region_attributes"]["apple_ID"]).strip() for r in values}
            if "" in ids or "0" in ids:
                raise ValueError("Unexpected unresolved TRAIN apple ID")
            rows[filename] = sorted(ids)
        records[branch] = rows
    if records["visible"] != records["amodal"]:
        raise ValueError("Paired TRAIN original instance inventory differs")
    rows = sorted(records["visible"].items())
    ordinary = next(i for i, (_, ids) in enumerate(rows) if ids)
    empty = next(i for i, (_, ids) in enumerate(rows) if not ids)
    maximum = max(range(len(rows)), key=lambda i: len(rows[i][1]))  # first tie
    planned = []
    for kind, index in (("ordinary", ordinary), ("max_gt", maximum), ("empty", empty)):
        filename, ids = rows[index]
        image = PAPPLE / "images/train" / filename
        if not image.is_file() or len(ids) > 100:
            raise ValueError("Missing selected RGB or GT exceeds inherited query capacity")
        sources[str(image)] = sha(image)
        planned.append({"case_kind": kind, "index": index, "filename": filename,
                        "apple_ids": ids, "GT_count": len(ids)})
    if [(row["index"], row["GT_count"]) for row in planned] != [(0, 1), (475, 12), (63, 0)]:
        raise ValueError("Exactly the fixed ordinary0N1/max475N12/empty63N0 cases are required")
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError("Pinned source or data bytes changed")
    identity = {"schema": "residual_dual_three_real_gpu_smoke_v1", "seed": SEED,
        "stages": list(STAGES), "batch_size": 1, "workers": 0, "planned_cases": planned,
        "criterion_points": 12544, "amp": "float16", "GradScaler_init_scale": 128.,
        "optimizer": {"name": "AdamW", "encoder_lr": 1e-5, "old_other_lr": 5e-5, "new_head_lr": 2e-4, "weight_decay": 0.05, "gradient_clip_norm": 0.1, "foreach": False},
        "formal_training_resumable": False, "future_formal_warm_start_allowed": False,
        "accuracy_reported": False, "official_source": model["source_identity"]}
    return {"identity": identity, "source_sha256_expected": sources, "planned_cases": planned,
            "outer_config": model["outer_config"], "parent_checkpoint": model["parent_checkpoint"],
            "parent_supervisor": model["parent_supervisor"], "train_images_inventory": len(rows)}


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


def gradients(model, torch, *, empty):
    """Check every present gradient; record independent readouts and encoder."""
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError("Nonfinite unscaled gradient: " + name)
    visible = model.shared.transformer_module.decoder.mask_predictor.mask_embedder
    result = {}
    for name, module in (("new_amodal_MLP", model.amodal_mask_embedder),
                         ("visible_MLP", visible), ("classifier", model.class_predictor),
                         ("residual_box_head", model.residual_box_head)):
        values = {}
        for key, parameter in module.named_parameters():
            if parameter.grad is None:
                raise ValueError("Missing readout gradient: " + name + "." + key)
            values[key] = float(parameter.grad.abs().sum())
        if name == "classifier" or not empty:
            if sum(values.values()) <= 0:
                raise ValueError("Expected nonzero gradient in " + name)
        elif sum(values.values()) != 0:
            raise ValueError("Empty GT must give zero branch mask/box gradients")
        result[name] = values
    shared = next(((name, float(p.grad.abs().sum())) for name, p in
                   model.shared.pixel_level_module.encoder.named_parameters()
                   if p.grad is not None and bool(p.grad.abs().sum() > 0)), None)
    if shared is None:
        raise ValueError("Missing nonzero shared image-feature gradient")
    result["shared_image_feature_parameter"] = {shared[0]: shared[1]}
    return result


def checkpoint_roundtrip(model, optimizer, scaler, generator, evidence, output, torch, np):
    """Only this process's own atomic CPU partial checkpoint is trusted/reloaded."""
    from atomic_io import replace_with_retry
    def cpu_snapshot(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_snapshot(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_snapshot(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_snapshot(item) for item in value)
        return value
    expected = {key: tensor_sha(value) for key, value in model.state_dict().items()}
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
           "criterion_generator": generator.get_state()}
    checkpoint = {"schema": "residual_dual_smoke_partial_not_formal_resume_v1",
        "formal_training_resumable": False, "future_formal_warm_start_allowed": False,
        "completed_smoke_updates": 3, "identity": evidence["identity"],
        "source_sha256": evidence["source_sha256_expected"], "source_outer_config": evidence["outer_config"],
        "state_dict": cpu_snapshot(model.state_dict()), "optimizer": cpu_snapshot(optimizer.state_dict()),
        "scaler": scaler.state_dict(), "rng": rng}
    destination = output / "smoke.partial.pt"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        torch.save(checkpoint, stream); stream.flush(); os.fsync(stream.fileno())
    replace_with_retry(temporary, destination)
    digest = sha(destination)
    expected_scaler = dict(checkpoint["scaler"])
    optimizer_fingerprints = {str(index): {key: tensor_sha(value) if isinstance(value, torch.Tensor) else value
        for key, value in state.items()} for index, state in checkpoint["optimizer"]["state"].items()}
    expected_groups = checkpoint["optimizer"]["param_groups"]
    del checkpoint
    # Perturb every state family so the roundtrip must actually restore it.
    optimizer.param_groups[0]["lr"] *= 2
    scaler.update(new_scale=64.)
    random.random(); np.random.random(); torch.rand(1); torch.rand(1, device="cuda")
    torch.rand(1, device="cuda", generator=generator)
    with torch.no_grad():
        model.class_predictor.bias.add_(1.)  # A real restore is required.
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.add_(1.)
    restored = torch.load(destination, map_location="cpu", weights_only=False)
    if (sha(destination) != digest or restored["identity"] != evidence["identity"]
            or restored["source_outer_config"] != evidence["outer_config"]
            or restored["source_sha256"] != evidence["source_sha256_expected"]
            or restored["formal_training_resumable"] is not False
            or restored["future_formal_warm_start_allowed"] is not False
            or restored["completed_smoke_updates"] != 3):
        raise ValueError("Own partial smoke checkpoint identity/config mismatch")
    loaded = model.load_state_dict(restored["state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Strict own model checkpoint loading failed")
    optimizer.load_state_dict(restored["optimizer"]); scaler.load_state_dict(restored["scaler"])
    actual_optimizer = {str(index): {key: tensor_sha(value) if isinstance(value, torch.Tensor) else value
        for key, value in state.items()} for index, state in optimizer.state_dict()["state"].items()}
    if ({key: tensor_sha(value) for key, value in model.state_dict().items()} != expected
            or actual_optimizer != optimizer_fingerprints or optimizer.state_dict()["param_groups"] != expected_groups
            or scaler.state_dict() != expected_scaler):
        raise ValueError("Strict smoke state/optimizer/scaler roundtrip differs")
    random.setstate(restored["rng"]["python"]); np.random.set_state(restored["rng"]["numpy"])
    torch.set_rng_state(restored["rng"]["torch_cpu"]); torch.cuda.set_rng_state_all(restored["rng"]["torch_cuda"])
    generator.set_state(restored["rng"]["criterion_generator"])
    if (random.getstate() != rng["python"] or np.random.get_state()[0] != rng["numpy"][0]
            or not np.array_equal(np.random.get_state()[1], rng["numpy"][1])
            or np.random.get_state()[2:] != rng["numpy"][2:]
            or not torch.equal(torch.get_rng_state(), rng["torch_cpu"])
            or not torch.equal(generator.get_state(), rng["criterion_generator"])
            or not all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng["torch_cuda"], strict=True))):
        raise ValueError("RNG roundtrip differs")
    del restored
    return {"path": str(destination), "sha256": digest, "strict_model": True,
            "optimizer_scaler_RNG_restored": True, "formal_training_resumable": False,
            "future_formal_warm_start_allowed": False, "completed_smoke_updates": 3}


def run(evidence, output, report, psutil):
    from runtime import authorized_guard
    torch = authorized_guard()
    import numpy as np
    from residual_dual_model import load_residual_dual_model
    from residual_papple_data import build_residual_papple_dataset, collate_residual_papple_batch
    from residual_dual_criterion import build_residual_dual_criterion
    from residual_dual_postprocess import postprocess_single_frame
    report["versions"] = {name: version(name) for name in ("torch", "transformers", "scipy", "safetensors")}
    if report["versions"]["torch"] != "2.12.1+cu132" or report["versions"]["transformers"] != "4.57.1":
        raise RuntimeError("Exact pinned native runtime required")
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if psutil.virtual_memory().available < 4*1024**3:
        raise RuntimeError("Need4GiB available host RAM before CPU model construction")
    model, loading = load_residual_dual_model(device="cpu")
    parent = loading["parent_checkpoint"]
    if (parent.get("strict_parent_model_load") is not True or parent.get("checkpoint_payload_verified") is not True
            or parent.get("optimizer_restored") is not False):
        raise ValueError("Parent must strictly restore MODEL only before a fresh smoke optimizer")
    report.update(parent_checkpoint=parent, parent_supervisor=loading["parent_supervisor"],
        strict_parent_model_load=True, initialization=loading["initialization"],
        model_loaded=True, torch_imported=True, gpu={"name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory})
    model.cuda()
    criterion = build_residual_dual_criterion().cuda()
    report["criterion"] = criterion.configuration
    if criterion.configuration["match_points"] != 12544 or criterion.configuration["loss_points"] != 12544:
        raise ValueError("Retain complete12544-point matching and loss sampling")
    encoder_ids = {id(p) for p in model.shared.pixel_level_module.encoder.parameters()}
    head_ids = {id(p) for p in model.residual_box_head.parameters()}
    encoder, other, head = [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            raise ValueError("Every model parameter must remain trainable")
        (encoder if id(parameter) in encoder_ids else head if id(parameter) in head_ids else other).append(parameter)
    if not encoder or not other or not head or len(encoder)+len(other)+len(head) != len(list(model.parameters())):
        raise ValueError("Three disjoint complete optimizer groups required")
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": 1e-5, "name": "encoder"},
        {"params": other, "lr": 5e-5, "name": "old_other"},
        {"params": head, "lr": 2e-4, "name": "residual_box_head"}],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, foreach=False)
    report["optimizer_groups"] = [{"name": g["name"], "lr": g["lr"], "weight_decay": g["weight_decay"],
                                   "parameters": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    dataset = build_residual_papple_dataset(PAPPLE, "train", input_size=1024)
    if len(dataset) != 2304 or len(dataset) != evidence["train_images_inventory"]:
        raise ValueError("Unchanged complete public TRAIN inventory required")
    counts = [len(dataset.image_metadata(i)["apple_ids"]) for i in range(len(dataset))]
    indices = (next(i for i,n in enumerate(counts) if n), max(range(len(counts)),key=counts.__getitem__), next(i for i,n in enumerate(counts) if not n))
    if list(indices) != [0,475,63]:
        raise ValueError("Runtime selection must match fixed source-only real cases")
    update = 0
    torch.cuda.reset_peak_memory_stats()
    for selected in evidence["planned_cases"]:
        report.update(phase="residual_gpu_smoke_running", active_input_size=1024, active_case=selected["case_kind"])
        write(output/"report.json", report)
        item = dataset[selected["index"]]
        target = item["target"]
        count = selected["GT_count"]
        if list(item["apple_ids"]) != selected["apple_ids"] or len(target["labels"]) != count:
            raise ValueError("Selected original TRAIN IDs/count changed")
        before = {key:tensor_sha(value) for key,value in item["original_target"].items()}
        if (target["boxes"] is not item["original_target"]["boxes"]
                or target["visible_masks"] is not item["original_target"]["visible_masks"]
                or target["amodal_masks"] is not item["original_target"]["masks"]
                or tuple(target["amodal_masks"].shape) != (count,1300,1300)
                or tuple(target["visible_masks"].shape) != (count,1300,1300)
                or target["boxes"].dtype != torch.float32 or tuple(target["boxes"].shape) != (count,4)
                or any(target[key].device.type != "cpu" for key in ("boxes","labels","mask_valid","visible_masks","amodal_masks"))):
            raise ValueError("All original masks and exact CPU float32 boxes must remain unchanged")
        batch = collate_residual_papple_batch([item])
        pixels = batch["pixel_values"].cuda(); mask = batch["pixel_mask"].cuda()
        optimizer.zero_grad(set_to_none=True); model.train()
        torch.cuda.synchronize(); started=time.monotonic()
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            outputs=model(pixels,mask)
            losses=criterion(outputs,[target],generator=generator)
            total=sum(losses.values())
        if len(losses)!=14 or not bool(torch.isfinite(total)):
            raise ValueError("Fourteen finite real V/A/box and penultimate losses required")
        for name,value in losses.items():
            if count==0 and name not in ("loss_class","loss_class_aux") and float(value.detach())!=0.:
                raise ValueError("Every empty-GT mask/box loss must be exactly zero")
        scaler.scale(total).backward(); scaler.unscale_(optimizer)
        grad=gradients(model,torch,empty=count==0)
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),0.1,error_if_nonfinite=True))
        parameter=model.class_predictor.weight
        old_weight=parameter.detach().clone()
        scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
        update+=1
        if (torch.equal(parameter,old_weight) or float(optimizer.state[parameter]["step"])!=update or scaler.get_scale()!=128.):
            raise ValueError("One finite fresh AdamW update required; AMP must not skip")
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):
                    raise ValueError("Nonfinite optimizer state")
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("Nonfinite updated parameter")
        after={key:tensor_sha(value) for key,value in item["original_target"].items()}
        unchanged=before==after
        if not unchanged:
            raise ValueError("Original GT tensors changed during training")
        report["cases"].append({**selected,"input_shape":list(pixels.shape),"GT_shapes":[count,1300,1300],
            "box_shape":[count,4],"losses":{key:float(value.detach()) for key,value in losses.items()},
            "output_dtypes":{key:str(value.dtype) for key,value in outputs.items() if isinstance(value,torch.Tensor)},
            "auxiliary_stages":len(outputs["auxiliary_outputs"]),"gradients_abs_sum":grad,
            "finite":True,"optimizer_update_verified":True,"optimizer_update_number":update,
            "GTunchanged":unchanged,"original_GT_sha256_before":before,"original_GT_sha256_after":after,
            "unclipped_gradient_norm":norm,"elapsed_seconds":time.monotonic()-started,
            "peak_torch_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_torch_reserved_bytes":torch.cuda.max_memory_reserved(),
            "available_RAM_bytes":psutil.virtual_memory().available,"child_RSS_bytes":psutil.Process().memory_info().rss})
        optimizer.zero_grad(set_to_none=True)
        del outputs,losses,total,batch,target,item,old_weight,pixels,mask
        write(output/"report.json",report)
    if update!=3:
        raise ValueError("Exactly three bounded updates required")
    item=dataset[0]
    model.eval()
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.float16):
        outputs=model(item["pixel_values"][None].cuda(),item["pixel_mask"][None].cuda())
    prediction=postprocess_single_frame(outputs,(1300,1300),return_masks=False)
    boxes,scores=prediction["boxes"],prediction["scores"]
    invalid=len(boxes) and (not bool((boxes[:,:2]>=0).all()) or not bool((boxes[:,2:]<=1300).all()) or not bool((boxes[:,2:]>boxes[:,:2]).all()))
    if invalid or len(boxes)>100 or bool((prediction["labels"]!=1).any()) or not bool(torch.isfinite(boxes).all() & torch.isfinite(scores).all()):
        raise ValueError("Original-coordinate clipped postprocess bounds failed")
    report.update(postprocess_bounds_passed=True,postprocess={"metadata":prediction["metadata"],"boxes":boxes.tolist(),"scores":scores.tolist(),"metrics_reported":False})
    del prediction,outputs,item,dataset,criterion
    report["checkpoint"]=checkpoint_roundtrip(model,optimizer,scaler,generator,evidence,output,torch,np)
    report.update(checkpoint_roundtrip_passed=True,completed_smoke_updates=update,
        checkpoint_path=report["checkpoint"]["path"],checkpoint_sha256=report["checkpoint"]["sha256"],
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output != HERE / "residual_gpu_smoke01" or output.exists() or Path(sys.executable).resolve() != PYTHON.resolve():
        parser.error("Exact native F: falldet Python and fresh direct candidate output required")
    sys.path.insert(0, str(SCRIPTS))
    from runtime import configure_paths
    if Path(configure_paths()).resolve() != ROOT:
        raise RuntimeError("Fixed authorized F: root required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_CACHE=str(ROOT / "cache/query_hf/hub"),
                      HUGGINGFACE_HUB_CACHE=str(ROOT / "cache/query_hf/hub"),
                      TORCH_EXTENSIONS_DIR=str(ROOT / "cache/query_torch_extensions"))
    sys.dont_write_bytecode = True
    import psutil
    from watch_residual_dual_gpu import preflight, configure_native_imports
    native_import_environment = configure_native_imports()
    preflight(psutil)
    evidence = metadata(args)
    from train_refinement import TrainingLock
    with TrainingLock(SCRIPTS / "training.lock", output, psutil):
        preflight(psutil)
        output.mkdir(exist_ok=False)
        report = {"phase": "initializing", "identity": evidence["identity"], "input_size": 1024, "stages": list(STAGES),
            "cases_per_stage": 3, "planned_cases": evidence["planned_cases"], "cases": [],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256_before": evidence["source_sha256_expected"], "model_loaded": False,
            "torch_imported": False, "test_used": False, "accuracy_reported": False,
            "native_import_environment": native_import_environment,
            "formal_training_resumable": False, "future_formal_warm_start_allowed": False,
            "limitations": "Three fixed-case1024 smoke updates only; no PR/AP, quality, throughput or general8GiB fit claim"}
        write(output / "report.json", report)
        try:
            run(evidence, output, report, psutil)
            report.update(phase="residual_gpu_smoke_passed", completed_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            report.update(phase="failed", error=repr(error))
            raise
        finally:
            actual = {path: sha(path) for path in evidence["source_sha256_expected"]}
            report.update(source_sha256_after=actual,
                          source_before_after_identical=actual == evidence["source_sha256_expected"])
            if not report["source_before_after_identical"]:
                report.update(phase="failed", source_changed=True)
            write(output / "report.json", report)
            if not report["source_before_after_identical"]:
                raise RuntimeError("Consumed source/image/GT/weight bytes changed")
    print(json.dumps({"phase": report["phase"], "report": str(output / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
