"""Frozen-mask training of dual-evidence, query-competitive quality scores."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import psutil
import torch

from .data import PAppleDataset, MinneAppleDataset, flip_item_horizontal
from .evaluate import evaluate_papple, REFERENCE_PINS
from .io_compat import install_windows_replace_retry
from .losses import build_annotation_aware_criterion
from .model import from_checkpoint
from .quality import FrozenQualityModel, quality_targets, quality_focal_loss
from .training import CyclingOrder, atomic_json, sha, target_fingerprint, _rng_state, _restore_rng

SCHEMA = "apple_dual_evidence_quality_training_v1"
SOURCE_ORDER = ("PApple", "MinneApple", "PApple", "MinneApple")


def load_quality_config(path, *, output=None, smoke=False):
    path = Path(path).absolute()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("schema") != SCHEMA:
        raise ValueError("Expected the dedicated quality-training schema")
    root = Path(os.path.abspath(path.parent / config["path_base"]))
    paths = {}
    for name,value in config["paths"].items():
        resolved = Path(os.path.abspath(root/value))
        if Path(value).is_absolute() or os.path.commonpath((str(root),str(resolved))) != str(root):
            raise ValueError("Quality config path escapes its project: " + name)
        paths[name] = resolved
    recipe = config["training"]
    for key in ("input_size","max_updates","validate_every","checkpoint_every","seed","cpu_threads"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError("Positive integer required: " + key)
    for key in ("competition","evidence","augment"):
        if type(recipe[key]) is not bool:
            raise ValueError("Boolean required: " + key)
    for key in ("quality_lr","weight_decay"):
        if isinstance(recipe[key],bool) or not math.isfinite(recipe[key]) or recipe[key] <= 0:
            raise ValueError("Positive finite value required: " + key)
    return dict(path=path, root=root, raw=config, paths=paths, recipe=recipe,
                output=Path(output).absolute() if output else paths["output"], smoke=smoke)


def training_sources():
    names = ("quality.py","quality_training.py","model.py","data.py","losses.py","evaluate.py","training.py","io_compat.py")
    return {name:sha(Path(__file__).with_name(name)) for name in names}


def code_version(root,config_path,smoke):
    commit = subprocess.run(["git","rev-parse","HEAD"],cwd=root,check=True,capture_output=True,text=True).stdout.strip()
    consumed = ["src/apple_seg/"+name for name in training_sources()]
    consumed += ["scripts/train_quality.py",config_path.relative_to(root).as_posix()]
    mismatches = []
    for relative in consumed:
        blob = subprocess.run(["git","show",commit+":"+relative],cwd=root,capture_output=True)
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != sha(root/relative):
            mismatches.append(relative)
    if mismatches and not smoke:
        raise ValueError("Commit consumed code/config before formal training: " + ", ".join(mismatches))
    return dict(training_commit=commit,consumed_files_match_commit=not mismatches,uncommitted_consumed_files=mismatches)


def save_payload(path, payload):
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("wb") as stream:
        torch.save(payload,stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)
    digest = sha(path)
    temporary_sha = path.with_suffix(".pt.sha256.tmp")
    temporary_sha.write_text(digest,encoding="ascii")
    os.replace(temporary_sha,path.with_suffix(".pt.sha256"))
    return digest


def run_quality_training(config_path, *, device="cuda", output=None, smoke=False, resume=False):
    install_windows_replace_retry()
    cfg = load_quality_config(config_path,output=output,smoke=smoke)
    root,out,recipe,paths = cfg["root"],cfg["output"],cfg["recipe"],cfg["paths"]
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError("Use a fresh quality experiment output or --resume")
    out.mkdir(parents=True,exist_ok=True)
    lock = out/"trainer.lock"
    if lock.exists():
        previous = json.loads(lock.read_text())
        try:
            process = psutil.Process(previous["pid"])
            alive = process.is_running() and abs(process.create_time()-previous["created"]) < .01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError("A verified live quality trainer already owns this output")
        lock.unlink()
    with lock.open("x",encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),stream)
    torch.set_num_threads(recipe["cpu_threads"])
    torch.manual_seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    random.seed(recipe["seed"])
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this launch")
    sources = training_sources()
    version = code_version(root,cfg["path"],smoke)
    for path,digest in REFERENCE_PINS.items():
        if sha(path) != digest:
            raise ValueError("Frozen postprocessing/metric source differs")
    parent,provenance = from_checkpoint(paths["parent_checkpoint"],cfg["raw"]["parent_sha256"])
    model = FrozenQualityModel(parent,competition=recipe["competition"],evidence=recipe["evidence"]).to(dev).train()
    optimizer = torch.optim.AdamW(model.quality_head.parameters(),lr=recipe["quality_lr"],weight_decay=recipe["weight_decay"])
    criterion = build_annotation_aware_criterion().to(dev)
    datasets = {"PApple":PAppleDataset(paths["papple_root"],"train",recipe["input_size"]),
                "MinneApple":MinneAppleDataset(paths["minne_root"],paths["minne_manifest"],split="train",input_size=recipe["input_size"],tiled=True)}
    metadata = {name:dataset.metadata() for name,dataset in datasets.items()}
    identity = dict(recipe=recipe, smoke=smoke, parent_sha256=cfg["raw"]["parent_sha256"],source_sha256=sources,
                    datasets=metadata, papple_provenance_sha256=sha(paths["papple_provenance"]),
                    minne_manifest_sha256=sha(paths["minne_manifest"]), source_order=list(SOURCE_ORDER),
                    teacher_precision="float32",quality_precision="float32",parent_frozen=True)
    cache_identity = dict(parent_sha256=identity["parent_sha256"],input_size=recipe["input_size"],sources=sources,
                          datasets=metadata,criterion=criterion.configuration)
    cache_key = hashlib.sha256(json.dumps(cache_identity,sort_keys=True).encode()).hexdigest()
    cache = root/"cache/quality_features"/cache_key
    cache.mkdir(parents=True,exist_ok=True)
    samplers = {name:CyclingOrder(len(dataset),recipe["seed"]+i) for i,(name,dataset) in enumerate(datasets.items())}
    generator = torch.Generator(device=dev).manual_seed(recipe["seed"])
    updates,history,validations,best = 0,[],[],None
    selection_floor = float(cfg["raw"]["parent_selection_floor"])
    maximum = 2 if smoke else recipe["max_updates"]
    if resume:
        checkpoint = out/"last.pt"
        if sha(checkpoint) != checkpoint.with_suffix(".pt.sha256").read_text().strip():
            raise ValueError("Quality checkpoint hash differs")
        saved = torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=False)
        if saved["schema"] != SCHEMA or saved["identity"] != identity or saved["smoke"]:
            raise ValueError("Resume requires this exact formal quality experiment")
        model.quality_head.load_state_dict(saved["quality_head"],strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for name,sampler in samplers.items():
            sampler.load_state_dict(saved["samplers"][name])
        _restore_rng(saved["rng"],torch,np,generator,dev.type=="cuda")
        updates,history,validations,best = saved["updates"],saved["history"],saved["validations"],saved["best"]
        version = {**saved["code_version"],"resumed_on_commit":version["training_commit"]}
        del saved
    stop = dict(requested=False)
    previous_signal = signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True))
    atomic_json(out/"training_status.json",dict(phase="initializing",pid=os.getpid(),identity=identity,updates=updates,code_version=version))

    def boundary(phase):
        if training_sources() != sources:
            raise RuntimeError("Consumed quality training sources changed")
        payload = dict(schema=SCHEMA,identity=identity,smoke=smoke,phase=phase,updates=updates,
                       quality_head=model.quality_head.state_dict(),optimizer=optimizer.state_dict(),
                       samplers={name:s.state_dict() for name,s in samplers.items()},
                       rng=_rng_state(torch,np,generator,dev.type=="cuda"),history=history,validations=validations,best=best,
                       parent_provenance=provenance,cache_key=cache_key,code_version=version)
        digest = save_payload(out/"last.pt",payload)
        return payload,digest

    def example(source,index,flipped):
        item = datasets[source][index]
        if flipped:
            item = flip_item_horizontal(item)
        fingerprint = target_fingerprint(item["target"])
        binding = dict(target=fingerprint,rgb_sha256=sha(item["target"]["source"]["rgb_path"]),
                       dataset=source,index=index,flipped=flipped)
        destination = cache/source/f"{index:06d}_{int(flipped)}.pt"
        if destination.exists():
            expected = destination.with_suffix(".pt.sha256").read_text().strip()
            if sha(destination) != expected:
                raise ValueError("Cached TRAIN features changed")
            entry = torch.load(destination,map_location="cpu",weights_only=True)
            if entry["binding"] != binding:
                raise ValueError("Cached TRAIN annotation/RGB identity differs")
            return entry
        original,features = model.extract(item["pixel_values"].unsqueeze(0).to(dev),item["pixel_mask"].unsqueeze(0).to(dev))
        # Cache hits must not change later Hungarian point samples or ablations.
        sample_seed = int(hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()[:16],16) % (2**63)
        sample_generator = torch.Generator(device=dev).manual_seed(sample_seed)
        supervision = quality_targets(original,item["target"],criterion=criterion,generator=sample_generator)
        if target_fingerprint(item["target"]) != fingerprint:
            raise RuntimeError("Quality-target construction mutated original GT")
        classes = original["class_queries_logits"].float()
        entry = {"binding":binding,"features":features[0].detach().cpu(),
                 "parent_margin":(classes[0,:,0]-classes[0,:,1]).detach().cpu(),**supervision}
        entry["quality_targets"] = entry["quality_targets"][0]
        destination.parent.mkdir(exist_ok=True)
        save_payload(destination,entry)
        return entry

    try:
        while updates < maximum and not stop["requested"] and not (out/"STOP").exists():
            started = time.monotonic()
            examples = []
            for source in SOURCE_ORDER:
                examples.append(example(source,samplers[source].next(),recipe["augment"] and random.random()<.5))
            features = torch.stack([entry["features"] for entry in examples]).to(dev)
            margin = torch.stack([entry["parent_margin"] for entry in examples]).to(dev)
            targets = torch.stack([entry["quality_targets"] for entry in examples]).to(dev)
            valid = torch.stack([entry["quality_valid"] for entry in examples]).to(dev)
            counts = torch.tensor([entry["GT_count"] for entry in examples],device=dev)
            optimizer.zero_grad(set_to_none=True)
            logits = margin.unsqueeze(-1)+model.quality_head(features)
            losses = quality_focal_loss(logits,targets,valid,counts)
            loss = losses["amodal_box"]+.5*losses["amodal_mask"]+.5*losses["visible_mask"]
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite quality loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.quality_head.parameters(),1.,error_if_nonfinite=True)
            if any(p.grad is not None or p.requires_grad for p in model.parent.parameters()):
                raise RuntimeError("Frozen parent unexpectedly entered the optimization graph")
            optimizer.step()
            updates += 1
            row = dict(update=updates,loss=float(loss.detach()),losses={k:float(v.detach()) for k,v in losses.items()},
                       grad_norm=float(norm),seconds=time.monotonic()-started,
                       samples=[{k:entry["binding"][k] for k in ("dataset","index","flipped")} for entry in examples])
            history.append(row)
            print(f"QUALITY successful={updates}/{maximum} loss={row['loss']:.6f} seconds={row['seconds']:.2f}",flush=True)
            if not smoke and (updates%recipe["validate_every"]==0 or updates==maximum):
                print(f"QUALITY_VALIDATION starting update={updates} images=814",flush=True)
                validation_dir = out/"validation"/f"update_{updates:06d}_{time.time_ns()}"
                dataset = PAppleDataset(paths["papple_root"],"val",recipe["input_size"])
                evaluated = evaluate_papple(model,dataset,dev,validation_dir,paths["papple_provenance"])
                summary = evaluated["summary"]
                if evaluated["phase"]!="complete" or not evaluated["model_selection_eligible"] or (summary["images"],summary["gt_count"],summary["empty_gt_images"])!=(814,3449,13):
                    raise ValueError("Only complete original PApple VAL can select quality weights")
                point = dict(update=updates,confidence=evaluated["confidence"],score=evaluated["score"],summary=summary,
                             directory=validation_dir.relative_to(out).as_posix())
                validations.append(point)
                improved = point["score"] > (best["score"] if best else selection_floor)
                if improved:
                    best = point
                payload,digest = boundary("training")
                save_payload(out/f"validated_{updates:06d}.pt",payload)
                if improved:
                    save_payload(out/"best.pt",payload)
                print(f"QUALITY_VALIDATION complete update={updates} precision={summary['precision']:.6f} recall={summary['recall']:.6f} improved={improved}",flush=True)
            elif updates%recipe["checkpoint_every"]==0:
                boundary("training")
            atomic_json(out/"training_status.json",dict(phase="training",pid=os.getpid(),updates=updates,
                        last_update=row,best=best,parent_selection_floor=selection_floor,cache_key=cache_key,code_version=version))
        phase = ("smoke_complete" if smoke else "complete") if updates==maximum else "interrupted"
        payload,digest = boundary(phase)
        restored = torch.load(out/"last.pt",map_location="cpu",weights_only=False)
        if any(not torch.equal(value.detach().cpu(),restored["quality_head"][key]) for key,value in model.quality_head.state_dict().items()):
            raise RuntimeError("Quality checkpoint tensor roundtrip differs")
        summary = dict(schema=SCHEMA,phase=phase,pid=os.getpid(),updates=updates,best=best,validations=validations,
                       identity=identity,checkpoint_sha256=digest,own_head_roundtrip_verified=True,
                       parent_frozen=True,parent_gradients_absent=True,code_version=version,
                       goal_achieved=bool(best and best["summary"]["goal_95_met"]))
        atomic_json(out/"training_summary.json",summary)
        atomic_json(out/"training_status.json",summary)
        return summary
    except BaseException as error:
        atomic_json(out/"training_status.json",dict(phase="failed",pid=os.getpid(),updates=updates,
                    failure_type=type(error).__name__,failure=str(error)))
        raise
    finally:
        signal.signal(signal.SIGINT,previous_signal)
        lock.unlink(missing_ok=True)
