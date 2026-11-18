"""A single matched training intervention for the deployed 1280px apple crop distribution."""
import argparse
import collections
import hashlib
import inspect
import os
from pathlib import Path

import cv2
import numpy as np
import torch

import afruit_grade_model as grade
import afruit_grade_train as paired
import afruit_grade_release as baseline
import afruit_visible_quality as visible


OUT = grade.ROOT / "experiments/afruit_grade_crop_alignment_v1"
SIDE = 1280
CROP_SEED = 2026111813


def freeze():
    base = paired.read(baseline.OUT/"selection.json")
    data = dict(
        initial_protocol=grade.file_sha(grade.OUT/"protocol.json"),
        baseline_selection=grade.file_sha(baseline.OUT/"selection.json"),
        baseline_weight=base["weights_sha256"], maturity_weight=visible.STAGE_WEIGHT_SHA,
        crop_context=visible.CROP_CONTEXT, long_edge=SIDE, manifest_sha256=grade.file_sha(paired.MANIFEST),
        selected_splits=["train","val"], untouched_split="test",
        crop_source="Exact QualityPipeline after prepare_input(frame,1280), scene mode. Exactly one detected fruit with a valid 10%-context crop is eligible. No manual or image-label crop selection.",
        train_change="On every eligible train exposure an independent Bernoulli(0.5) chooses the actual deployment crop, otherwise the original image. Unresolved train exposures always use the original. Same batch ordering, augmentation, optimizer, initializer, 24 epochs, 336 updates, existing validation checkpoint/guard selection on whole images.",
        validation_primary="Same 57 validation images. At 1280px scene input, unresolved if no/multiple fruit or invalid crop. Count all 57, with a fourth unresolved column and original class denominators. Compare selected models and cutoffs on identical crops.",
        validation_secondary="Original supplied full fruit at the same working point as each arm's checkpoint selection.",
        gate="On auto crops, strictly more medium correct, no more severe->good, no fewer severe correct, no fewer total correct; also no more severe->good on original whole images. All five conditions required.",
        history="Previous V1 test had already been inspected before this experiment. No test pixel, prediction, or ground truth is read by this study, even when evaluating its gate. A future test comparison can only be a retrospective regression measurement.",
        crop_coin_seed=CROP_SEED, training_seed=paired.SEED,source_sha256=grade.file_sha(__file__),
        crop_functions={f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                        for f in [visible.crop_bounds,visible.prepare_input,visible.QualityPipeline.__call__]})
    if data["crop_context"]!=.10 or base["manifest_sha256"]!=data["manifest_sha256"]:
        raise ValueError("The deployed crop policy or grouped AFruitDB manifest changed")
    f = OUT/"protocol.json"
    if f.exists():
        if paired.read(f)!=data:
            raise ValueError("Frozen protocol no longer matches the source")
    else:
        paired.atomic(f,data)


def split_rows(tag):
    if tag not in ("train","val"):
        raise ValueError("This study allows only train and validation")
    rows = [r for r in paired.read(paired.MANIFEST)["samples"] if r["split"]==tag]
    if len(rows)!={"train":315,"val":57}[tag]:
        raise ValueError("Unrecognized split")
    return rows


def read_frame(row):
    path = grade.ROOT/row["path"]
    if grade.file_sha(path)!=row["sha256"]:
        raise ValueError("Original image no longer matches manifest")
    im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("Could not decode original apple photo")
    return im


def cache():
    freeze()
    if (OUT/"crop_cache.json").exists():
        raise ValueError("A verified deployment crop cache already exists")
    pipe=visible.QualityPipeline()
    summary={}
    for subset in ("train","val"):
        records=split_rows(subset)
        squares=np.zeros((len(records), grade.SIDE,grade.SIDE,3),np.uint8)
        evidence=[]
        for i,row in enumerate(records):
            im=visible.prepare_input(read_frame(row), SIDE)
            result,_=pipe(im,known_single_fruit=False)
            fruits=result["fruits"]
            bounds=fruits[0]["crop"] if len(fruits)==1 else None
            if bounds is not None:
                x,y,r,b=bounds
                squares[i]=grade.rgb_square(im[y:b,x:r])
            evidence.append(dict(sha256=row["sha256"],grade=row["grade"],group_id=row["group_id"],
                                 count=len(fruits),crop=bounds,hw=list(im.shape[:2])))
        file=OUT/f"{subset}_crops.npy"
        with file.with_suffix(".tmp").open("wb") as fh:
            np.save(fh,squares,allow_pickle=False)
        os.replace(file.with_suffix(".tmp"),file)
        paired.atomic(OUT/f"{subset}_regions.json",evidence)
        summary[subset]=dict(n=len(records),eligible=sum(v["crop"] is not None for v in evidence),
                  per_class={grade.LABELS[c]:dict(n=sum(v["grade"]==c for v in evidence),
                            eligible=sum(v["grade"]==c and v["crop"] is not None for v in evidence)) for c in range(3)},
                  region_sha=grade.file_sha(OUT/f"{subset}_regions.json"),numpy_sha=grade.file_sha(file))
    paired.atomic(OUT/"crop_cache.json",summary)
    print(summary,flush=True)


def load_cache(subset,records):
    summary=paired.read(OUT/"crop_cache.json")[subset]
    pixels=OUT/f"{subset}_crops.npy"
    regions=OUT/f"{subset}_regions.json"
    if grade.file_sha(pixels)!=summary["numpy_sha"] or grade.file_sha(regions)!=summary["region_sha"]:
        raise ValueError("Crop cache changed")
    meta=paired.read(regions)
    if [r["sha256"] for r in meta]!=[r["sha256"] for r in records]:
        raise ValueError("Crop order / underlying images changed")
    array=np.load(pixels,allow_pickle=False)
    return [array[i] if v["crop"] is not None else None for i,v in enumerate(meta)]


def train():
    freeze()
    original_read, common_augment=paired.open_rows,paired.augment
    coin=np.random.default_rng(CROP_SEED)
    crop_lookup={}
    counts=collections.Counter()
    def bind(subsets):
        if set(subsets)!={"train","val"}:
            raise ValueError("Only training and validation may be consumed")
        data=original_read(subsets)
        rows,ims,_=data["train"]
        crops=load_cache("train",rows)
        for im,crop in zip(ims,crops,strict=True):
            crop_lookup[id(im)]=crop
        return data
    def augment(im,rng):
        crop=crop_lookup[id(im)]
        use=coin.random()<.5 and crop is not None
        counts["auto_crop" if use else "whole"]+=1
        return common_augment(crop if use else im,rng)
    paired.OUT=OUT
    paired.freeze=freeze
    paired.open_rows=bind
    paired.augment=augment
    paired.train()
    if sum(counts.values())!=315*paired.EPOCHS:
        raise ValueError("Unexpected number of training exposures")
    paired.atomic(OUT/"training_exposures.json",dict(counts))


def deployed_model():
    selected=paired.read(OUT/"selection.json")
    if selected["protocol_sha256"]!=grade.file_sha(OUT/"protocol.json") or selected["weights_sha256"]!=grade.file_sha(OUT/"weights.pt"):
        raise ValueError("Crop-aligned model or protocol differs from frozen checkpoint")
    model=grade.network()
    model.load_state_dict(torch.load(OUT/"weights.pt",map_location="cpu",weights_only=True))
    return model.eval().cuda().requires_grad_(False),selected["threshold"]


def score(truth,pred):
    result=np.zeros((3,4),int)
    for y,p in zip(truth,pred,strict=True):
        result[y,p]+=1
    return dict(confusion=result.tolist(),columns=["good","medium","bad","unresolved"],
                correct=int(np.trace(result)),medium_correct=int(result[1,1]),severe_correct=int(result[2,2]),
                severe_to_good=int(result[2,0]),unresolved=int(result[:,3].sum()),n=len(pred))


def screen():
    freeze()
    if (OUT/"gate.json").exists():
        raise ValueError("Frozen validation gate has already been evaluated")
    rows,ims,gt=paired.open_rows(("val",))["val"]
    crops=load_cache("val",rows)
    ids=[i for i,c in enumerate(crops) if c is not None]
    metrics={}; ledger={}
    for mode,(model,threshold) in (("baseline",baseline.load_model("cuda")),("crop_aligned",deployed_model())):
        whole=grade.decide(paired.logits_for(model,ims),threshold).tolist()
        chosen=grade.decide(paired.logits_for(model,[crops[i] for i in ids]),threshold).tolist() if ids else []
        auto=np.full(len(rows),3)
        auto[ids]=chosen
        metrics[mode]=dict(whole=score(gt,whole),auto=score(gt,auto))
        ledger[mode]=[dict(sha256=r["sha256"],grade=r["grade"],whole=int(full),auto=int(instance)) for r,full,instance in zip(rows,whole,auto,strict=True)]
    a,b=metrics["baseline"]["auto"],metrics["crop_aligned"]["auto"]
    checks=dict(medium_gain=b["medium_correct"]>a["medium_correct"],
                no_new_severe_to_good=b["severe_to_good"]<=a["severe_to_good"],
                protect_severe_correct=b["severe_correct"]>=a["severe_correct"],
                protect_total=b["correct"]>=a["correct"],
                whole_no_new_severe_to_good=metrics["crop_aligned"]["whole"]["severe_to_good"]<=metrics["baseline"]["whole"]["severe_to_good"])
    paired.atomic(OUT/"paired_val_predictions.json",ledger)
    receipt=dict(metrics=metrics,conditions=checks,passed=all(checks.values()),n=len(rows),
                 protocol_sha256=grade.file_sha(OUT/"protocol.json"),input_manifest_sha256=grade.file_sha(paired.MANIFEST),
                 test_opened=False)
    paired.atomic(OUT/"gate.json",receipt)
    print(receipt,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["freeze","cache","train","screen"])
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    {"freeze":freeze,"cache":cache,"train":train,"screen":screen}[parser.parse_args().action]()
