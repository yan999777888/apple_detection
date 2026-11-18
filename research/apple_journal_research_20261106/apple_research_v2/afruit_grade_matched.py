"""Controlled continuation from the crop-aligned apple grade checkpoint."""
import argparse
import collections
import hashlib
import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import afruit_grade_crop_alignment as aligned
import afruit_grade_model as grade
import afruit_grade_train as original


OUT = grade.ROOT / "experiments/afruit_grade_matched_v1"
AUX = grade.ROOT / "datasets/apple_matched_quality_v1/manifest.json"
SEED = 19112026
EPOCHS = 20
STEPS = 18
AF_PER_CLASS = 6
EXTERNAL_PAIRS = 6
ARMS = ("control", "paired")


def freeze():
    start = original.read(aligned.OUT / "selection.json")["weights_sha256"]
    if grade.file_sha(aligned.OUT / "weights.pt") != start:
        raise ValueError("Parent weights changed")
    protocol = dict(
        parent_sha256=start, original_manifest_sha256=grade.file_sha(original.MANIFEST),
        auto_cache_sha256=grade.file_sha(aligned.OUT / "crop_cache.json"), source_sha256=grade.file_sha(__file__),
        arm_descriptions={"control":"AF grade- and capture-group-balanced replay with crop mixing",
                          "paired":"same AF replay and training plus paired public good/minor examples"},
        optimizer="20 epochs x 18 AdamW updates. Backbone 0.00010 and classifier 0.00045 with cosine factor down to 0.1. BN running stats fixed.",
        replay="Each update 6 per original grade, then equal chance for each train acquisition group with that grade, then one image. For eligible fruits choose actual 1280px deployment crop or full image with equal probability.",
        auxiliary="6 pairs each update from a condition with both grades; uniformly sample condition and eligible group before image, with independent RNG for the auxiliary data. BCE-equivalent conditional good/minor CE plus within-condition ranking, both applied only during training.",
        objective="AF 3-grade CE (0.015 label smoothing), plus 0.30 BCE on old severe probability. Paired arm additionally 0.65 CE on external two logits + 0.15 softplus(0.5 - paired log-odds improvement).",
        validation="All 57 official validation photos; scene 1280 with zero or multiple instances unresolved. Also full original images; all metrics fixed argmax.",
        gate="Scene: zero severe->good, at least 2/15 minor correct, 16/17 severe correct, 19/25 good correct, 38/57 all correct. Whole photo zero severe->good.",
        selection="Among gate passers prioritize minor correct, then total correct then severe correct then good correct, then lower whole-image NLL; failing checkpoints first maximize safety conditions including severe route and minimum good and overall correctness.",
        test="No test pixels or predictions enter training, checkpoint selection or the decision. External evaluation partitions excluded.",
        epochs=EPOCHS, steps_per_epoch=STEPS, seed=SEED, threshold=1.,
    )
    lock = OUT / "protocol.json"
    if lock.is_file() and original.read(lock) != protocol:
        raise ValueError("Recipe changed after freezing")
    if not lock.is_file():
        original.atomic(lock, protocol)
    return protocol


def groups(rows, indices):
    result = collections.defaultdict(list)
    for i in indices:
        result[str(rows[i]["group_id"])].append(i)
    return list(result.values())


def pick(grouped, rng):
    members = grouped[rng.integers(len(grouped))]
    return members[rng.integers(len(members))]


def auxiliary():
    manifest = original.read(AUX)
    rows = manifest["samples"]
    owners, lookup = {}, {}
    for row in rows:
        for key in (("group", row.get("source"), row["group_id"]), ("sha", row["sha256"])):
            if owners.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("Auxiliary train/evaluation overlap")
    data = [r for r in rows if r["split"] == "train"]
    original_hashes = {r["sha256"] for r in original.read(original.MANIFEST)["samples"]}
    if len({r["sha256"] for r in data}) != len(data) or any(r["sha256"] in original_hashes for r in data):
        raise ValueError("Duplicate public training content or AF source overlap")
    for i,row in enumerate(data):
        level = int(row["grade"])
        if level not in (0,1):
            raise ValueError("Paired supervision requires explicit good or minor labels")
        key=(str(row.get("source","")),str(row["condition_id"]))
        lookup.setdefault(key, [[],[]])[level].append(i)
    conditions = [tuple(groups(data, part) for part in bins) for bins in lookup.values() if all(bins)]
    if not conditions:
        raise ValueError("Public source lacks training conditions containing both grades")
    images=[]
    for row in data:
        path=grade.ROOT / row["path"]
        if grade.file_sha(path)!=row["sha256"]:
            raise ValueError(f"Public input changed: {path.name}")
        image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
        if row.get("crop_xyxy") is not None:
            x,y,r,b=map(int,row["crop_xyxy"])
            if not (0<=x<r<=image.shape[1] and 0<=y<b<=image.shape[0]):
                raise ValueError("Invalid public fruit bounds")
            image=image[y:b,x:r]
        images.append(grade.rgb_square(image))
    return images,conditions


def evaluate(model, validation):
    _, images, truth, crops, valid = validation
    whole_z=original.logits_for(model,images)
    result=np.full(len(truth),3)
    if valid:
        result[valid]=original.logits_for(model,[crops[i] for i in valid]).argmax(1).numpy()
    return dict(whole=aligned.score(truth, whole_z.argmax(1).numpy()),auto=aligned.score(truth,result),
                nll=float(F.cross_entropy(whole_z,torch.from_numpy(truth))))


def check(metrics):
    a,w=metrics["auto"],metrics["whole"]
    return dict(severe_good=a["severe_to_good"]==0,whole_severe_good=w["severe_to_good"]==0,
                severe_correct=a["severe_correct"]>=16,good_correct=a["confusion"][0][0]>=19,
                overall=a["correct"]>=38,minor_gain=a["medium_correct"]>=2)


def order(metrics):
    checks=check(metrics)
    a=metrics["auto"]
    safe=sum(checks[k] for k in ("severe_good","whole_severe_good","severe_correct","good_correct","overall"))
    return (all(checks.values()),safe,-a["severe_to_good"],-metrics["whole"]["severe_to_good"],
            a["medium_correct"],a["correct"],a["severe_correct"],a["confusion"][0][0],-metrics["nll"])


def receipt(path,arm,epoch,metrics,public_sha):
    return dict(arm=arm,epoch=epoch,updates=EPOCHS*STEPS,metrics=metrics,threshold=1.,
                checks=check(metrics),passed=all(check(metrics).values()),weights_sha256=grade.file_sha(path/"weights.pt"),
                protocol_sha256=grade.file_sha(OUT/"protocol.json"),public_train_manifest_sha256=public_sha)


def train(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    freeze()
    path=OUT/arm
    if (path/"selection.json").is_file():
        raise ValueError("This arm already finished")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    af_rng=np.random.default_rng(SEED)
    ext_rng=np.random.default_rng(SEED+1)
    data=original.open_rows(("train","val"))
    rows,images,labels=data["train"]
    crops=aligned.load_cache("train",rows)
    val_rows,val_images,val_y=data["val"]
    val_crops=aligned.load_cache("val",val_rows)
    validation=(val_rows,val_images,val_y,val_crops,[i for i,v in enumerate(val_crops) if v is not None])
    by_class=[groups(rows,np.flatnonzero(labels==i)) for i in range(3)]
    external,conditions=auxiliary() if arm=="paired" else ([],[])
    public_sha=grade.file_sha(AUX) if arm=="paired" else None
    model=grade.network().cuda()
    parent=torch.load(aligned.OUT/"weights.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(parent)
    teacher=grade.network().cuda()
    teacher.load_state_dict(parent)
    teacher.eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([dict(params=model.features.parameters(),lr=.0001),
                                 dict(params=model.classifier.parameters(),lr=.00045)],weight_decay=.008)
    history=[]
    best=None
    af_indices_hash=hashlib.sha256()
    ext_indices_hash=hashlib.sha256()
    target=torch.arange(3,device="cuda").repeat_interleave(AF_PER_CLASS)
    for epoch in range(1,EPOCHS+1):
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        sumloss=0.
        for step in range(STEPS):
            af_ids=[pick(gr,af_rng) for gr in by_class for _ in range(AF_PER_CLASS)]
            crop_choice=af_rng.random(len(af_ids))<.5
            af_indices_hash.update(np.asarray(af_ids,dtype="<i4").tobytes()+crop_choice.tobytes())
            squares=[original.augment(crops[i] if use and crops[i] is not None else images[i],af_rng)
                     for i,use in zip(af_ids,crop_choice,strict=True)]
            features=grade.tensor_rgb(squares,"cuda")
            z=model(features)
            with torch.no_grad():
                anchored=teacher(features).softmax(1)[:,2]
            odds=z[:,2]-z[:,:2].logsumexp(1)
            loss=F.cross_entropy(z,target,label_smoothing=.015)+.30*F.binary_cross_entropy_with_logits(odds,anchored)
            if external:
                pairs=[conditions[ext_rng.integers(len(conditions))] for _ in range(EXTERNAL_PAIRS)]
                ids=[pick(gr,ext_rng) for pair in pairs for gr in pair]
                ext_indices_hash.update(np.asarray(ids,dtype="<i4").tobytes())
                ext_x=grade.tensor_rgb([original.augment(external[i],ext_rng) for i in ids],"cuda")
                # Keep the AF dropout sequence identical between the two controlled arms.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.cuda.manual_seed(SEED+epoch*STEPS+step)
                    scores=model(ext_x)[:,:2]
                difference=(scores[:,1]-scores[:,0]).reshape(-1,2)
                loss=loss+.65*F.cross_entropy(scores,torch.arange(2,device="cuda").repeat(len(pairs)))
                loss=loss+.15*F.softplus(.5-difference[:,1]+difference[:,0]).mean()
            progress=((epoch-1)*STEPS+step)/(EPOCHS*STEPS-1)
            for g,base in zip(optimizer.param_groups,(.0001,.00045),strict=True):
                g["lr"]=base*(.1+.9*(1+math.cos(math.pi*progress))/2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            sumloss+=float(loss.detach())
        metrics=evaluate(model,validation)
        row=dict(epoch=epoch,mean_loss=sumloss/STEPS,metrics=metrics,checks=check(metrics))
        history.append(row)
        if best is None or order(metrics)>order(best["metrics"]):
            best=row
            original.atomic(path/"weights.pt",model.state_dict(),tensor=True)
        original.atomic(path/"history.json",history)
        print(arm,epoch,metrics["auto"]["confusion"],"best",best["epoch"],flush=True)
    selected=receipt(path,arm,best["epoch"],best["metrics"],public_sha)
    selected.update(af_draw_sha256=af_indices_hash.hexdigest(),auxiliary_draw_sha256=ext_indices_hash.hexdigest(),
                    auxiliary_pairs=EPOCHS*STEPS*EXTERNAL_PAIRS if external else 0)
    original.atomic(path/"selection.json",selected)


def compare():
    freeze()
    values=[original.read(OUT/arm/"selection.json") for arm in ARMS]
    if values[0]["af_draw_sha256"]!=values[1]["af_draw_sha256"]:
        raise ValueError("AF replay not paired across arms")
    for x in values:
        if x["weights_sha256"]!=grade.file_sha(OUT/x["arm"]/"weights.pt"):
            raise ValueError("Selected weights changed")
    if values[1]["public_train_manifest_sha256"]!=grade.file_sha(AUX):
        raise ValueError("Public train manifest changed")
    best=max(values,key=lambda x:(order(x["metrics"]),x["arm"]=="control"))
    result=dict(candidate=best["arm"] if best["passed"] else None,selected_weight=best["weights_sha256"] if best["passed"] else None,
                passed=best["passed"],control=values[0],paired=values[1],source="57 AF validation photos; no held-out data used")
    original.atomic(OUT/"decision.json",result)
    print(result["candidate"],result["passed"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","control","paired","compare"))
    arg=parser.parse_args().action
    if arg=="freeze":
        freeze()
    elif arg=="compare":
        compare()
    else:
        train(arg)
