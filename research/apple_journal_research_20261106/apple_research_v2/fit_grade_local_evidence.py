"""Train the protected mild-defect head from grouped AFruitDB and paired public photos."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import afruit_grade_crop_alignment as crop
import afruit_grade_matched as matched
import afruit_grade_model as grade
import afruit_grade_train as base
from grade_local_evidence import LocalEvidenceGrade


def training_weights(metadata, include_public):
    # Equalise labels, capturing setups and group contributions within each source.
    weights = np.zeros(len(metadata), np.float64)
    fractions = {"af_matched": .46, "af_other": .26, "public": .28}
    if not include_public:
        fractions.pop("public")
    for domain, budget in fractions.items():
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, (source, condition, gid, label, view_weight) in enumerate(metadata):
            if domain == source:
                groups[label][condition][gid].append((i,view_weight))
        if set(groups) != {0,1}:
            raise ValueError(f"Both good and mild need support in {domain}")
        for conditions in groups.values():
            for captures in conditions.values():
                per_group = budget / (2*len(conditions)*len(captures))
                for items in captures.values():
                    denom = sum(w for _,w in items)
                    for i,w in items:
                        weights[i] = per_group*w/denom
    return weights / weights.sum()


@torch.inference_mode()
def extract(model, images, device):
    vectors=[]
    for start in range(0,len(images),32):
        feats, _ = model.features_and_grade(grade.tensor_rgb(images[start:start+32],device))
        vectors.append(feats.cpu().numpy())
    return np.concatenate(vectors)


def data(aux, use_public):
    rows, orig, _ = base.open_rows(("train",))["train"]
    cropped = crop.load_cache("train",rows)
    images, meta = [],[]
    def add(im, domain, condition, gid, label, weight):
        for sample in (im,np.ascontiguousarray(im[:,::-1])):
            images.append(sample)
            meta.append((domain,condition,gid,int(label),weight))
    for r, im, region in zip(rows,orig,cropped,strict=True):
        if r["grade"] not in (0,1):
            continue
        group = r["group_id"]
        domain = "af_matched" if group=="burst_0100" else "af_other"
        add(im,domain,"AF",group,r["grade"],1)
        if region is not None:
            add(region,domain,"AF",group,r["grade"],2)
    if use_public:
        # Reuse the source validation, hash checks, crop and RGB conventions of the primary trial.
        matched.AUX = aux
        published, _ = matched.auxiliary()
        source_rows=[r for r in base.read(aux)["samples"] if r["split"]=="train"]
        conditions = defaultdict(set)
        for r in source_rows:
            conditions[(r.get("source",""),r["condition_id"])].add(int(r["grade"]))
        for r,im in zip(source_rows,published,strict=True):
            condition=(r.get("source",""),r["condition_id"])
            if conditions[condition]=={0,1}:
                add(im,"public",condition,(r.get("source",""),r["group_id"]),r["grade"],1)
    return images,meta


def run(args):
    dest=args.out.resolve()
    dest.mkdir(exist_ok=True,parents=True)
    spec=dict(architecture="mobilenet_v3_small_local_evidence_v1", start_sha256=grade.file_sha(crop.OUT/"weights.pt"),
              data_manifest=grade.file_sha(base.MANIFEST), grade_rule="original non-good argmax preserved; positive local log-odds can move good to mild",
              source="All AFruitDB train good/mild, grouped, upweight same capture; real auto crops and whole images plus mirrors. Matched public GOOD/MILD proxies only.",
              local="mean and maximum of 4x4 pooled middle-block grids, one regularized logistic discriminator, weighted training-set feature scaling",
              c=.12, effective_weight=96, domain_weights=[.46,.26,.28],
              val_gate="same auto validation as paired training; fixed binary boundary and original 3-class argmax; no threshold scan",
              code=grade.file_sha(__file__), runtime_code=grade.file_sha(Path(__file__).with_name("grade_local_evidence.py")))
    if (dest/"protocol.json").exists() and base.read(dest/"protocol.json")!=spec:
        raise ValueError("Local trial protocol changed")
    base.atomic(dest/"protocol.json",spec)
    if args.action=="freeze":
        return
    out=dest/args.action
    if (out/"selection.json").exists():
        raise ValueError("A completed arm already exists")
    out.mkdir(exist_ok=True)
    device=args.device
    model=LocalEvidenceGrade().to(device).eval()
    model.parent.load_state_dict(torch.load(crop.OUT/"weights.pt",weights_only=True,map_location=device))
    add_public=args.action=="paired"
    images,meta=data(args.aux,add_public)
    feat=extract(model,images,device).astype(np.float64)
    w=training_weights(meta,add_public)
    target=np.array([r[3] for r in meta])
    center=np.average(feat,axis=0,weights=w)
    scale=np.maximum(.025,np.sqrt(np.average((feat-center)**2,axis=0,weights=w)))
    clf=LogisticRegression(C=spec["c"],max_iter=1200,solver="lbfgs")
    clf.fit((feat-center)/scale,target,sample_weight=w*spec["effective_weight"])
    linear=clf.coef_[0]/scale
    with torch.no_grad():
        model.local.weight.copy_(torch.tensor(linear[None,:],dtype=torch.float32))
        model.local.bias.fill_(float(clf.intercept_[0]-linear@center))
    # Evaluate the exact frozen inference module, not the sklearn fitting surrogate.
    inputs,ims,gt=base.open_rows(("val",))["val"]
    auto=crop.load_cache("val",inputs)
    ids=[i for i,im in enumerate(auto) if im is not None]
    model.to("cuda")
    metrics=matched.evaluate(model,(inputs,ims,gt,auto,ids))
    base.atomic(out/"weights.pt",model.state_dict(),tensor=True)
    receipt=dict(architecture=spec["architecture"],weights_sha256=grade.file_sha(out/"weights.pt"),threshold=1.,
                 protocol_sha256=grade.file_sha(dest/"protocol.json"),public_manifest=grade.file_sha(args.aux) if add_public else None,
                 arm=args.action,metrics=metrics,checks=matched.check(metrics),feature_views=len(meta))
    receipt["passed"]=all(receipt["checks"].values())
    base.atomic(out/"selection.json",receipt)
    print(args.action,receipt["passed"],metrics["auto"]["confusion"],metrics["whole"]["confusion"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["freeze","control","paired"])
    parser.add_argument("--aux",type=Path,default=matched.AUX)
    parser.add_argument("--out",type=Path,default=grade.ROOT/"experiments/grade_local_evidence_v1")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    run(parser.parse_args())
