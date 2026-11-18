"""A paired experiment using training-area judgements as point supervision on frozen proposals."""
import argparse
import copy
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from class0_patch_refinement import ROOT, read, sha, atomic_json, configure, load_image
from audit_defect_causes_20261112 import ious
from defect_continue_scratch_net import LockedInsect, Deployed
import defect_continue_scratch_specialist as parent
import defect_continue_public_training as replay


OUT = ROOT / "experiments/defect_blind_review_update_v1"
PACK = ROOT / "experiments/ssda_blind_defects_v1"
SEED = 491116
EPOCHS = 12
SOURCE_SHA = "a8ce9e6c9035e1c1c8a2ff6c1907b4d98272e1e33e64d31e3d27c4b43ef8d980"
RESPONSE_CLASS = {"scratch": 0, "insect": 1, "other": -1}


def freeze(export):
    key_file = PACK / "组织者留存/source_key.json"
    key = read(key_file)
    answers = read(export)
    cases = {c["id"]: c for c in key["cases"]}
    lookup = {c["id"]: c["response"] for c in answers["cases"]}
    assert len(cases) == len(lookup) == 34 and cases.keys() == lookup.keys()
    assert answers["case_manifest_sha256"] == key["case_manifest_sha"] == sha(PACK/"盲审材料/cases.json")
    assert key["train_only"] and key["dev_cases"] == 0
    assert sum(bool(v) for v in lookup.values()) == answers["answered"]
    rows = []
    for id_, c in cases.items():
        response = lookup[id_]
        assert response in {"scratch", "insect", "other", "uncertain", ""}
        if c["source"] == "unmatched OOF candidate" and response in RESPONSE_CLASS:
            assert c["max_gt_iou"] < 0.10
            rows.append(dict(case=id_, image=c["image"], support=c["box"], response=RESPONSE_CLASS[response]))
    payload = dict(export_sha256=sha(export), key_sha256=sha(key_file), answered=answers["answered"],
                   total=34, cases=rows, scope="training-only, center semantics; no localization truth")
    destination = OUT / "frozen_review.json"
    if destination.exists() and read(destination) != payload:
        raise ValueError("The experiment is already bound to a different review export")
    atomic_json(destination, payload)
    return payload


def base_network():
    check = read(parent.OUT / "receipt.json")
    if check["weight_sha"] != SOURCE_SHA or sha(parent.OUT/"weights.pt") != SOURCE_SHA:
        raise ValueError("Unexpected scratch checkpoint")
    scratch = parent.specialist()
    scratch.load_state_dict(torch.load(parent.OUT/"weights.pt", map_location="cpu", weights_only=True))
    return Deployed(LockedInsect("cuda").eval(), scratch.eval()).eval().requires_grad_(False)


@torch.no_grad()
def extract(detector, im):
    rgb, (dx, dy, sx, sy) = replay.original.square(im)
    x = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255
    fine, levels = detector.locked.features(x)
    insect = detector.locked.detections(levels)
    scratch = detector.scratch(fine, levels)
    rows = []
    h, w = im.shape[:2]
    for raw, cls in ((insect, 1), (scratch, 0)):
        p = raw[0]
        p = p[p[:,4] >= .001].cpu().numpy()
        p[:,[0,2]] = ((p[:,[0,2]] - dx) / sx).clip(0,w)
        p[:,[1,3]] = ((p[:,[1,3]] - dy) / sy).clip(0,h)
        p = replay.original.nms(p)
        rows.append(p[p[:,5] == cls])
    p = np.concatenate(rows)
    p = p[np.argsort(-p[:,4], kind="stable")]
    t = torch.as_tensor(p, device=x.device)
    b = t[:,:4] * t.new_tensor([sx,sy,sx,sy]) + t.new_tensor([dx,dy,dx,dy])
    wh = (b[:,2:]-b[:,:2]).clamp_min(1)
    context = torch.cat([b[:,:2]-.45*wh, b[:,2:]+.45*wh], 1).clamp(0, replay.original.SIZE)
    # Pool the same frozen features used for detection. Each ROI retains a 2x2 spatial grid.
    regions = torch.cat([b, context])
    pooled = roi_align(levels[0], [regions], 2, spatial_scale=levels[0].shape[-1]/replay.original.SIZE,
                       sampling_ratio=2, aligned=True).flatten(1)
    n = len(b)
    feat = torch.cat([F.layer_norm(pooled[:n], (pooled.shape[-1],)),
                      F.layer_norm(pooled[n:], (pooled.shape[-1],)),
                      (wh/replay.original.SIZE).log(), t[:,4:5], t[:,5:6]], 1)
    return p, feat


class Residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64,2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def shift(self, feats, cls):
        return 2.0 * self.net(feats).gather(1,cls[:,None]).squeeze(1).tanh()

    def forward(self, feats, p):
        if not len(p):
            return p
        prior = p[:,4].clamp(.00001,.99999)
        logit = prior.logit()
        delta = self.shift(feats, p[:,5].long())
        q = p.clone()
        q[:,4] = (p[:,4] + ((logit+delta).sigmoid() - logit.sigmoid())).clamp(0,1)
        return q


def native_targets(p, gt):
    # Borderline extent and duplicate positives are excluded from classification supervision.
    y = np.full(len(p), -1, np.int8)
    if not len(gt):
        y[:] = 0
        return y
    overlap = ious(p[:,:4],gt[:,:4])
    y[overlap.max(1) < .10] = 0
    for i, g in enumerate(gt):
        same = np.flatnonzero((p[:,5] == g[4]) & (overlap[:,i] >= .50))
        if len(same):
            best = same[np.argmax(overlap[same,i])]
            y[best] = 1
        mismatch = (p[:,5] != g[4]) & (overlap[:,i] >= .50)
        y[mismatch & (overlap.argmax(1)==i)] = 0
    return y


def reviewed_region(p, gt, case):
    box = np.asarray(case["support"])
    size = np.maximum(box[2:]-box[:2],1)
    mid = (box[:2]+box[2:])*.5
    inside = ((mid >= p[:,:2]) & (mid <= p[:,2:4])).all(1)
    center = (p[:,:2]+p[:,2:4])*.5
    near = (abs(center-mid)/size).max(1) <= .55
    # Center agreement with a very large unrelated box does not constitute local evidence.
    support = ious(p[:,:4],box[None,:])[:,0]
    clean = (ious(p[:,:4],gt[:,:4]).max(1) < .10) if len(gt) else np.ones(len(p), bool)
    eligible = inside & near & (support >= .15) & clean
    ids, labels = [], []
    for cls in (0,1):
        group = np.flatnonzero(eligible & (p[:,5]==cls))
        if len(group):
            best = group[support[group].argmax()]
            ids.append(int(best))
            labels.append(int(case["response"] == cls))
    return eligible, ids, labels


def prepare(export):
    data = freeze(export)
    rows = replay.original.samples("train")
    val = {r["image_id"] for r in replay.original.samples("dev")}
    assert len(rows) == 596 and not val.intersection([r["image_id"] for r in rows])
    lookup = {r["image"]:r for r in data["cases"]}
    assert len(lookup) == len(data["cases"]) and set(lookup) <= {r["image_id"] for r in rows}
    model = base_network()
    feats, ps, native, protect, weak, frame = [], [], [], [], [], []
    coverage = []
    for i, row in enumerate(rows):
        im = load_image(row["rgb_path"])
        p, z = extract(model, im)
        if i < 3:
            previous = parent.boxes(model, im)
            if not np.array_equal(previous,p):
                raise AssertionError("The frozen candidate stream changed")
            with torch.no_grad():
                t = torch.from_numpy(p).cuda()
                initial = Residual(z.shape[1]).cuda().eval()
                assert torch.equal(initial(z,t),t)
        gt = []
        for g in row["instances"]:
            cx,cy,w,h = g["xywh_normalized"]
            gt.append([(cx-w/2)*im.shape[1],(cy-h/2)*im.shape[0],(cx+w/2)*im.shape[1],(cy+h/2)*im.shape[0],g["class_id"]])
        gt = np.asarray(gt, np.float32).reshape(-1,5)
        y = native_targets(p,gt)
        human = np.full(len(p), -1, np.int8)
        neut = np.zeros(len(p),bool)
        case = lookup.get(row["image_id"])
        if case:
            neut, idx, labels = reviewed_region(p,gt,case)
            human[idx] = labels
            coverage.append(dict(case=case["case"], response=case["response"], near=int(neut.sum()),
                                 proposal_classes=p[idx,5].astype(int).tolist(), labels=labels,
                                 old_scores=p[idx,4].tolist()))
        feats.append(z.cpu().half())
        ps.append(torch.from_numpy(p.copy()))
        native.append(torch.from_numpy(y))
        protect.append(torch.from_numpy(neut))
        weak.append(torch.from_numpy(human))
        frame.append(torch.full((len(p),),i))
    tensors={k:torch.cat(v) for k,v in (("feats",feats),("p",ps),("y",native),
                                        ("protect",protect),("weak",weak),("frame",frame))}
    tensors["export_sha256"]=data["export_sha256"]
    replay.atomic_latest(OUT/"train_features.pt",tensors)
    atomic_json(OUT/"coverage.json",dict(coverage=coverage, rows=len(tensors["p"]), train_images=len(rows),
                original_boxes=sum(len(r["instances"]) for r in rows),
                native_positive=int((tensors["y"]==1).sum()), native_negative=int((tensors["y"]==0).sum()),
                cases_reaching_model=sum(bool(r["near"]) for r in coverage),
                covered_positives=sum(sum(r["labels"]) for r in coverage), zero_output_preserved=True))
    print("train features",len(tensors["p"]), "review candidates",[(r["response"],r["labels"]) for r in coverage],flush=True)


def fit():
    data = torch.load(OUT/"train_features.pt",map_location="cpu",weights_only=True)
    frozen = read(OUT/"frozen_review.json")
    assert data.pop("export_sha256") == frozen["export_sha256"]
    train = {k:v.cuda() for k,v in data.items()}
    feats, p = train["feats"].float(), train["p"]
    cls, prior = p[:,5].long(), p[:,4].clamp(1e-5,1-1e-5).logit()
    torch.manual_seed(SEED)
    init = Residual(feats.shape[1]).cuda()
    options = {}
    for mode in ("control","review"):
        m=copy.deepcopy(init)
        y = train["y"].clone()
        if mode=="review":
            y[train["protect"]]=-1
        options[mode] = (m,torch.optim.AdamW(m.parameters(),lr=.00035,weight_decay=.01),y,[])
    rng = np.random.default_rng(SEED)
    audit_ids=torch.where(train["weak"]>=0)[0]
    if not (train["weak"][audit_ids]==1).any() or not (train["weak"][audit_ids]==0).any():
        raise ValueError("No support for the planned review contrast")
    steps=0
    for epoch in range(EPOCHS):
        perm = rng.permutation(596)
        for chunk in np.array_split(perm,20):
            idx = torch.where(torch.isin(train["frame"],torch.tensor(chunk,device="cuda")))[0]
            for mode,(m,opt,y,hist) in options.items():
                delta = m.shift(feats[idx],cls[idx])
                score = prior[idx]+delta
                nativeloss = []
                for c in (0,1):
                    for target in (0,1):
                        mask = (y[idx]==target) & (cls[idx]==c)
                        if mask.any():
                            observed=F.binary_cross_entropy_with_logits(score[mask],torch.full_like(score[mask],float(target)),reduction="none")
                            if not target:
                                observed=observed.topk(min(len(observed),max(8,2*len(chunk)))).values
                            nativeloss.append(observed.mean())
                loss=torch.stack(nativeloss).mean()+.12*delta.square().mean()
                if mode=="review":
                    # Each protected site contributes at most one class candidate per source branch.
                    wh=train["weak"][audit_ids].float()
                    wd=m.shift(feats[audit_ids],cls[audit_ids])
                    auxiliary=F.binary_cross_entropy_with_logits(prior[audit_ids]+wd,wh,reduction="none")
                    loss=loss+.20*(auxiliary[wh==0].mean()+auxiliary[wh==1].mean())*.5
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite review loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),2.,error_if_nonfinite=True)
                opt.step()
                hist.append(float(loss.detach()))
            steps+=1
    for mode,(m,_,_,hist) in options.items():
        output=OUT/mode
        output.mkdir(parents=True,exist_ok=True)
        replay.atomic_latest(output/"weights.pt",m.cpu().state_dict())
        atomic_json(output/"receipt.json",dict(mode=mode,weight_sha=sha(output/"weights.pt"),epochs=EPOCHS,
                    updates=steps,train_frames=596,train_candidate_rows=len(p),feature_dim=feats.shape[1],
                    thresholds=[.40,.34],export_sha256=frozen["export_sha256"],scratch_parent_sha=SOURCE_SHA,
                    cache_sha256=sha(OUT/"train_features.pt"),epoch_loss=np.mean(np.array(hist).reshape(EPOCHS,-1),1).tolist()))
    print("paired fit",steps,flush=True)


def evaluate(mode):
    output=OUT/mode
    receipt=read(output/"receipt.json")
    assert receipt["weight_sha"]==sha(output/"weights.pt") and receipt["scratch_parent_sha"]==SOURCE_SHA

    def deploy():
        head=Residual(receipt["feature_dim"]).cuda().eval()
        head.load_state_dict(torch.load(output/"weights.pt",map_location="cpu",weights_only=True))
        return base_network(),head.requires_grad_(False)

    @torch.inference_mode()
    def boxes(model,im):
        detector,head=model
        p,z=extract(detector,im)
        updated=head(z,torch.from_numpy(p).cuda()).cpu().numpy()
        return updated[np.argsort(-updated[:,4],kind="stable")]
    original=replay.original
    old=original.deploy_model,original.boxes
    try:
        original.deploy_model,original.boxes=deploy,boxes
        with replay.evaluation_output(output):
            original.evaluate()
    finally:
        original.deploy_model,original.boxes=old


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","fit","control","review"])
    parser.add_argument("--export",type=Path)
    args=parser.parse_args()
    configure(OUT,"cuda")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    with replay.exclusive_run(OUT):
        if args.action=="prepare":
            prepare(args.export)
        elif args.action=="fit":
            fit()
        else:
            evaluate(args.action)
