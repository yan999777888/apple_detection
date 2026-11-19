"""Paired real-scratch vs identical-support normal-skin supervision on native320."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import afruit_grade_model as grade
import afruit_grade_train as io
import afruit_grade_crop_alignment as align
from grade_detail_patch import square,pack,ARCH
from matched_grade_loader import load
import prepare_grade_transplant_20261119 as edits
import select_grade_repair as sel


OUT=edits.OUT
PARENT=grade.ROOT/"experiments/grade_error49_v1/border_native320_woo_ref"
WHITE=grade.ROOT/"datasets/harvard_grade_severe_20261119/manifest.json"
SEED=919027


def mask_for(polygons,fruit):
    x,y,r,b=fruit;s=320/max(r-x,b-y)
    sw,sh=max(1,round((r-x)*s)),max(1,round((b-y)*s))
    mask=np.zeros((320,320),np.uint8)
    for poly in polygons:
        xy=(np.array(poly,np.float64).reshape(-1,2)-[x,y])*[sw/(r-x),sh/(b-y)]
        xy+=[(320-sw)//2,(320-sh)//2]
        cv2.fillPoly(mask,[np.rint(xy).astype(np.int32)],1)
    out=mask.reshape(80,4,80,4).mean((1,3))
    return out>min(.12,.9*out.max()) if out.any() else out>0


def polygon_rect(bounds):
    x,y,r,b=bounds
    return [[[x,y],[r,y],[r,b],[x,b]]]


def real_set(subset):
    rows=io.read(edits.REGIONS)["regions"]
    if subset=="studio":
        rows=[]
        for r in io.read(WHITE)["local_pairs"]:
            for k,label in (("positive",1),("control",0)):
                rows.append(dict(path=r["source_path"],image_sha256=r["sha256"],source="Harvard_studio",split=subset,
                    parent_fruit_xyxy=r["fruit_crop_xyxy"],target="surface_defect" if label else "confirmed_normal",
                    source_image_xyxy=r[k+"_xyxy_raw"],id=str(r["number"])+"_"+k,group_id=r["group_id"]))
    imgs=[];small=[];masks=[];info=[];prev=""
    for z in sorted(rows,key=lambda a:a["path"]):
        if z["split"]!=subset or z["target"] not in ("surface_defect","confirmed_normal"):continue
        if not z["parent_fruit_xyxy"]:continue
        if z["path"]!=prev:
            frame=align.read_frame(dict(path=z["path"],sha256=z["image_sha256"]));prev=z["path"]
        x,y,r,b=map(round,z["parent_fruit_xyxy"])
        im=frame[y:b,x:r]
        m=mask_for(z.get("author_polygon_xy") or polygon_rect(z["source_image_xyxy"]),(x,y,r,b))
        if not m.any():continue
        imgs.append(square(im,320));small.append(grade.rgb_square(im));masks.append(m)
        info.append(dict(id=z["id"],path=z["path"],source=z["source"],group=z["group_id"],positive=z["target"]=="surface_defect"))
    return small,imgs,np.array(masks),info


@torch.no_grad()
def encode(net,low,hi):
    blocks=[]
    for i in range(0,len(hi),12):
        field,_=net.encode(pack(low[i:i+12],hi[i:i+12],"cuda"))
        blocks.append(field.half())
    return torch.cat(blocks)


def regional(net,features,masks):
    heat,_=net.evidence(features.float())
    a=heat.masked_fill(~masks,-1e4).flatten(1).topk(3,dim=1).values
    valid=a>-1e3
    return (a*valid).sum(1)/valid.sum(1).clamp_min(1)


def image_sets(model):
    data={}
    for split in ("train","val","studio"):
        low,hi,mask,rows=real_set(split)
        data[split]=encode(model,low,hi),torch.from_numpy(mask).cuda(),rows
        print(split,len(rows),Counter(r["positive"] for r in rows),flush=True)
    available=edits.hosts();discard={6,11};hosts=[h for i,h in enumerate(available) if i not in discard]
    excluded={"4e3c60b690b9838324ae","0d00a03cd59410f2f038","ca9f52537775e1a4633c"}
    donors=[d for d in edits.donor_pairs() if d["id"] not in excluded]
    if len(hosts)!=13 or len(donors)!=8:raise ValueError("Manually reviewed set changed")
    hi1=[];hi0=[];mask=[];pairs=[]
    for h in hosts:
        for d in donors:
            pos,sham,zone=edits.edit(h,d)
            if not zone.any():raise ValueError("Vanishing projected mark")
            hi1.append(pos);hi0.append(sham);mask.append(zone)
            pairs.append(dict(host=h["path"],xy=h["xy"],donor=d["id"],normal_donor=d["normal_id"],donor_capture=d["capture"]))
    # The 224 branch never enters any edit loss; preserve the same unmodified parent
    # frame in both passes, so only the 320 local field can vary.
    dummy=[np.zeros((224,224,3),np.uint8)]*len(hi1)
    data["edited"]=encode(model,dummy,hi1),encode(model,dummy,hi0),torch.from_numpy(np.stack(mask)).cuda()
    io.atomic(OUT/"training_pair_ledger.json",dict(pairs=pairs,hosts=len(hosts),donors=len(donors),
        unique_donor_timestamps=len({d["capture"] for d in donors}),source="train only; identical geometry, alpha, host and placement",
        host_groups=sorted({h["group"] for h in hosts}),area320=[float(d["area"]) for d in donors]))
    return data


@torch.no_grad()
def assess(net,dataset):
    features,masks,rows=dataset;values=[]
    for i in range(0,len(rows),12):
        values.extend(regional(net,features[i:i+12],masks[i:i+12]).tolist())
    by=defaultdict(lambda:defaultdict(list))
    for r,z in zip(rows,values,strict=True):by[r["path"]][r["positive"]].append(z)
    rank=[float(g>n)+.5*float(g==n) for sides in by.values() for g in sides[True] for n in sides[False]]
    return dict(n=len(rows),positive=sum(r["positive"] for r in rows),normal=sum(not r["positive"] for r in rows),
                tp=sum(z>0 and r["positive"] for r,z in zip(rows,values,strict=True)),
                fp=sum(z>0 and not r["positive"] for r,z in zip(rows,values,strict=True)),
                paired_wins=sum(rank),paired_comparisons=len(rank),rows=[dict(r,score=z) for r,z in zip(rows,values,strict=True)])


def train():
    if (OUT/"local_comparison.json").is_file():raise FileExistsError("Completed local mechanism comparison")
    torch.manual_seed(SEED);cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    prior=grade.file_sha(PARENT/"weights.pt")
    if prior!="b77cecb5695dcc6296e7d0c86b8aa3683fdc2dc5dcee6dbf2e0143bcb66b769a":raise ValueError("Wrong native320 parent")
    spec=dict(parent=prior,seed=SEED,updates_each=120,checkpoints="last only",
        learning="AdamW .00025; frozen parent, train native320 local verifier alone. Same real supervised items, placebos, tensors, step count and RNG draws in both arms; one adds an intra-footprint true/sham ranking loss .65.",
        local_gate="On the separately marked real AF and PSU dev sites, TP, normal FP and same-photo paired wins must all be noninferior with at least one strict gain. On the three single-session white-studio pairs, at least two of three correct and no decrease versus control. Fixed local logit zero used for both. Only on pass evaluate both final frozen checkpoints on the AF57 graded development set.",
        synthesis="Actual author Woo scratch residual relative to inpainted surroundings versus identical-shaped normal residual from visually verified healthy skin on that same donor image, same scale/gain/location/feather. Only score within the altered region; no fabricated image grade.",
        quality_review="Exclude hosts 6/11 for ambiguous pre-existing marks. Exclude two pit-like donor appearances and smallest nearly invisible transfer. Fixed before optimization, using training-only visual montages.",
        code=grade.file_sha(__file__),builder=grade.file_sha(edits.__file__),region_source=grade.file_sha(edits.REGIONS),
        white=grade.file_sha(WHITE),eligibility=grade.file_sha(OUT/"eligibility.json"))
    io.atomic(OUT/"protocol.json",spec)
    net,_,_=load(PARENT,"cuda");net.verifier.requires_grad_(True)
    data=image_sets(net);field,mask,records=data["train"];ta,tb,zone=data["edited"]
    initial={k:t.detach().cpu().clone() for k,t in net.verifier.state_dict().items()}
    by=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(records):by[r["positive"]][r["path"]].append(i)
    buckets={k:list(v.values()) for k,v in by.items()}
    verdict={}
    for arm,factor in (("real_only",0.),("paired_sham",.65)):
        rng=np.random.default_rng(SEED);net.verifier.load_state_dict(initial)
        opt=torch.optim.AdamW(net.verifier.parameters(),lr=.00025,weight_decay=.012)
        draw_hash=hashlib.sha256()
        for step in range(1,121):
            ids=[]
            for label in (True,False):
                groups=buckets[label]
                for _ in range(8):
                    group=groups[rng.integers(len(groups))];ids.append(group[rng.integers(len(group))])
            ed=rng.integers(len(ta),size=8);draw_hash.update(np.asarray(ids+ed.tolist(),np.int64).tobytes())
            z=regional(net,field[ids],mask[ids])
            p=regional(net,ta[ed],zone[ed]);n=regional(net,tb[ed],zone[ed])
            loc=F.softplus(.35-z[:8]).mean()+F.softplus(.45+z[8:]).mean()+.25*F.softplus(.8-z[:8,None]+z[None,8:]).mean()
            loss=loc+factor*F.softplus(.85-(p-n)).mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.verifier.parameters(),3,error_if_nonfinite=True);opt.step()
        path=OUT/arm;io.atomic(path/"weights.pt",net.state_dict(),tensor=True)
        a=assess(net,data["val"]);s=assess(net,data["studio"])
        verdict[arm]=dict(local=a,studio=s,order=draw_hash.hexdigest(),weight=grade.file_sha(path/"weights.pt"))
        io.atomic(path/"selection.json",dict(input_side=320,architecture=ARCH,threshold=1.0,step=120,
                 weights_sha256=verdict[arm]["weight"],protocol_sha256=grade.file_sha(OUT/"protocol.json")))
        print(arm,"real",a["tp"],a["fp"],a["paired_wins"],a["paired_comparisons"],"studio",s["paired_wins"],s["paired_comparisons"],flush=True)
    c,s=verdict["real_only"],verdict["paired_sham"]
    if c["order"]!=s["order"]:raise ValueError("Different draws across arms")
    gain=[s["local"]["tp"]-c["local"]["tp"],c["local"]["fp"]-s["local"]["fp"],
          s["local"]["paired_wins"]-c["local"]["paired_wins"]]
    ok=min(gain)>=0 and max(gain)>0 and s["studio"]["paired_wins"]>=max(2,c["studio"]["paired_wins"])
    io.atomic(OUT/"local_comparison.json",dict(passed=ok,gains=gain,arms=verdict))
    if ok:
        panels=sel.input_panels(detail=True)
        both={a:sel.evaluate(OUT/a,panels,"cuda") for a in verdict}
        io.atomic(OUT/"conditional_development.json",both)
    print("Transfer gate",ok,gain,flush=True)


if __name__=="__main__":train()
