"""Single-variable training contrast using grades observed under shared capture conditions."""
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import afruit_grade_crop_alignment as align
import afruit_grade_model as grade
import afruit_grade_train as io
import select_grade_repair as development
from defect9485_repr import WEIGHTS, META_SHA
from grade_capture_contrast import ARCH, CaptureContrastGrade


OUT=grade.ROOT/"experiments/grade_pairs_20261119/capture_contrast"
WOO=grade.ROOT/"experiments/afruit_grade_matched_v1/paired"
SOURCES=grade.ROOT/"datasets/grade_guard_sources_v2/manifest.json"
WOO_DATA=grade.ROOT/"datasets/apple_matched_quality_v1/manifest.json"
SEED=61119


def training_set():
    rows,squares,_=io.open_rows(("train",))["train"]
    crops=align.load_cache("train",rows)
    images=[]; labels=[]
    def add(square,record,source,group):
        images.append(square)
        labels.append((source,group,record["grade"],record["sha256"]))
    for r,whole,fruit in zip(rows,squares,crops,strict=True):
        add(whole,r,"AF",r["group_id"])
        if fruit is not None:
            add(fruit,r,"AF",r["group_id"])
    for manifest in (WOO_DATA,SOURCES):
        for r in io.read(manifest)["samples"]:
            if r["split"]!="train":continue
            source="Woo" if manifest==WOO_DATA else r["source"]
            if "hdf" in source.lower() or "plos" in source.lower():raise ValueError(source)
            group=r["condition_id"] if manifest==WOO_DATA else r["group_id"]
            frame=align.read_frame(r)
            add(grade.rgb_square(frame),r,source,group)
    return images,labels


def features(model,images):
    original=[];appearance=[]
    for i in range(0,len(images),12):
        x=grade.tensor_rgb(images[i:i+12],"cuda")
        z,e=model.encode(x)
        original.append(z);appearance.append(e)
    return torch.cat(original),torch.cat(appearance)


def tables(panel,pred):
    result={}; start=0
    for key,(rows,images) in panel.items():
        n=sum(img is not None for img in images)
        it=iter(pred[start:start+n])
        result[key]=development.counts(rows,[next(it) if img is not None else None for img in images])
        start+=n
    assert start==len(pred)
    return result


def ranking(t):
    a,h=t["auto"],t["whole"]
    safety=[a["good"]>=20,a["severe_good"]==0,a["severe"]>=16,h["severe_good"]==0]
    return (all(safety) and a["mild"]>=6,sum(safety),a["mild"],a["good"],a["severe"],h["mild"],h["good"])


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if (OUT/"result.json").exists():raise FileExistsError("Completed contrast already exists")
    dino_sha=grade.file_sha(WEIGHTS);woo_sha=grade.file_sha(WOO/"weights.pt")
    assert dino_sha==META_SHA and woo_sha==io.read(WOO/"selection.json")["weights_sha256"]
    io.atomic(OUT/"protocol.json",dict(seed=SEED,steps=400,checks_every=40,adamw_lr=.00035,dino=dino_sha,woo=woo_sha,
        input_manifests=[grade.file_sha(p) for p in (io.MANIFEST,WOO_DATA,SOURCES)],fitting="Only train. Both arms see identical source/class/group-balanced rows and identical pairs. Differ solely by coefficient of ordinal pair loss 0 or 0.65. Pair different labels within AF burst, Woo station or FruitLife specimen. Woo weak scuff and FruitLife storage stages retained as proxies.",
        ranking="all development safety + auto mild at least 6; then safety count, auto mild, good, severe, whole mild, good; ties keep earlier checkpoint",
        parameters="Only final small multilayer head learns. Freeze Woo and DINOv2 encoders. CPU FP32 DINO, CUDA FP16 autocast DINO. Use native 224 preprocessing and fixed argmax.",
        code=grade.file_sha(__file__),model_code=grade.file_sha(Path(__file__).with_name("grade_capture_contrast.py"))))
    torch.manual_seed(SEED)
    net=CaptureContrastGrade().cuda().eval()
    net.appearance.load_state_dict(torch.load(WEIGHTS,map_location="cpu",weights_only=True))
    net.woo.load_state_dict(torch.load(WOO/"weights.pt",map_location="cpu",weights_only=True))
    net.woo.requires_grad_(False);net.appearance.requires_grad_(False)
    imgs,meta=training_set();z,e=features(net,imgs);del imgs
    panel=development.input_panels()
    zv,ev=features(net,[im for _,ims in panel.values() for im in ims if im is not None])
    by_source_grade=defaultdict(lambda:defaultdict(list))
    paired=defaultdict(lambda:defaultdict(list))
    for ix,(source,group,g,sha) in enumerate(meta):
        by_source_grade[source,g][group].append(ix)
        paired[source,group][g].append(ix)
    cells={k:list(groups.values()) for k,groups in by_source_grade.items()}
    choices=[]
    for tag,gs in paired.items():
        for a in sorted(gs):
            for b in sorted(gs):
                if a<b:choices.append((tag,a,b,gs[a],gs[b]))
    paired_counts=[dict(source=tag[0],group=tag[1],lower=a,upper=b,
          lower_images=len({meta[i][3] for i in L}),upper_images=len({meta[i][3] for i in H})) for tag,a,b,L,H in choices]
    io.atomic(OUT/"inputs.json",dict(unique_training_images=len({r[3] for r in meta}),n_training_views=len(meta),pair_cells=paired_counts))
    initial={k:v.cpu().clone() for k,v in net.adjust.state_dict().items()}
    y=torch.tensor([m[2] for m in meta],device="cuda")
    for arm,coef in (("absolute",0.),("paired",.65)):
        torch.manual_seed(SEED)
        net.adjust.load_state_dict(initial);rng=np.random.default_rng(SEED)
        opt=torch.optim.AdamW(net.adjust.parameters(),lr=.00035,weight_decay=.015)
        best=None;history=[]
        for step in range(1,401):
            net.adjust.train()
            absolute=[]
            for (src,_),groups in cells.items():
                for _ in range(8 if src=="AF" else 4):
                    absolute.append(int(rng.choice(groups[rng.integers(len(groups))])))
            lower=[];upper=[];grade_lo=[];grade_hi=[]
            for _,a,b,L,H in choices:
                for _ in range(2):
                    lower.append(int(rng.choice(L)));upper.append(int(rng.choice(H)))
                    grade_lo.append(a);grade_hi.append(b)
            ix=absolute+lower+upper
            out=net.combine(z[ix],e[ix])
            logits,low,high=out.split((len(absolute),len(lower),len(upper)))
            lo=torch.tensor(grade_lo,device="cuda")[:,None];hi=torch.tensor(grade_hi,device="cuda")[:,None]
            delta=(high.gather(1,hi)-high.gather(1,lo))-(low.gather(1,hi)-low.gather(1,lo))
            loss=F.cross_entropy(logits,y[absolute],label_smoothing=.04)+coef*F.softplus(1.5-delta).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%40:continue
            net.adjust.eval()
            with torch.no_grad():
                t=tables(panel,net.combine(zv,ev).argmax(1).tolist())
            order=ranking(t)
            entry=dict(step=step,metrics=t,rank=list(order))
            history.append(entry)
            if best is None or order>best:
                best=order;chosen=entry
                io.atomic(OUT/arm/"weights.pt",net.state_dict(),tensor=True)
        io.atomic(OUT/arm/"history.json",history)
        receipt=dict(architecture=ARCH,threshold=1.,input_side=224,weights_sha256=grade.file_sha(OUT/arm/"weights.pt"),
             step=chosen["step"],metrics=chosen["metrics"],passed=bool(best[0]),protocol_sha256=grade.file_sha(OUT/"protocol.json"))
        io.atomic(OUT/arm/"selection.json",receipt)
        print(arm,chosen["step"],chosen["metrics"]["auto"]["confusion"],receipt["passed"],flush=True)
    io.atomic(OUT/"result.json",{a:io.read(OUT/a/"selection.json") for a in ("absolute","paired")})


if __name__=="__main__":
    main()
