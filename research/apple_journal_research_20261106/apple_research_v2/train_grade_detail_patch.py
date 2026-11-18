"""One fixed native-resolution apple surface experiment, using training-only confirmed zones."""
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import afruit_grade_crop_alignment as align
import afruit_grade_model as grade
import afruit_grade_train as io
from afruit_visible_quality import prepare_input
from grade_detail_patch import ARCH,NativePatchGrade,pack,square
from matched_grade_loader import load
import select_grade_repair as select
import train_grade_spatial_repair as spatial
from prepare_matched_woo_quality import ZIP,SHA


ROOT=grade.ROOT
OUT=ROOT/"experiments/grade_error49_v1/border_native320"
PARENT=ROOT/"experiments/grade_guard_sources_v2/border"
MARKS=ROOT/"datasets/afruit_train_spots_v1/manifest.json"
SEED=762094


def project(polygons,physical):
    x,y,r,b=physical
    scale=320/max(r-x,b-y)
    width,height=max(1,round((r-x)*scale)),max(1,round((b-y)*scale))
    result=np.zeros((320,320),np.uint8)
    for poly in polygons:
        vertices=np.array(poly,np.float64).reshape(-1,2)
        coords=(vertices-[x,y])*[width/(r-x),height/(b-y)]+[(320-width)//2,(320-height)//2]
        cv2.fillPoly(result,[np.rint(coords).astype(np.int32)],1)
    return result.reshape(80,4,80,4).max((1,3)).astype(bool)


def frame_background(rgb,good):
    # Only neutral background connected to the frame edge or explicit letterboxing.
    # On a clean screened face both sides of its normal interface are negative;
    # on damaged fruit the band on the skin is never assigned a negative target.
    a=rgb.astype(np.int16)
    lo=a.min(2);hi=a.max(2)
    padding=(rgb==[124,116,104]).all(2)
    neutral=(lo>190)&(hi-lo<19)
    _,cc=cv2.connectedComponents((neutral|padding).astype(np.uint8))
    ids=np.unique(np.r_[cc[0],cc[-1],cc[:,0],cc[:,-1]])
    connected=np.isin(cc,ids[ids!=0]).astype(np.uint8)
    kernel=np.ones((7,7),np.uint8)
    if good:
        mask=cv2.morphologyEx(connected,cv2.MORPH_GRADIENT,kernel)>0
        mask|=padding
    else:
        mask=cv2.erode(connected,kernel).astype(bool)
    return mask.reshape(80,4,80,4).all((1,3))


def training():
    rows=align.split_rows("train")
    truth={r["path"]:r for r in rows}
    positions=io.read(MARKS)["samples"]
    coords=io.read(align.OUT/"train_regions.json")
    region={r["sha256"]:r for r in coords}
    if len(positions)!=44 or len(region)!=len(rows):
        raise ValueError("Unexpected training inputs")
    small,detail,masks,meta=[],[],[],[]
    def add(frame,physical,pos,neg,row,source):
        hi=square(frame,320)
        p,n=project(pos,physical),project(neg,physical)
        bg=frame_background(hi,row["grade"]==0)
        # Ignore uncertain areas; labelled damage always wins over background.
        mask=np.full((80,80),-1,np.int8);mask[bg]=0;mask[n]=2;mask[p]=1
        small.append(grade.rgb_square(frame));detail.append(hi);masks.append(mask)
        meta.append(dict(key=row["path"],label=row["grade"],source=source,positive=int(p.sum()),
                         hard_normal=int(n.sum()),background=int(bg.sum())))
    for a in positions:
        row=truth[a["path"]];reg=region[row["sha256"]]
        if a["split"]!="train" or a["sha256"]!=row["sha256"] or a["crop_xyxy_1280"]!=reg["crop"]:
            raise ValueError("Wrong training crop or split")
        image=align.read_frame(row);sh,sw=image.shape[:2]
        if [sh,sw]!=a["source_hw"]:raise ValueError("Training annotation geometry changed")
        pos=spatial.rectangles(a["positives_xyxy"]);neg=spatial.rectangles(a["negatives_xyxy"])
        add(image,(0,0,sw,sh),pos,neg,row,"AF")
        if reg["crop"] is not None:
            frame=prepare_input(image,1280);h,w=frame.shape[:2]
            if list((h,w))!=reg["hw"]:raise ValueError("AF detector input changed")
            x,y,r,b=reg["crop"]
            add(frame[y:b,x:r],(x*sw/w,y*sh/h,r*sw/w,b*sh/h),pos,neg,row,"AF")
    if grade.file_sha(ZIP)!=SHA:raise ValueError("Woo archive changed")
    marks=spatial.woo_marks()
    for row in io.read(spatial.WOO)["samples"]:
        if row["split"]!="train":continue
        poly=marks[row["source_member"]]
        if bool(poly)!=bool(row["grade"]):raise ValueError("Woo labels changed")
        image=spatial.picture(row)
        x,y,r,b=row["source_box"]
        if image.shape[:2]!=(b-y,r-x):raise ValueError("Woo crop must retain native resolution")
        add(image,(x,y,r,b),poly,[],row,"Woo")
    return small,detail,np.stack(masks),meta


@torch.no_grad()
def features(net,small,hi):
    f,z=[],[]
    for j in range(0,len(small),16):
        x=pack(small[j:j+16],hi[j:j+16],"cuda")
        field,old=net.encode(x)
        f.append(field.half());z.append(old)
    return torch.cat(f),torch.cat(z)


def masked_peak(heat,mask):
    scores=heat.masked_fill(~mask,-1e4).flatten(1).topk(5,dim=1).values
    keep=scores>-9990
    return (scores*keep).sum(1)/keep.sum(1).clamp_min(1),keep.any(1)


def tables(pred,panels):
    out={}
    offset=0
    for name,items in panels.items():
        rows,small=items[:2];length=sum(s is not None for s in small)
        actual=iter(pred[offset:offset+length])
        labels=[next(actual) if s is not None else None for s in small]
        out[name]=select.counts(rows,labels)
        offset+=length
    if offset!=len(pred):raise ValueError("Validation indexing changed")
    return out


def run(out=OUT,checkpoint_observer=None):
    if (out/"selection.json").exists():raise FileExistsError("Completed native trial")
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED);cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(SEED)
    model=NativePatchGrade().cuda()
    parent,cut,side=load(PARENT,"cuda")
    if cut!=1 or side!=224:raise ValueError("Wrong border parent")
    model.parent.load_state_dict(parent.state_dict());model.parent.eval().requires_grad_(False)
    protocol=dict(architecture=ARCH,parent_sha256=grade.file_sha(PARENT/"weights.pt"),
        marks=grade.file_sha(MARKS),woo=grade.file_sha(spatial.WOO),seed=SEED,steps=240,checkpoint_every=40,
        source="AF44 train-only original shallow/hard-normal boxes on the full photo and original detector crop, plus Woo train-only scratch polygons and screened clean crops. No other photos in training.",
        negatives="Exact calyx/highlight/shadow boxes; known neutral exterior of damaged fruit; clean author/screened faces with contact band sampled on both sides. Positive marks override, unknown skin receives no local negative labels.",
        parent_input="independently resize real fruit to 224 for the frozen parent and 320 for the local verifier; only parent good may become mild",
        optimization="fixed AdamW 0.001, 240 updates, 6 AF positives + 6 Woo positives + 7 normal-structure views + 8 clean fruits; local and global losses with damage evidence attenuated by fixed 0.2 of parent good-minus-mild logit margin",
        select="Fixed dev comparison versus border using the same 3-panel safety, mild and lexicographic checks as overall selection; first checkpoint wins exact ties.",
        script=grade.file_sha(__file__),network=grade.file_sha(Path(__file__).with_name("grade_detail_patch.py")))
    old=out/"protocol.json"
    if old.exists() and io.read(old)!=protocol:raise ValueError("A different native trial protocol exists")
    io.atomic(old,protocol)
    small,hi,mask,meta=training();print("Train inputs",Counter(r["source"] for r in meta),flush=True)
    xt,zt=features(model,small,hi);del small,hi
    val=select.input_panels(detail=True)
    vs=[];vh=[]
    for _,sm,large in val.values():
        for a,b in zip(sm,large,strict=True):
            if a is not None:vs.append(a);vh.append(b)
    xv,zv=features(model,vs,vh)
    with torch.inference_mode():
        exact=torch.cat([parent(grade.tensor_rgb(vs[j:j+16],"cuda")) for j in range(0,len(vs),16)])
    parent_diff=float((exact-zv).abs().max())
    if not torch.equal(exact.argmax(1),zv.argmax(1)) or parent_diff>1e-5:
        raise ValueError(f"Parent input/output ABI changed: {parent_diff}")
    baseline=tables(zv.argmax(1).tolist(),val)
    training_rows=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(meta):
        if r["positive"]:training_rows[r["source"]][r["key"]].append(i)
        if r["hard_normal"]:training_rows["normal"][r["key"]].append(i)
        if r["label"]==0:training_rows["clean"][r["key"]].append(i)
    bags={key:list(v.values()) for key,v in training_rows.items()}
    trace=dict(images=len(meta),source=Counter(r["source"] for r in meta),
               unique_positive_and_negative_inputs={k:len(v) for k,v in bags.items()},
               mask_counts={str(i):int(np.sum(mask==i)) for i in [-1,0,1,2]},
               dev_parent_max_abs_logit_error=parent_diff,dev_parent_argmax_equal=len(vs),baseline=baseline)
    io.atomic(out/"input_audit.json",trace)
    mask=torch.from_numpy(mask).cuda();history=[];best=None
    opt=torch.optim.AdamW(model.verifier.parameters(),lr=.001,weight_decay=.016)
    labels=torch.tensor([r["label"] for r in meta],device="cuda")
    for step in range(1,241):
        ids=[]
        for name,count in (("AF",6),("Woo",6),("normal",7),("clean",8)):
            for _ in range(count):
                source=bags[name][rng.integers(len(bags[name]))]
                ids.append(source[rng.integers(len(source))])
        x,lab=xt[ids].float(),mask[ids]
        if rng.random()<.5:x=x.flip((-1,));lab=lab.flip((-1,))
        heat,e=model.evidence(x)
        p,pm=masked_peak(heat,lab==1);n,nm=masked_peak(heat,lab==2);bg,bm=masked_peak(heat,lab==0)
        clean=labels[ids]==0
        margin=(zt[ids,0]-zt[ids,1]).relu()*.2
        loss=F.softplus(-p[pm]).mean()+.9*F.softplus(n[nm]).mean()
        loss+=.65*F.softplus(bg[bm]).mean()+.95*F.softplus(e[clean]-margin[clean]).mean()
        loss+=.5*F.softplus(margin[pm]-p[pm]).mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.verifier.parameters(),3.,error_if_nonfinite=True);opt.step()
        if step%40:continue
        with torch.inference_mode():
            pred=[]
            for j in range(0,len(vs),16):
                _,ev=model.evidence(xv[j:j+16].float())
                pred+=model.repair(zv[j:j+16],ev).argmax(1).tolist()
        m=tables(pred,val);candidate=dict(metrics=m,architecture=ARCH,directory="native320",weight_sha256="")
        select.compare(candidate,dict(metrics=baseline))
        rank=select.candidate_order(candidate)
        h=dict(step=step,loss=float(loss.detach()),metrics=m,passed=candidate["passed"])
        history.append(h);io.atomic(out/"history.json",history)
        if best is None or rank<best:
            chosen=h;best=rank;io.atomic(out/"weights.pt",model.state_dict(),tensor=True)
        if checkpoint_observer is not None:
            checkpoint_observer(step,model,h)
        print("step",step,"auto",m["auto"]["confusion"],"best",chosen["step"],flush=True)
    io.atomic(out/"selection.json",dict(architecture=ARCH,input_side=320,threshold=1.0,step=chosen["step"],
        weights_sha256=grade.file_sha(out/"weights.pt"),parent_sha256=protocol["parent_sha256"],passed=chosen["passed"],
        protocol_sha256=grade.file_sha(old),metrics=chosen["metrics"]))


if __name__=="__main__":
    run()
