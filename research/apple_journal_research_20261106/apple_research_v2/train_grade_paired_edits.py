"""Teach a local score to change when a confirmed mark, but not a normal landmark, is removed."""
from collections import Counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import afruit_grade_train as io
import afruit_grade_model as grade
import select_grade_repair as dev
import train_grade_detail_patch as patch
from grade_detail_patch import ARCH
from matched_grade_loader import load
from train_grade_capture_contrast import ranking


OUT=grade.ROOT/"experiments/grade_pairs_20261119/paired_edits"
PARENT=grade.ROOT/"experiments/grade_error49_v1/border_native320_woo_ref"
SEED=119027


def repair_skin(image,mask):
    zone=cv2.resize(mask.astype(np.uint8),(320,320),interpolation=cv2.INTER_NEAREST)
    boundary=cv2.dilate(zone,np.ones((3,3),np.uint8))
    edited=cv2.inpaint(image,boundary,3,cv2.INPAINT_TELEA)
    if not np.array_equal(image[boundary==0],edited[boundary==0]):
        raise ValueError("Counterfactual edit escaped the annotated area")
    return edited


def main():
    if (OUT/"summary.json").exists():raise FileExistsError(OUT)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(SEED)
    digest=grade.file_sha(PARENT/"weights.pt")
    assert digest=="b77cecb5695dcc6296e7d0c86b8aa3683fdc2dc5dcee6dbf2e0143bcb66b769a"
    io.atomic(OUT/"protocol.json",dict(parent=digest,steps=150,seed=SEED,check_every=25,lr=.00035,
            instruction="True author scratch polygons and supervised training marks only. Make an edited companion where just the mark is locally filled. For known normal anatomy apply the identical operator. Penalise a failure to prefer the real trace to its counterpart; keep normal landmark response invariant. No whole-image grade is assigned to the edit, or to unannotated skin. Absolute control and relative arm have identical exposures and parent.",
            selection="Frozen protocol in enclosing experiment. No new data at inference.",training=grade.file_sha(__file__)))
    m,cut,side=load(PARENT,"cuda")
    m.eval();m.parent.requires_grad_(False);m.verifier.requires_grad_(True)
    images,high,mask,meta=patch.training()
    field,prior=patch.features(m,images,high)
    panels=dev.input_panels(detail=True)
    low_val=[];high_val=[]
    for _,low,hi in panels.values():
        low_val.extend(x for x in low if x is not None)
        high_val.extend(x for x in hi if x is not None)
    vf,vz=patch.features(m,low_val,high_val)
    ix=[];targets=[];types=[];edited=[]
    for i,lab in enumerate(mask):
        for label in (1,2):
            area=(lab==label)
            if not area.any():continue
            edited.append(repair_skin(high[i],area));types.append(label)
            targets.append(area);ix.append(i)
    ef,_=patch.features(m,[images[i] for i in ix],edited)
    zones=torch.tensor(np.stack(targets),device="cuda")
    source_ids=torch.tensor(ix,device="cuda")
    mark=np.array(types)==1
    healthy=np.array([i for i,r in enumerate(meta) if r["label"]==0])
    assert mark.any() and (~mark).any() and len(healthy)
    io.atomic(OUT/"inputs.json",dict(real_images=len(set(r["key"] for r in meta)),paired_views=len(ix),
               mark_views=int(mark.sum()),normal_control_views=int((~mark).sum()),
               unique_mark_photos=len({meta[i]["key"] for i,k in zip(ix,types) if k==1}),
               source_counts=dict(Counter(r["source"] for r in meta))))
    original={k:v.cpu().clone() for k,v in m.verifier.state_dict().items()}
    for arm,coef in (("anatomy_control",0.),("localized_change",1.)):
        rng=np.random.default_rng(SEED);m.verifier.load_state_dict(original)
        opt=torch.optim.AdamW(m.verifier.parameters(),lr=.00035,weight_decay=.01)
        best=None;history=[]
        for step in range(1,151):
            ids=np.concatenate([rng.choice(np.flatnonzero(mark),8),rng.choice(np.flatnonzero(~mark),8)])
            valid=rng.choice(healthy,8);inds=source_ids[ids]
            z=prior[inds]
            heat,global_ev=m.evidence(field[inds].float())
            heat_edit,_=m.evidence(ef[ids].float())
            real,_=patch.masked_peak(heat,zones[ids])
            erased,_=patch.masked_peak(heat_edit,zones[ids])
            _,ev_good=m.evidence(field[valid].float())
            bias=.2*(z[:,0]-z[:,1]).relu()
            clean_bias=.2*(prior[valid,0]-prior[valid,1]).relu()
            pos=F.softplus(.5+bias[:8]-real[:8]).mean()
            neg=F.softplus(1+real[8:]).mean()+F.softplus(.75+ev_good-clean_bias).mean()
            change=F.softplus(1.2-real[:8]+erased[:8]).mean()+F.smooth_l1_loss(real[8:],erased[8:])
            loss=pos+neg+coef*change
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.verifier.parameters(),3.,error_if_nonfinite=True);opt.step()
            if step%25:continue
            with torch.inference_mode():
                pred=[]
                for j in range(0,len(vf),12):
                    _,e=m.evidence(vf[j:j+12].float());pred.extend(m.repair(vz[j:j+12],e).argmax(1).tolist())
            totals=patch.tables(pred,panels);rank=ranking(totals)
            history.append(dict(step=step,metrics=totals,rank=list(rank)))
            if best is None or rank>best:
                best=rank;chosen=history[-1];io.atomic(OUT/arm/"weights.pt",m.state_dict(),tensor=True)
        io.atomic(OUT/arm/"history.json",history)
        io.atomic(OUT/arm/"selection.json",dict(architecture=ARCH,input_side=320,threshold=cut,
                    weights_sha256=grade.file_sha(OUT/arm/"weights.pt"),parent_sha256=digest,step=chosen["step"],
                    metrics=chosen["metrics"],passed=bool(best[0])))
        print(arm,chosen["metrics"]["auto"]["confusion"],flush=True)
    io.atomic(OUT/"summary.json",{a:io.read(OUT/a/"selection.json") for a in ("anatomy_control","localized_change")})


if __name__=="__main__":main()
