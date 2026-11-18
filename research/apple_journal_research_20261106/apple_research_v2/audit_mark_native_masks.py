"""Recheck loss-peak placement against the existing scene segmenter's masks on AF development only."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import afruit_grade_crop_alignment as aligned
import afruit_visible_quality as visible
from train_grade_mark_contrast import OUT,io,grade
from matched_grade_loader import load


def mask_level(mask):
    h,w=mask.shape
    s=224/max(h,w);nw,nh=round(w*s),round(h*s)
    target=np.zeros((224,224),dtype=np.float32)
    dy,dx=(224-nh)//2,(224-nw)//2
    target[dy:dy+nh,dx:dx+nw]=cv2.resize(mask.astype(np.float32),(nw,nh),interpolation=cv2.INTER_AREA)
    return target.reshape(28,8,28,8).mean((1,3))


def summary(rows,key):
    eligible=[r for r in rows if r["parent"]!=2]
    good=[r[key] for r in eligible if r["grade"]==0]
    mild=[r[key] for r in eligible if r["grade"]==1]
    auc=np.mean([int(a>b)+.5*int(a==b) for a in mild for b in good])
    candidates=[float("inf")]+sorted(set(good+mild),reverse=True)
    best=max(sum(x>=t for x in mild) for t in candidates if sum(x>=t for x in good)<=1)
    return dict(eligible_good=len(good),eligible_mild=len(mild),auc=float(auc),upper_mild_at_1_false_good=best,
                deployed_zero=dict(good_positive=sum(x>0 for x in good),mild_positive=sum(x>0 for x in mild)))


@torch.inference_mode()
def check():
    model,_,_=load(OUT,"cuda");pipe=visible.QualityPipeline()
    prep=pipe.predictor
    rows=aligned.split_rows("val")
    regions=io.read(aligned.OUT/"val_regions.json")
    out=[]
    for r,a in zip(rows,regions,strict=True):
        if a["grade"]>1 or a["crop"] is None:continue
        im=visible.prepare_input(aligned.read_frame(r),aligned.SIDE)
        tensor=prep.preprocess([im]);prep.batch=(["frame"],[im],[""])
        stage=prep.postprocess(pipe.stage.model.predict_source(tensor,"stage"),tensor,[im])[0].cpu()
        if len(stage.boxes)!=1 or a["crop"]!=visible.crop_bounds(stage.boxes.data.numpy()[0,:4],im.shape):
            raise ValueError("Native stage crop differs from frozen AF development crop")
        x,y,t,b=a["crop"];mask=mask_level(stage.masks.data.numpy()[0,y:b,x:t])
        mm=torch.tensor(mask[2:-2,2:-2],device="cuda")[None,None]
        fraction=F.avg_pool2d(mm,2,1)[0,0]
        # Require the whole scoring kernel to be at least a grid cell inside the predicted fruit.
        inside=-F.max_pool2d(-mm,4,1,padding=1)[0,0]>.99
        f,z=model.encode(grade.batch_preprocess([im[y:b,x:t]],"cuda"))
        sc,e=model.evidence(f)
        heat=F.avg_pool2d(sc[:,None,2:-2,2:-2],2,1)[0,0]
        ij=np.unravel_index(heat.argmax().item(),heat.shape)
        assert inside.shape==heat.shape and int(inside.sum())>=3
        protected=heat[inside].topk(3).values.mean()
        out.append(dict(key=r["path"],grade=r["grade"],parent=z.argmax(1).item(),peak_foreground_fraction=round(fraction[ij].item(),5),
                        peak_inside_margin=bool(inside[ij]),full=e.item(),interior=protected.item()))
    d=dict(development_only=True,model_sha=grade.file_sha(OUT/"weights.pt"),stage_sha=visible.STAGE_WEIGHT_SHA,
           rating="AUC and optimum recall under one false downgrade diagnose score separation on this development sample only. Threshold was left unchanged.",
           full=summary(out,"full"),interior=summary(out,"interior"),
           clearly_out_or_edge=sum(not r["peak_inside_margin"] for r in out),mostly_background=sum(r["peak_foreground_fraction"]<.5 for r in out),n=len(out),rows=out)
    io.atomic(OUT/"development_mask_peak_audit.json",d)
    print(d["full"],d["interior"],d["clearly_out_or_edge"],d["mostly_background"],d["n"],flush=True)


if __name__=="__main__":check()

