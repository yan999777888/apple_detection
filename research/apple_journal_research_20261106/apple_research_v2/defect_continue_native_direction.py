"""CPU finite-step diagnostic: does the constrained external direction increase all low-score SSDA sites?"""
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from class0_patch_refinement import read,sha,atomic_json,configure,load_image
from defect_continue_native_partial import OUT,entries,native_record,responsive_params,native_norm,object_loss,tensor,combine_grads,preserve_native_groups,remove_native_drift
from defect_continue_native_train import schedule,only_heads
from defect_continue_causal import native_cpu_augment,as_raw
import defect_continue_public_training as base
import defect9485_small as native
from ultralytics.utils.loss import E2ELoss
from joint_model_v1 import LossView


@torch.no_grad()
def signals(model,rows):
    outputs=[]
    head=model.model[-1]
    with native_norm(model):
        for r in rows:
            # Source training frames are 1024 squares; preserve the production 768 scale.
            im=native.square(load_image(r["rgb_path"]))[0]
            out=model(torch.from_numpy(im).permute(2,0,1)[None].float()/255)["one2one"]
            H,W=out["feats"][0].shape[-2:]
            raw=out["scores"][0,:,:H*W].float().reshape(2,H,W)
            # Nearby pixels are excluded only when describing far-away site distribution;
            # they provide no labels or gradients in this audit.
            inside=np.zeros((H,W),bool)
            for a in r["instances"]:
                x,y,w,h=a["xywh_normalized"]
                l,t=np.floor(np.array([x-w,y-h])*[W,H]).astype(int).clip(0,[W,H])
                rr,b=np.ceil(np.array([x+w,y+h])*[W,H]).astype(int).clip(0,[W,H])
                inside[t:b,l:rr]=True
            outputs.append((raw,torch.as_tensor(inside)))
    return outputs


def main():
    configure(OUT,"cpu")
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    snap,rows=entries()
    objects=[native_record(snap,r) for r in rows]
    stream=schedule(objects)
    assert len(stream)==400 and len(set(stream.flatten().tolist()))==257
    short=read(OUT/"cpu_preflight.json")
    ss=native.samples("train")
    rng=np.random.default_rng(98862)
    indices=rng.permutation(len(ss))
    used=indices[:24]
    val=indices[24:36]
    m=base.parent().train()
    params=responsive_params(m)
    criterion=E2ELoss(LossView(m.model[-1],8))
    direction=[torch.zeros_like(p) for p in params]
    active=0
    for step in range(12):
        sel=used[step*2:step*2+2]
        images=[native.square(load_image(ss[i]["rgb_path"]))[0] for i in sel]
        labels=[np.float32([[g["class_id"],*g["xywh_normalized"]] for g in ss[i]["instances"]]).reshape(-1,5) for i in sel]
        x,y=native_cpu_augment(images,labels,range(len(sel)),rng)
        m.zero_grad(set_to_none=True)
        with only_heads(m,params):
            loss=criterion(as_raw(m(x)),y)[0].sum()/len(sel)
            loss.backward()
            basis=preserve_native_groups(m,params,x,y)
            work=[]
            for idx in stream[step]:
                r=objects[idx]
                work.append(object_loss(m(tensor(r,"cpu")),m.model[-1],r)[0])
            g=torch.autograd.grad(torch.stack(work).mean(),params,allow_unused=True)
            g,retained=remove_native_drift(params,g,basis)
        receipt=combine_grads(params,g,apply=False)
        if receipt["applied"]:
            active+=1
            # Equalize each contributing step; clipping in the real fit uses its
            # own native gradient scale. This probe isolates direction only.
            norm=sum(v.square().sum() for v in g if v is not None).sqrt().item()
            for ds,z in zip(direction,g):
                if z is not None:
                    ds.add_(z,alpha=1/max(norm,1e-12))
    if not active:
        out=dict(active=0,native_probe_frames=len(val),note="All 12 source batches rejected the partial update.")
    else:
        subset=[ss[i] for i in val]
        before=signals(m,subset)
        norm=sum(v.square().sum() for v in direction).sqrt().item()
        weights=sum(v.detach().square().sum() for v in params).sqrt().item()
        alpha=.0001*weights/max(norm,1e-12)
        with torch.no_grad():
            for p,d in zip(params,direction):
                p.add_(d,alpha=-alpha)
        after=signals(m,subset)
        deltas={0:[],1:[]}
        for (a,inside),(b,_) in zip(before,after):
            for c in (0,1):
                selected=(~inside)&(a[c].sigmoid()<native.PRIMARY[c])
                deltas[c].extend((b[c]-a[c])[selected].tolist())
        out=dict(active=active,source_steps=12,native_probe_frames=len(val),
                 heldout_train_ids=[r["image_id"] for r in subset],
                 has_both_score_directions={
                    c:bool(np.any(np.array(v)>.00001) and np.any(np.array(v)<-.00001)) for c,v in deltas.items()},
                 outside_GT_low_score_cells={c:dict(cells=len(v),up=int(np.sum(np.array(v)>.00001)),
                    down=int(np.sum(np.array(v)<-.00001)),median_logit_change=float(np.median(v))) for c,v in deltas.items()},
                 weight_relative_step=.0001,probe_precision="cpu FP32 train-only; normalized summed allowed direction, not an optimizer epoch",
                 combined_schedule_exposures=dict(scratch=520,pest=280),distinct_regions_seen=len(set(stream.ravel())))
    atomic_json(OUT/"cpu_direction_preserved.json",out)
    print(out,flush=True)
if __name__=="__main__":
    main()
