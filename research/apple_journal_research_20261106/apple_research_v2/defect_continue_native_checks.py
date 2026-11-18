"""Check source geometry, positive-only gradients and native/SSDA gradient agreement on CPU."""
from collections import Counter
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from class0_patch_refinement import configure,atomic_json,read,load_image
from defect_continue_native_partial import OUT,entries,native_record,tensor,object_loss,native_norm,responsive_params,combine_grads
import defect_continue_public_training as base
from defect_continue_causal import native_cpu_augment,as_raw
import defect9485_small as original
from joint_model_v1 import LossView
from ultralytics.utils.loss import E2ELoss


def gradient_support(model,obj):
    head=model.model[-1]
    x=tensor(obj,"cpu")
    with native_norm(model):
        out=model(x)
        for item in out.values():
            for k in ("boxes","scores"):
                item[k].retain_grad()
        loss,info=object_loss(out,head,obj)
        loss.backward()
        for name,branch in out.items():
            for k in ("scores","boxes"):
                grad=branch[k].grad
                allowed=torch.zeros_like(grad,dtype=torch.bool)
                if k=="scores":
                    allowed[0,obj["cls"],info["sites"]]=True
                else:
                    allowed[0,:,info["sites"]]=True
                assert not torch.any(grad[~allowed]) and (k!="boxes" or grad.abs().sum()>0)
                if k=="scores" and info["details"][name]["valid_scores"]>0:
                    assert grad.abs().sum()>0
    return info["details"]["one2one"]["valid_scores"].item()


def main():
    configure(OUT,"cpu")
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    snap,rows=entries()
    records=[native_record(snap,r) for r in rows]
    keys={o["key"]:o for o in records}
    assert len(keys)==257
    groups=Counter((r["cls"],r["context_side"]) for r in records)
    torch.manual_seed(base.SEED)
    m=base.parent().train();params=responsive_params(m)
    for p in m.parameters():
        p.requires_grad_(any(p is u for u in params))
    prior=read(base.ROOT/"experiments/defect_continue_causal/cpu_shortcut_probe.json")["observations"]
    synth_comparison={}
    with native_norm(m), torch.no_grad():
        for k,items in prior.items():
            observations=[]
            for r in items:
                item=keys[r["donor"]]
                out=m(tensor(item,"cpu"))
                branch=out["one2one"]
                h,w=branch["feats"][0].shape[-2:]
                mask=torch.from_numpy(item["mask"]).float()[None,None]
                weights=F.adaptive_avg_pool2d(mask,(h,w)).flatten()
                weights/=weights.sum()
                real=(branch["scores"][0,item["cls"],:h*w]*weights).sum().item()
                observations.append(dict(key=item["key"],real_photograph_logit=real,
                    prior_synthetic_logit=r["score_baseline"]+r["delta_logit"]["full"]))
            rv=np.array([v["real_photograph_logit"] for v in observations])
            sv=np.array([v["prior_synthetic_logit"] for v in observations])
            synth_comparison[k]=dict(samples=len(rv),original_median=np.median(rv).item(),
                synthetic_median=np.median(sv).item(),actual_higher=int((rv>sv).sum()),
                note="same donor identity; actual/context384 vs donated/context320, different pixel scales and lighting",
                rows=observations)
    selected=sorted(records,key=lambda r:r["key"])
    # 16 independent paired probes against native SSDA supervision, balanced classes,
    # with no access to any development labels, metrics or fold model outputs.
    rng=np.random.default_rng(74215)
    chosen=[]
    for c in (0,1):
        pool=[r for r in selected if r["cls"]==c]
        chosen.extend(pool[i] for i in rng.choice(len(pool),size=8,replace=False))
    ss=original.samples("train")
    criterion=E2ELoss(LossView(m.model[-1],base.EPOCHS))
    support=[gradient_support(m,q) for q in chosen[:3]+chosen[8:11]]
    samples=[]
    for obj in chosen:
        ids=rng.choice(len(ss),size=2,replace=False)
        images=[original.square(load_image(ss[i]["rgb_path"]))[0] for i in ids]
        lab=[np.float32([[g["class_id"],*g["xywh_normalized"]] for g in ss[i]["instances"]]).reshape(-1,5) for i in ids]
        x,y=native_cpu_augment(images,lab,range(2),rng)
        m.zero_grad(set_to_none=True)
        # Compare against each native minibatch using the locked pretrained running stats;
        # avoid modifying the parent while measuring event-by-event gradient agreement.
        with native_norm(m):
            native=criterion(as_raw(m(x)),y)[0].sum()/2
            native.backward()
            ext,details=object_loss(m(tensor(obj,"cpu")),m.model[-1],obj)
            g=torch.autograd.grad(ext,params,allow_unused=True)
            report=combine_grads(params,g,apply=False)
            report.update(cls=obj["cls"],key=obj["key"],selected_sites=len(details["sites"]),
                          qualified_score_sites=int(details["details"]["one2one"]["valid_scores"]),
                          mean_selected_iou=float(details["details"]["one2one"]["iou"]))
            samples.append(report)
    directions={c:dict(n=sum(s["cls"]==c for s in samples),aligned=sum(s["applied"] for s in samples if s["cls"]==c),
                           median_cosine=float(np.median([s["cos"] for s in samples if s["cls"]==c])),
                           score_eligible_cases=sum(s["qualified_score_sites"]>0 for s in samples if s["cls"]==c))
                for c in (0,1)}
    out=dict(real_photos=len(set(r["full_image"] for r in rows)),annotated_regions=len(records),
             native_windows={str(k):v for k,v in groups.items()},native_shape_and_mask_verified_count=len(records),
             support_checked_objects=len(support),nonempty_positive_logit_grad_objects=sum(v>0 for v in support),
             nonpositive_external_class_or_site_gradients=0,manual_directional_probes=samples,
             gradient_alignment=directions,native_vs_paste=synth_comparison,device="cpu",threads=2)
    atomic_json(OUT/"cpu_preflight.json",out)
    print("real_vs_paste",{k:{j:x for j,x in v.items() if j!="rows"} for k,v in synth_comparison.items()},flush=True)
    print("alignment",directions,"support",support,"photos",out["real_photos"],flush=True)


if __name__=="__main__":
    main()
