"""Native SSDA matched replay plus constrained partial-label updates from original public photos."""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from class0_patch_refinement import configure,atomic_json,sha,read
from defect_continue_native_partial import OUT,MANIFEST,SNAP,entries,native_record,tensor,object_loss,native_norm,responsive_params,combine_grads,preserve_native_groups,remove_native_drift
from defect_continue_causal import as_raw
import defect_continue_public_training as base
import defect9485_small as original
from joint_state_v1 import canonical_sha


def schedule(records):
    # Two known lesions per update. 520 scratch + 280 insect photographs/regions
    # (several views may come from one source); cycle without replacement within class.
    rng=np.random.default_rng(29616)
    classes=rng.permutation([0]*520+[1]*280)
    order={}
    for c in (0,1):
        ids=np.array([i for i,r in enumerate(records) if r["cls"]==c])
        n=sum(classes==c)
        order[c]=iter(np.concatenate([rng.permutation(ids) for _ in range(math.ceil(n/len(ids)))]).tolist())
    result=[next(order[c]) for c in classes]
    assert set(result)==set(range(len(records)))
    return np.asarray(result).reshape(-1,2)


@contextmanager
def only_heads(model,params):
    selected={id(p) for p in params}
    others=[p for p in model.parameters() if p.requires_grad and id(p) not in selected]
    try:
        for p in others:
            p.requires_grad_(False)
        with native_norm(model):
            yield
    finally:
        for p in others:
            p.requires_grad_(True)


def fit():
    from ultralytics.utils.loss import E2ELoss
    from ultralytics.utils.torch_utils import ModelEMA
    from joint_model_v1 import LossView
    snap,rows=entries()
    check=read(OUT/"cpu_preflight.json")
    if check["annotated_regions"]!=257 or check["nonpositive_external_class_or_site_gradients"]!=0:
        raise RuntimeError("Positive-only audit not complete")
    drift=read(OUT/"cpu_direction_preserved.json")
    if not (drift["active"]>0 and all(drift["has_both_score_directions"].values())):
        raise RuntimeError("Native score-preservation probe not verified")
    amounts=[native_record(snap,r) for r in rows]
    order=schedule(amounts)
    protocol=base.protocol("replay")
    protocol.update(mode="native_partial",external_snapshot=SNAP,external_manifest_sha=MANIFEST,
                    code_sha=sha(__file__),partial_sha=sha(Path(__file__).with_name("defect_continue_native_partial.py")),
                    check_sha=sha(OUT/"cpu_preflight.json"),drift_check_sha=sha(OUT/"cpu_direction_preserved.json"),raw_regions_by_class=[224,33],exposures=[520,280],
                    real_views_per_optimizer_step=2,predeclared_auxiliary_stream_sha=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
                    main_comparator="original SSDA replay 8 epochs, same parent and per-example augmentation stream",
                    auxiliary_contract="top3 sites intersect native polygon; only labeled class and box at these sites. Bias/BN/trunk frozen for external update. Preserve the linearized means of each class's ordinary and hard native out-of-GT-region logit groups. External score only when box overlaps and score below hinge; positive inner product with native detection head gradient, cap at 10% of its norm.")
    key=canonical_sha(protocol)
    if (OUT/"receipt.json").exists():
        raise RuntimeError("Completed run already present")
    if (OUT/"protocol.json").exists() and canonical_sha(read(OUT/"protocol.json"))!=key:
        raise RuntimeError("Existing output uses different protocol")
    atomic_json(OUT/"protocol.json",protocol)
    ims,gt=original.prepare()
    rng=np.random.default_rng(base.SEED)
    torch.manual_seed(base.SEED)
    torch.backends.cudnn.benchmark=False
    model=base.parent().cuda().train()
    ema=ModelEMA(model,decay=.997,tau=200)
    criterion=E2ELoss(LossView(model.model[-1],base.EPOCHS))
    params=responsive_params(model)
    opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim>1],weight_decay=.0005),
                           dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim<=1],weight_decay=0.)],lr=base.LR)
    latest=OUT/"latest.pt"
    history=base.load_resume(latest,key,model,ema,opt,rng,criterion)
    steps=history[-1]["updates"] if history else 0
    for e in range(len(history),base.EPOCHS):
        perm=rng.permutation(len(ims))
        batches=np.array_split(perm,len(perm)//original.BATCH)
        events=[];losses=[]
        for start in range(0,len(batches),original.ACCUM):
            group=batches[start:start+original.ACCUM]
            age=e+start/len(batches)
            gain=(.05+.95*(1+math.cos(math.pi*max(0,age-2)/(base.EPOCHS-2)))/2)*min(1,(age+.05)/2)
            for g in opt.param_groups:
                g["lr"]=base.LR*gain
            opt.zero_grad(set_to_none=True)
            for ids in group:
                x,y=original.augment(ims,gt,ids,rng)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    pred=model(x)
                main=criterion(as_raw(pred),y)[0].sum()/len(ids)
                if not main.isfinite():
                    raise RuntimeError("Nonfinite native detection loss")
                (main/len(group)).backward()
                losses.append(main.detach().item())
                del main,pred
            chosen=order[steps]
            with only_heads(model,params):
                basis=preserve_native_groups(model,params,x,y)
                extra=[]
                for idx in chosen:
                    r=amounts[idx]
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        raw=model(tensor(r,"cuda"))
                    extra.append(object_loss(raw,model.model[-1],r)[0])
                obj=torch.stack(extra).mean()
                if not obj.isfinite():
                    raise RuntimeError("Nonfinite external positive-only objective")
                aux_grads=torch.autograd.grad(obj,params,allow_unused=True)
                aux_grads,retained=remove_native_drift(params,aux_grads,basis)
            event=combine_grads(params,aux_grads)
            event.update(ids=chosen.tolist(),keys=[amounts[i]["key"] for i in chosen],
                         classes=[amounts[i]["cls"] for i in chosen],loss=obj.detach().item(),
                         native_constraints=len(basis),retained_external_norm=retained)
            events.append(event)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
            opt.step();ema.update(model);steps+=1
        criterion.update()
        info=dict(epoch=e+1,updates=steps,base_permutation=hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest(),
                  base_rng_after=canonical_sha(rng.bit_generator.state),original_samples=len(perm),inserted=0,
                  native_loss=float(np.mean(losses)),external_updates_accepted=sum(v["applied"] for v in events),
                  exposed=dict(Counter(c for v in events for c in v["classes"])),
                  median_aux_cosine=float(np.median([v["cos"] for v in events])))
        expected=read(base.OUT/"replay/training.json")[e]
        for k in ("epoch","updates","base_permutation","base_rng_after","original_samples"):
            if info[k]!=expected[k]:
                raise RuntimeError(f"Unpaired replay stream: {k} at epoch {e+1}")
        history.append(info)
        atomic_json(OUT/f"auxiliary_epoch{e+1:02}.json",events)
        base.atomic_latest(latest,dict(protocol=key,model=model.state_dict(),ema=ema.ema.state_dict(),
                   updates=steps,optimizer=opt.state_dict(),rng=rng.bit_generator.state,
                   torch_rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state(),epochs=history))
        atomic_json(OUT/"training.json",history)
        print(info,flush=True)
    assert steps==400
    base.publish(OUT,ema.ema.float().cpu().state_dict(),history,key)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["fit","evaluate"])
    args=p.parse_args()
    configure(OUT,"cuda:0")
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    with base.exclusive_run():
        if args.action=="fit":
            fit()
        else:
            with base.evaluation_output(OUT):
                original.evaluate()
