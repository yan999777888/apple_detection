"""Paired pasted lesion supervision. Native Small checkpoint and inference stay standalone."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import defect_continue_public_training as base
import defect9485_small as original
import defect_continue_causal_pair as pair
from class0_patch_refinement import ROOT,configure,read,sha,atomic_json,load_image
from defect_continue_public_transfer import load_library,empirical_sizes,draw_transfer
from joint_state_v1 import canonical_sha

OUT=ROOT/"experiments/defect_continue_causal"
# Averaged over *all original batch samples*. With .35 insertion probability
# the expected multiplier per native supervised batch is small.
RANK=.20
MASK=.10


def bank_data():
    bank,receipt=load_library(name=pair.SNAP)
    ref=read(base.OUT/"preflights"/pair.SNAP/"preflight.json")
    assert receipt["manifest_sha256"]==pair.MANIFEST and receipt["accepted_polygons_by_class"]=={0:200,1:33}
    assert ref["transfer_code_sha256"]==sha(Path(base.__file__).with_name("defect_continue_public_transfer.py"))
    return bank,receipt,pair.donor_lookup(bank)


def native_cpu_augment(*args):
    """Call the unmodified native augmentation. Redirect only explicit CUDA transfers in the CPU test."""
    to=torch.Tensor.to
    tensor=torch.tensor
    as_tensor=torch.as_tensor
    def cpu_to(t,*a,**kw):
        a=tuple("cpu" if isinstance(v,str) and v=="cuda" else v for v in a)
        return to(t,*a,**kw)
    def construct(fn):
        def f(*a,**kw):
            if kw.get("device")=="cuda":
                kw["device"]="cpu"
            return fn(*a,**kw)
        return f
    with patch.object(torch.Tensor,"to",cpu_to),patch.object(torch.Tensor,"cuda",lambda self,*a,**kw:self), \
            patch.object(torch,"as_tensor",construct(as_tensor)),patch.object(torch,"tensor",construct(tensor)):
        return original.augment(*args)


def as_raw(pred):
    return {name:{k:([x.float() for x in v] if isinstance(v,list) else v.float())
                  for k,v in vals.items()} for name,vals in pred.items()}


def sources():
    return {p.name:sha(p) for p in (Path(__file__),Path(pair.__file__),Path(base.__file__),Path(original.__file__),
                                    Path(base.__file__).with_name("defect_continue_public_transfer.py"))}


def audit():
    bank,receipt,table=bank_data()
    rows=original.samples("train")
    gt=[np.float32([[g["class_id"],*g["xywh_normalized"]] for g in r["instances"]]).reshape(-1,5) for r in rows]
    sizes=empirical_sizes(gt,original.SIZE)
    assert (len(rows),sum(map(len,gt)))==(596,1276)
    rng=np.random.default_rng(base.SEED)
    data=[]
    counts=Counter()
    n_images=0
    verified_masks=0
    # Real host photos and immutable public donor polygons; audit first 100 in manifest order.
    for i,row in enumerate(rows[:100]):
        im=original.square(load_image(row["rgb_path"]))[0]
        lab=gt[i]
        after,targets,pairs,info=pair.prepared({i:im},{i:lab},[i],bank,table,sizes,0)
        gold,glab,gin=draw_transfer(im,lab,bank,sizes,base.transfer_rng(0,i))
        assert np.array_equal(after[0],gold) and np.array_equal(targets[0],glab)
        assert bool(info)==bool(gin)
        if info:
            assert {k:v for k,v in info[0].items() if k!="host"}==gin
            g=pairs[0]["mask"].astype(bool)
            for _,x,y,w,h in lab:
                x0,y0=np.floor(np.array([x-w/2,y-h/2])*original.SIZE).astype(int).clip(0,original.SIZE)
                x1,y1=np.ceil(np.array([x+w/2,y+h/2])*original.SIZE).astype(int).clip(0,original.SIZE)
                assert not g[y0:y1,x0:x1].any()
            verified_masks+=1
            counts[int(gin["target_class"])]+=1
            if len(data)<6:
                data.append((after[0],targets[0],pairs[0]))
        n_images+=1
    assert min(counts.values())>=3 and len(data)==6
    # Compare the pair path against the exact native augment independently on CPU.
    after,targets,pairs=map(list,zip(*data))
    for i,p in enumerate(pairs):
        p["i"]=i
    r0=deepcopy(rng)
    params=pair.replay_params(rng,len(data))
    native,labels=native_cpu_augment(after,targets,range(len(data)),rng)
    assert rng.bit_generator.state==params[-1]
    native_pre,_=native_cpu_augment([p["before"] for p in pairs],[g[:-1] for g in targets],range(len(data)),r0)
    assert rng.bit_generator.state==r0.bit_generator.state
    for i,im in enumerate(after):
        expected=pair.before_tensor(pair.warped(im,params[0][i]),params[1][i],"cpu")
        assert torch.equal(expected,native[i])
        exp_pre=pair.before_tensor(pair.warped(pairs[i]["before"],params[0][i]),params[1][i],"cpu")
        assert torch.equal(exp_pre,native_pre[i])
        mask=pair.warped(pairs[i]["mask"],params[0][i],True)
        diff=(native_pre[i]!=native[i]).any(0).numpy()
        assert not np.any(diff & ~(cv2.dilate(mask,np.ones((5,5),np.uint8)).astype(bool)))
    mapped=pair.map_pairs(pairs,params,native)
    assert len(mapped)>=4
    model=base.parent().train()
    aux=pair.DeltaMask()
    norm_before={n:b.clone() for n,b in model.named_buffers() if "running_" in n}
    with pair.locked_norm(model):
        ranking,local=pair.metric_pair(model,aux,mapped[0])
        assert torch.isfinite(ranking+local)
        (RANK*ranking+MASK*local).backward()
        grad=float(aux.proj.weight.grad.norm())
        stem=float(model.model[0].conv.weight.grad.norm())
        head=model.model[-1]
        one2one_grad=sum(float(p.grad.abs().sum()) for p in head.one2one["cls_head"].parameters() if p.grad is not None)
        assert min(grad,stem,one2one_grad)>0
        # Pair ranking decreases if only the post-lesion class logits are raised.
        with torch.no_grad():
            raw=model(mapped[0]["views"])
            initial=pair.losses(raw,aux,mapped[0])[0].item()
            for mode in ("one2many","one2one"):
                raw[mode]["scores"][1,mapped[0]["cls"]]+=1.
            assert pair.losses(raw,aux,mapped[0])[0].item()<initial
    for n,b in model.named_buffers():
        if n in norm_before:
            assert torch.equal(norm_before[n],b)
    model.eval()
    with torch.no_grad():
        x=mapped[0]["views"][1:2]
        ref=model(x)[0]
        del aux
        same=model(x)[0]
        assert torch.equal(ref,same) and not any(k.startswith("aux") for k in model.state_dict())
    report=dict(source_hashes=sources(),class_counts=receipt["accepted_polygons_by_class"],native_host_images=n_images,
                identical_native_transfer_outputs=True,identical_native_augmentation=True,
                author_masks_validated=verified_masks,author_mask_counts=counts,
                no_original_GT_pixel_supervised=True,identical_base_rng=True,
                paired_warps_checked=len(data),paired_crops_after_edge_filter=len(mapped),
                aux_rank=float(ranking.detach()),aux_delta_mask=float(local.detach()),
                gradients=dict(temporal_mask_head=grad,shared_first_conv=stem,one2one_classifier=one2one_grad),
                setting_all_post_logits_higher_reduces_rank=True,aux_preserves_BN_buffers=True,
                discard_aux_bitwise_identical_deployment_output=True,device="cpu",threads=torch.get_num_threads())
    atomic_json(OUT/"cpu_preflight.json",report)
    print(report,flush=True)


def fit():
    from ultralytics.utils.loss import E2ELoss
    from ultralytics.utils.torch_utils import ModelEMA
    from joint_model_v1 import LossView
    if (OUT/"receipt.json").exists():
        raise RuntimeError("Completed causal run already exists")
    check=read(OUT/"cpu_preflight.json")
    if check["source_hashes"]!=sources():
        raise RuntimeError("CPU preflight is stale")
    bank,receipt,table=bank_data()
    spec=base.protocol("external",receipt)
    spec.update(mode="paired_causal",source_hashes=sources(),rank_weight=RANK,delta_mask_weight=MASK,
                only_change="pairs matching original public placements; native-pitch 320 crops; same-coordinate relative logits plus localized response change",
                extra_deployment_parameters=0)
    key=canonical_sha(spec)
    if (OUT/"protocol.json").exists() and canonical_sha(read(OUT/"protocol.json"))!=key:
        raise RuntimeError("Protocol differs from existing run")
    atomic_json(OUT/"protocol.json",spec)
    images,labels=original.prepare()
    sizes=empirical_sizes(labels,original.SIZE)
    rng=np.random.default_rng(base.SEED)
    torch.manual_seed(base.SEED);torch.backends.cudnn.benchmark=False
    model=base.parent().cuda().train()
    ema=ModelEMA(model,decay=.997,tau=200)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(846)
        aux=pair.DeltaMask()
    aux=aux.cuda()
    trained=list(model.parameters())+list(aux.parameters())
    opt=torch.optim.AdamW([dict(params=[v for v in trained if v.requires_grad and v.ndim>1],weight_decay=.0005),
                           dict(params=[v for v in trained if v.requires_grad and v.ndim<=1],weight_decay=0.)],lr=base.LR)
    criterion=E2ELoss(LossView(model.model[-1],base.EPOCHS))
    path=OUT/"latest.pt"
    history=base.load_resume(path,key,model,ema,opt,rng,criterion)
    if history:
        aux.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["aux"])
    steps=history[-1]["updates"] if history else 0
    for epoch in range(len(history),base.EPOCHS):
        perm=rng.permutation(len(images));batches=np.array_split(perm,len(perm)//original.BATCH)
        values=[];adds=[];ranks=[];deltas=[]
        for start in range(0,len(batches),original.ACCUM):
            group=batches[start:start+original.ACCUM]
            age=epoch+start/len(batches)
            gain=(.05+.95*(1+math.cos(math.pi*max(0,age-2)/(base.EPOCHS-2)))/2)*min(1,(age+.05)/2)
            for p in opt.param_groups:
                p["lr"]=base.LR*gain
            opt.zero_grad(set_to_none=True)
            for ids in group:
                ims,gt,pairs,records=pair.prepared(images,labels,ids,bank,table,sizes,epoch)
                adds.extend(records)
                params=pair.replay_params(rng,len(ids))
                x,y=original.augment(ims,gt,range(len(ids)),rng)
                assert rng.bit_generator.state==params[-1]
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    raw=model(x)
                supervised=criterion(as_raw(raw),y)[0].sum()/len(ids)
                if not supervised.isfinite():
                    raise RuntimeError("Nonfinite native detection loss")
                (supervised/len(group)).backward()
                values.append(float(supervised.detach()))
                del supervised,raw
                with pair.locked_norm(model):
                    for matched in pair.map_pairs(pairs,params,x):
                        with torch.autocast("cuda",dtype=torch.bfloat16):
                            rank,delta=pair.metric_pair(model,aux,matched)
                        extra=(RANK*rank+MASK*delta)/(len(ids)*len(group))
                        if not extra.isfinite():
                            raise RuntimeError("Nonfinite paired supervision")
                        extra.backward()
                        ranks.append(float(rank.detach()));deltas.append(float(delta.detach()))
            torch.nn.utils.clip_grad_norm_(trained,10.,error_if_nonfinite=True)
            opt.step();ema.update(model);steps+=1
        criterion.update()
        if not ranks or not adds:
            raise RuntimeError("Missing positive pairs")
        record=dict(epoch=epoch+1,updates=steps,base_permutation=hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest(),
                    base_rng_after=canonical_sha(rng.bit_generator.state),original_samples=len(perm),
                    inserted=len(adds),inserted_by_class=dict(Counter(a["target_class"] for a in adds)),
                    valid_paired_crops=len(ranks),rank_loss=np.mean(ranks).item(),delta_loss=np.mean(deltas).item(),
                    native_loss=np.mean(values).item())
        atomic_json(OUT/f"inserted_epoch{epoch+1:02}.json",adds)
        history.append(record)
        base.atomic_latest(path,dict(protocol=key,model=model.state_dict(),ema=ema.ema.state_dict(),
              aux=aux.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,updates=steps,
              torch_rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state(),epochs=history))
        atomic_json(OUT/"training.json",history)
        print(record,flush=True)
    assert steps==400
    control=base.mode_folder("external",receipt)
    for a,b in zip(history,read(control/"training.json"),strict=True):
        for k in ("base_rng_after","base_permutation","original_samples","updates","inserted"):
            assert a[k]==b[k],f"Control mismatch {a['epoch']=} {k=}"
        filename=f"inserted_epoch{a['epoch']:02}.json"
        assert read(OUT/filename)==read(control/filename)
    base.publish(OUT,ema.ema.float().cpu().state_dict(),history,key)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=["check","fit","evaluate"])
    args=ap.parse_args()
    configure(OUT,"cpu" if args.action=="check" else "cuda:0")
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    if args.action=="check":
        audit()
    else:
        with base.exclusive_run():
            if args.action=="fit":
                fit()
            else:
                with base.evaluation_output(OUT):
                    original.evaluate()
