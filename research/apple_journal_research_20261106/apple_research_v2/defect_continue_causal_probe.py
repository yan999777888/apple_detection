"""CPU support audit and pretraining diagnostic for residual border/colour shortcuts."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from class0_patch_refinement import configure,read,atomic_json,load_image
from defect_continue_public_transfer import empirical_sizes
import defect_continue_causal as experiment
import defect_continue_causal_pair as pair
import defect9485_small as original

LIMIT=24


def support_check(v):
    features=torch.zeros(2,128,40,40)
    branches={name:dict(feats=[features],scores=torch.zeros(2,2,2100,requires_grad=True)) for name in ("one2one","one2many")}
    rank,_=pair.losses(branches,pair.DeltaMask(),v)
    rank.backward()
    footprint=F.adaptive_avg_pool2d(v["mask"].float()[None,None],(40,40))[0,0].flatten()>0
    expected=torch.zeros(2,2,2100,dtype=torch.bool)
    expected[:,v["cls"],:1600]=footprint
    for b in branches.values():
        g=b["scores"].grad
        assert torch.equal(g.abs()>0,expected)
        assert torch.all(g[1][expected[1]]<0) and torch.all(g[0][expected[0]]>0)
    return dict(cells=int(footprint.sum()),leaked_logit_gradients=0,pre_post_direction_correct=True)


@torch.no_grad()
def measure(model,v):
    views=v["views"]
    before,after=views[0],views[1]
    mask=v["mask"].numpy().astype(bool)
    dist=cv2.distanceTransform(np.pad(mask.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
    band=float(min(4,max(1.5,.25*dist.max())))
    core=dist>band
    if core.sum()<12 or core.sum()/mask.sum()<.20:
        return None
    edge=~core
    # Keep the identical host values everywhere except the appropriate intervention.
    inside=torch.from_numpy(core)[None]
    border=torch.from_numpy(edge)[None]
    values={"before":before,"full":after,"core":torch.where(inside,after,before),
            "edge":torch.where(border,after,before)}
    pre=before.permute(1,2,0).numpy()
    post=after.permute(1,2,0).numpy()
    changed=np.any(pre!=post,axis=2)
    lab0=cv2.cvtColor(pre,cv2.COLOR_RGB2LAB)
    lab1=cv2.cvtColor(post,cv2.COLOR_RGB2LAB)
    for key,chromatic in (("luminance",False),("colour",True)):
        lab=lab0.copy()
        if chromatic:
            lab[:,:,1:]=lab1[:,:,1:]
        else:
            lab[:,:,0]=lab1[:,:,0]
        mixed=pre.copy()
        mixed[changed]=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)[changed]
        values[key]=torch.from_numpy(mixed).permute(2,0,1)
    names=list(values)
    pred=model(torch.stack(list(values.values())))
    # Same class, sites, stride and weights as the deployment-head part of the ranking loss.
    feat=pred["one2one"]["feats"][0]
    h,w=feat.shape[-2:]
    pooled=F.adaptive_avg_pool2d(v["mask"].float()[None,None],(h,w))[0,0].flatten()
    weights=pooled/pooled.sum()
    scores=pred["one2one"]["scores"][:,v["cls"],:h*w]
    signals=(scores*weights[None]).sum(1).numpy()
    return dict(host=v["host"],cls=v["cls"],class1_or_class0="pest" if v["cls"] else "scratch",
                polygon_pixels=int(mask.sum()),core_pixels=int(core.sum()),boundary_band_px=band,
                score_baseline=float(signals[0]),
                delta_logit={key:float(z-signals[0]) for key,z in zip(names[1:],signals[1:])})


def main():
    configure(experiment.OUT,"cpu")
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    bank,receipt,table=experiment.bank_data()
    gtrows=original.samples("train")
    gt=[np.float32([[v["class_id"],*v["xywh_normalized"]] for v in r["instances"]]).reshape(-1,5) for r in gtrows]
    sizes=empirical_sizes(gt,original.SIZE)
    model=experiment.base.parent().train()
    results={0:[],1:[]}
    checks=[]
    excluded={0:0,1:0}
    # Use only SSDA train images in manifest order, with original immutable donors.
    with pair.locked_norm(model):
        for i,r in enumerate(gtrows):
            im=original.square(load_image(r["rgb_path"]))[0]
            views,targets,pairs,detail=pair.prepared({i:im},{i:gt[i]},[i],bank,table,sizes,0)
            if not pairs or len(results[pairs[0]["cls"]])>=LIMIT:
                continue
            # Dedicated CPU intervention RNG. The actual fit retains the native training RNG.
            rng=np.random.default_rng(np.random.SeedSequence([experiment.base.SEED,i,552]))
            params=pair.replay_params(rng,1)
            x=pair.before_tensor(pair.warped(views[0],params[0][0]),params[1][0],"cpu")[None]
            v=pair.map_pairs(pairs,params,x)
            if not v:
                continue
            v=v[0]
            checks.append(support_check(v))
            outcome=measure(model,v)
            if outcome is None:
                excluded[v["cls"]]+=1
            else:
                outcome.update(donor=detail[0]["donor"],group=detail[0]["group"])
                results[v["cls"]].append(outcome)
            if all(len(a)>=LIMIT for a in results.values()):
                break
    out={}
    for c,items in results.items():
        assert len(items)==LIMIT
        d={k:np.array([r["delta_logit"][k] for r in items]) for k in items[0]["delta_logit"]}
        active=d["full"]>=.25
        out[c]=dict(usable=len(items),unique_donor_groups=len({r["group"] for r in items}),
                    no_resolved_interior=excluded[c],positive_by_025=int(active.sum()),
                    median_core_fraction=float(np.median([r["core_pixels"]/r["polygon_pixels"] for r in items])),
                    median_delta={k:float(np.median(v)) for k,v in d.items()},
                    full_positive_edge_ge_core=int(np.sum(active & (d["edge"]>=d["core"]))),
                    full_positive_colour_ge_light=int(np.sum(active & (d["colour"]>=d["luminance"]))))
    atomic_json(experiment.OUT/"cpu_shortcut_probe.json",dict(summary=out,observations=results,
        parameter_gradient_support=dict(actual_transformed_pairs=len(checks),all_outside_logits_exactly_zero=True,
            only_correct_class_and_finest_grid=True,background_far_from_polygon_zero=True),
        selection="first 24 usable pairs per class in training manifest order; require minimum 12 core pixels and 20% core, no sorting by scores",
        response="fixed original Small CPU FP32 one2one finest-stride class logits, footprint-weighted; deltas vs identical unmodified host",
        crop_patch=pair.PATCH,warning="pretraining intervention on synthetic images. Contributions can interact and depend on intervention size, so are not additive or causal evidence on actual pathology."))
    print(out,flush=True)
if __name__=="__main__":
    main()
