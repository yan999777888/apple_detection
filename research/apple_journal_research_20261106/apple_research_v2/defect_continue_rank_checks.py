"""CPU numerical and indexing regression checks for the dense quality detector."""

import numpy as np
import torch
from ultralytics.utils.loss import E2ELoss
from defect_continue_rank import OUT, spatial, train_pair, QualityDetector, SEED, BASE_SHA, pack
from defect_continue_rank_net import assign, auxiliary, flatten_quality, fused_logits, batch_targets, joint_loss
from class0_patch_refinement import atomic_json, sha
from defect_continue_transplant import split_train
from joint_model_v1 import LossView


def check():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    passed=[]
    def require(name,value):
        if not value:raise AssertionError(name)
        passed.append(name)

    fit,cal=split_train()
    require("490 fit and 106 separate calibration",(len(fit),len(cal))==(490,106))
    require("disjoint image IDs",not ({r["image_id"] for r in fit}&{r["image_id"] for r in cal}))
    im=np.zeros((24,24,3),np.uint8);im[3:9,4:11]=230
    gt=np.array([[4,3,11,9,1]],np.float32)
    for flip in (False,True):
        for k in range(4):
            image,b=spatial(im,gt,flip,k)
            yy,xx=np.where(image[...,0]>0)
            require(f"box is transformed with encoded pixels ({flip},{k})",
                    np.array_equal(b[0,:4],[xx.min(),yy.min(),xx.max()+1,yy.max()+1]) and b[0,4]==1)

    mats=[torch.arange(v,v+4).view(1,1,2,2).repeat(1,2,1,1) for v in (0,4,8)]
    require("pyramid level and row-major order",torch.equal(flatten_quality(mats)[0,1],torch.arange(12)))
    boxes=torch.tensor([[0,0,10,10],[0,0,4,10],[44,4,50,8],[20,0,30,10],[0,0,1.5,10],[60,60,68,70]],
                       dtype=torch.float32).repeat(2,1,1).requires_grad_(True)
    # The second frame is empty. Flat anchor IDs must not borrow frame 0's positives.
    ann=[torch.tensor([[0.,0,10,10,0],[20,0,30,10,1]]),torch.empty(0,5)]
    logits=torch.tensor([[-2.,-2.,1.,.5,-2.,-9.],[-2.,-2.,.1,-1.,-2.,-9.]]).repeat(2,1,1).requires_grad_(True)
    q=torch.zeros_like(logits,requires_grad=True)
    rows=assign(boxes,logits,ann)
    mapping={(t,c):(p.tolist(),border.tolist(),hard.tolist()) for t,c,p,border,hard,_,_ in rows}
    require("class0 genuine GT recovered",mapping[0,0][0]==[0])
    require("class1 genuine GT recovered",mapping[0,1][0]==[3])
    require("border regression has correct anchor and image",mapping[0,0][1]==[1])
    require("misclassified high-score other-class box is penalized",3 in mapping[0,0][2])
    require("ambiguous overlaps are not mined as background",4 not in mapping[0,0][2] and 1 not in mapping[0,0][2])
    require("low-score background filtered out",5 not in mapping[0,0][2])
    require("empty image only yields annotated negatives",not mapping[1,0][0] and not mapping[1,1][0] and 2 in mapping[1,0][2])
    terms=auxiliary(boxes,logits,q,ann)
    require("all three auxiliary terms live and finite",all(torch.isfinite(v) and v>0 for v in terms.values()))
    sum(terms.values()).backward()
    require("ranking raises low-score positive and suppresses high-score negative",
            logits.grad[0,0,0]<0 and logits.grad[0,0,2]>0 and q.grad[0,0,0]<0 and q.grad[0,0,2]>0)
    require("border supervision changes xyxy in the direction of own GT",boxes.grad[0,1,2]<0 and boxes.grad[1,1,2]==0)
    b=batch_targets(ann,(20,40),"cpu")
    require("native loss consumes exact class,batch,normalized xywh",
            torch.equal(b["cls"].flatten(),torch.tensor([0.,1.])) and torch.equal(b["batch_idx"],torch.zeros(2)) and
            torch.allclose(b["bboxes"],torch.tensor([[.125,.25,.25,.5],[.625,.25,.25,.5]])))
    require("constant quality leaves scores bit exact",torch.equal(fused_logits(logits,torch.full_like(q,1.333)),logits))

    model=QualityDetector().eval()
    rng=np.random.default_rng(SEED)
    examples=[train_pair(r,rng,224) for r in fit[:2]]
    x=pack([r[0] for r in examples],"cpu").float().div_(255)
    with torch.inference_mode():
        base=model.base(x)[0]
        levels=model.features(x)
        raw=model.head.forward_head(levels,**model.head.one2one)
        native=model.export_raw(raw,torch.full_like(raw["scores"],1.333))
        actual=model(x)
        require("all 300 production boxes/classes/scores identical at constant quality",torch.equal(base,native))
        require("fresh rank head exactly replicates reference detector",torch.equal(base,actual))
        # Promote a chosen class and anchor and locate it in the real top-k output.
        custom=torch.full_like(raw["scores"],-20.)
        custom[:,1,8]=5.;custom[:,0,3]=5.4
        zero=torch.zeros_like(custom);zero[:,1,8]=6
        ov=dict(raw,scores=custom)
        ranked=model.export_raw(ov,zero)
        decoded=model.head._get_decode_boxes(ov).transpose(1,2)
        require("production top-k indexes adjusted class+anchor across P3/P4/P5",
                torch.equal(ranked[:,0,:4],decoded[:,8,:]) and torch.all(ranked[:,0,5]==1))
    model.train()
    crit=E2ELoss(LossView(model.head,36))
    raw,q=model(x)
    total,all_terms=joint_loss(model,crit,raw,q,[p[1] for p in examples],(224,224))
    total.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
    require("real labeled images run combined native and quality loss/gradient on CPU",torch.isfinite(norm) and norm>0)
    require("native box-head and shared trunk have finite gradient",
            model.base.model[0].conv.weight.grad.abs().sum()>0 and model.head.one2one_cv2[0][-1].weight.grad.abs().sum()>0)
    quality_grad=sum(float(br[-1].weight.grad.abs().sum()) for br in model.quality)
    require("real training sends gradient into lightweight quality branch",quality_grad>0)
    bad=q.detach().clone();bad[0,0,0]=float("nan")
    try:
        joint_loss(model,crit,raw,bad,[p[1] for p in examples],(224,224))
    except FloatingPointError:
        passed.append("nonfinite objective rejected")
    else:raise AssertionError("nonfinite objective accepted")
    model.quality[0][-1].weight.grad.view(-1)[0]=float("nan")
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
    except RuntimeError:
        passed.append("nonfinite gradients rejected before optimizer update")
    else:raise AssertionError("nonfinite gradient accepted")
    additions=sum(p.numel() for p in model.quality.parameters())
    report=dict(passed=passed,passed_count=len(passed),fit_images=len(fit),calibration_images=len(cal),
                source_sha256=BASE_SHA,extra_parameters=additions,CPU_real_batch_size=2,CPU_real_size=224,
                quality_terminal_grad_abs_after_clip=quality_grad,CPU_real_loss={k:float(v.detach()) for k,v in all_terms.items()},
                detector_source_sha256=sha(__import__("defect_continue_rank_net").__file__),
                no_training_run_started=True)
    atomic_json(OUT/"cpu_check.json",report)
    print(report,flush=True)
