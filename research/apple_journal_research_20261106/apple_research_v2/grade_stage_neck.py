"""Stage-neck appearance grader; single scene feature pass and frozen native-320 severity."""
from contextlib import ExitStack
from pathlib import Path
from copy import deepcopy
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

import afruit_grade_model as grade
import afruit_grade_train as io
import afruit_visible_quality as visible
from matched_grade_loader import install

ROOT=grade.ROOT
OUT=ROOT/"experiments/grade_pairs_20261119/stage_neck_joint_v1"
PARENT=ROOT/"experiments/grade_error49_v1/border_native320_woo_ref"
WOO=ROOT/"datasets/apple_matched_quality_v1/manifest.json"
GRID=10


def content_rect(hw,net_hw):
    h,w=hw;nh,nw=net_hw
    gain=min(nh/h,nw/w);sx,sy=round(w*gain),round(h*gain)
    left,top=round((nw-sx)/2-.1),round((nh-sy)/2-.1)
    return left,top,left+sx,top+sy


def map_boxes(boxes,hw,net_hw):
    l,t,r,b=content_rect(hw,net_hw);h,w=hw
    return [[l+x*(r-l)/w,t+y*(b-t)/h,l+u*(r-l)/w,t+v*(b-t)/h] for x,y,u,v in boxes]


def same_response(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_response(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same_response(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return bool(np.isclose(a,b,rtol=2e-6,atol=2e-6))
    return a==b


class NeckGrade(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.project=nn.ModuleList([nn.Sequential(nn.Conv2d(c,16,1),nn.GroupNorm(4,16),nn.SiLU()) for c in channels])
        self.mark=nn.Sequential(nn.Conv2d(48,24,1),nn.SiLU(),nn.Conv2d(24,1,1))
        self.readout=nn.Sequential(nn.LayerNorm(48),nn.Linear(48,16),nn.SiLU(),nn.Linear(16,1))
        self.channels=channels

    def forward(self,grid,support,parent):
        pieces=grid.split(self.channels,1)
        features=torch.cat([layer(x) for layer,x in zip(self.project,pieces,strict=True)],1)
        mask=(support>=.70)
        mask=torch.where(mask.sum((2,3),keepdim=True)>=3,mask,torch.ones_like(mask))
        # Keep spatial responses on the native visible fruit when an assigned mask exists.
        descriptor=(features*mask).sum((2,3))/mask.sum((2,3))
        heat=self.mark(features).masked_fill(~mask,-30.)
        peak=heat.flatten(1).topk(3,1).values.mean(1)
        evidence=self.readout(descriptor)[:,0]+peak
        margin=.35*(parent[:,1]-parent[:,0])+evidence
        severe_parent=parent.argmax(1)==2
        proposed=torch.stack((parent[:,0],parent[:,0]+margin,torch.minimum(parent[:,0],parent[:,0]+margin)-1),1)
        return torch.where(severe_parent[:,None],parent,proposed),margin,heat


class Runtime:
    def __init__(self,selection=None):
        self.pipe=visible.QualityPipeline()
        install(self.pipe,PARENT)
        self.shapes={}
        self.head=None
        if selection is not None:
            folder=Path(selection)
            r=io.read(folder/"selection.json")
            if r["weights_sha256"]!=grade.file_sha(folder/"weights.pt") or r["native320_sha256"]!=grade.file_sha(PARENT/"weights.pt") or r["stage_sha256"]!=visible.STAGE_WEIGHT_SHA:
                raise ValueError("Stage grade weights changed")
            self.head=NeckGrade(r["channels"]).cuda().eval()
            self.head.load_state_dict(torch.load(folder/"weights.pt",map_location="cpu",weights_only=True))

    def check_geometry(self,frame,net_hw):
        hw=frame.shape[:2]
        key=(*hw,*net_hw)
        if key in self.shapes:return
        dummy=np.full_like(frame,11)
        render=self.pipe.predictor.pre_transform([dummy])[0]
        region=cv2.boundingRect(((render==11).all(2)).astype(np.uint8))
        x,y,w,h=region
        expected=content_rect(hw,net_hw)
        if tuple((x,y,x+w,y+h))!=expected or tuple(render.shape[:2])!=net_hw:
            raise ValueError(f"Native letterboxing disagrees: {hw}, {net_hw}, {region}, expected {expected}")
        self.shapes[key]=expected

    @torch.inference_mode()
    def observe(self,frame,known=True,verify=False,both=False):
        if known and both:raise ValueError("Use scene mode to collect both training views")
        p=self.pipe
        stage=[];received=[];native=[]
        def in_shape(mod,args):
            received.append(tuple(args[0].shape[-2:]))
        def features(mod,args):
            stage.append(tuple(args[0]))
        def old_scores(mod,args,result):
            native.append(result.detach())
        with ExitStack() as hooks:
            for handle in (p.stage.model.trunk_for("stage")[0].register_forward_pre_hook(in_shape),
                           p.stage.model.heads["stage"].register_forward_pre_hook(features),
                           p.grade.register_forward_hook(old_scores)):
                hooks.callback(handle.remove)
            result,masks=p(frame,known_single_fruit=known)
        if verify:
            raw,a=p(frame,known_single_fruit=known)
            if not same_response(raw,result) or not np.array_equal(a,masks) or not np.array_equal(visible.overlay(frame,raw,a),visible.overlay(frame,result,masks)):
                raise ValueError("Read-only hooks affected frozen scene outputs")
        if len(stage)!=1 or len(received)!=1 or len(stage[0])!=3:raise ValueError("Unexpected number of stage forwards")
        hw=frame.shape[:2];nh=received[0];self.check_geometry(frame,nh)
        owners=[] if known else [v["fruit_id"] for v in result["fruits"] if v["crop"] is not None]
        boxes=[result["fruits"][i]["crop"] for i in owners]
        support=[masks[i] for i in owners]
        if both:native.append(p.grade(p.preprocess_grade([frame],"cuda")))
        if known or both:
            boxes.append([0,0,hw[1],hw[0]])
            support.append(masks[0] if len(masks)==1 else None)
        if len(boxes)!=result["grade_inference_items"]+int(both):raise ValueError("Parent crop/batch order changed")
        if not boxes:return result,masks,owners,None
        dev=stage[0][0].device
        mapped=map_boxes(boxes,hw,nh)
        rois=torch.tensor([[0,*box] for box in mapped],device=dev,dtype=torch.float32)
        grids=[];cs=[]
        for level,f in enumerate(stage[0]):
            stride=nh[0]/f.shape[-2]
            if stride!=nh[1]/f.shape[-1] or stride!=8*2**level:raise ValueError("Unexpected FPN footprint")
            grids.append(roi_align(f.float(),rois,(GRID,GRID),spatial_scale=1/stride,sampling_ratio=2,aligned=True))
            cs.append(f.shape[1])
        # Area-resizing the correctly aligned native mask to the ROI grid avoids transferring full camera masks to the GPU.
        coverage=[]
        for mask,(l,t,r,b) in zip(support,boxes,strict=True):
            if mask is None:coverage.append(np.ones((GRID,GRID),np.float32));continue
            coverage.append(cv2.resize(mask[t:b,l:r].astype(np.float32),(GRID,GRID),interpolation=cv2.INTER_AREA))
        return result,masks,owners,(torch.cat(grids,1),torch.tensor(np.stack(coverage)[:,None],device=dev),torch.cat(native),cs)

    @torch.inference_mode()
    def __call__(self,frame,known_single_fruit=False):
        if self.head is None:raise ValueError("Load a frozen scene-neck head")
        result,masks,owners,pack=self.observe(frame,known=known_single_fruit)
        if pack is None:return result,masks
        grids,support,parent,cs=pack
        logits,_,_=self.head(grids,support,parent);labels=logits.argmax(1).tolist()
        probs=logits.softmax(1).tolist()
        grades=[dict(grade=visible.GRADES[a],grade_id=a,grade_probabilities=b) for a,b in zip(labels,probs)]
        for i,v in zip(owners,grades):result["fruits"][i].update(v)
        if known_single_fruit:
            result["known_single_fruit_grade"]=grades[-1]
            if len(result["fruits"])==1:result["fruits"][0].update(grades[-1])
        return result,masks


