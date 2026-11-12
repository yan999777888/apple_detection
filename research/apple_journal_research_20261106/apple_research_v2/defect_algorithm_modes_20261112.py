"""Explicit head/view ablations; no checkpoint or GT edits.
WBF-style clustering is class-aware IoU>.55, score-weighted coordinates,
mean member confidence times unique-source support / source count. No merging
of disjoint adjacent wounds. This policy is fixed, not tuned to dev errors.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'apple_research_v2/joint'))
import numpy as np,torch
from torchvision.ops import batched_nms
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from deployment_adapter_v1 import FrozenDeploymentAdapter
MODES=('o2o_nms','o2m_nms','heads_wbf_nms','o2o_wbf','flip_nms','flip_wbf')

def nms(p):
    p=np.asarray(p,dtype=np.float32).reshape(-1,6)
    if not len(p):return p
    x=torch.from_numpy(p);ids=batched_nms(x[:,:4],x[:,4],x[:,5],.7)[:300]
    return p[ids.numpy()]

def wbf(sources):
    candidates=sorted([(r.copy(),si) for si,p in enumerate(sources) for r in p],key=lambda x:-float(x[0][4]));clusters=[]
    for row,si in candidates:
        best=-1;overlap=.55
        for i,c in enumerate(clusters):
            if c['class']!=row[5]:continue
            b=c['box'];inter=np.maximum(np.minimum(row[2:4],b[2:])-np.maximum(row[:2],b[:2]),0).prod()
            union=np.prod(row[2:4]-row[:2])+np.prod(b[2:]-b[:2])-inter
            iou=float(inter/max(union,1e-9))
            if iou>overlap:best=i;overlap=iou
        if best<0:clusters.append({'class':row[5],'box':row[:4].copy(),'rows':[row],'sources':{si}})
        else:
            c=clusters[best];c['rows'].append(row);c['sources'].add(si);r=np.asarray(c['rows']);c['box']=np.average(r[:,:4],axis=0,weights=r[:,4])
    return np.asarray([np.r_[c['box'],np.mean([r[4] for r in c['rows']])*len(c['sources'])/len(sources),c['class']] for c in clusters],dtype=np.float32).reshape(-1,6)

def combine(sources,mode,thresholds=(.25,.25)):
    if mode=='o2o_nms':p=nms(sources['o2o'])
    elif mode=='o2m_nms':p=nms(sources['o2m'])
    elif mode=='heads_wbf_nms':p=nms(wbf([nms(sources['o2o']),nms(sources['o2m'])]))
    elif mode=='o2o_wbf':p=wbf([sources['o2o']])
    elif mode=='flip_nms':p=nms(np.concatenate([sources['o2o'],sources['flip']]))
    elif mode=='flip_wbf':p=nms(wbf([nms(sources['o2o']),nms(sources['flip'])]))
    else:raise ValueError(mode)
    if len(p):p=p[p[:,4]>=np.asarray(thresholds)[p[:,5].astype(int)]];p=p[np.argsort(-p[:,4],kind='stable')][:300]
    return p

class AlgorithmAdapter(FrozenDeploymentAdapter):
    @torch.inference_mode()
    def sources(self,tensor,hw,heads=('o2o',),flip=False):
        x=torch.cat([tensor,tensor.flip(-1)],0) if flip else tensor
        features=self.model.features(x,'defect');head=self.model.heads['defect'];out={}
        for name in heads:
            branch=head.one2one if name=='o2o' else head.one2many
            if branch['box_head'] is None:raise ValueError('Missing trained '+name+' branch')
            raw=head.forward_head(features,**branch);decoded=head._inference(raw).permute(0,2,1)
            if name=='o2o':pred=head.postprocess(decoded)
            for vi in range(len(x)):
                if vi and name=='o2m':continue  # only original o2m in this fixed ablation
                if name=='o2o':rows=pred[vi];rows=rows[rows[:,4]>=.001]
                else:
                    boxes=decoded[vi,:,:4];scores=decoded[vi,:,4:];ii,cc=torch.where(scores>=.001)
                    rows=torch.cat([boxes[ii],scores[ii,cc,None],cc[:,None].float()],1)
                rows=rows.clone()
                if vi:
                    xx=rows[:,0].clone();rows[:,0]=640-rows[:,2];rows[:,2]=640-xx
                ops.scale_boxes((640,640),rows[:,:4],hw)
                # Dense o2m has no top300-before-NMS cap: preserve all .001 candidates.
                p=rows.cpu().numpy()
                out['flip' if vi and name=='o2o' else name]=nms(p) if name=='o2m' else p
        return out

    @torch.inference_mode()
    def predict_mode(self,image,mode='o2o_nms',thresholds=(.25,.25),both=True):
        tensor=self.predictors['stage'].preprocess([image]);assert tensor.shape==(1,3,640,640)
        heads=('o2o','o2m') if mode=='heads_wbf_nms' else ('o2m',) if mode=='o2m_nms' else ('o2o',)
        sources=self.sources(tensor,image.shape[:2],heads,mode.startswith('flip'))
        p=combine(sources,mode,thresholds);result={'defect':[Results(image,'frame',self.model.names['defect'],boxes=torch.from_numpy(p))]}
        if both:
            raw=self.model.predict_source(tensor,'stage');predictor=self.predictors['stage'];predictor.batch=(['frame'],[image],[''])
            result['stage']=[r.cpu() for r in predictor.postprocess(raw,tensor,[image])]
        return result
