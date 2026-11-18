"""Transfer a reviewed location into existing native classifiers under feature-space preservation."""

import argparse
from collections import Counter, defaultdict
import hashlib
import math

import cv2
import numpy as np
from scipy.special import expit, logit
import torch

from audit_defect_causes_20261112 import ious, match
from class0_patch_refinement import ROOT, atomic_json, configure, load_image, read, sha
from defect_blind_review_pack import MANIFEST, OUT as PACK, native_gt
from defect_blind_review_update import base_network, replay, extract, Residual
from defect_continue_rank_net import flatten_quality, fused_logits
from defect_continue_scratch_net import boxes as deployment_boxes
from defect_review_point_behavior import OUT as AUDIT
from run_defect_zero_train_20261112 import metrics

OUT = ROOT/'experiments/defect_review_native_transfer_v1'
FROZEN = ROOT/'experiments/defect_blind_center_add_v1/protocol.json'
POST = '125f3b781628'
CUTS = [.40,.34]


def withheld_and_teacher():
    reviewed = read(AUDIT/'training_point_behavior.json')
    rcnn = {r['id']:r for r in read(AUDIT/'rcnn_training_points.json')['records']}
    cases = {r['id']:r for r in read(PACK/'组织者留存/source_key.json')['cases']}
    groups = {}
    for r in reviewed['records']:
        if (r['source'] == 'unmatched OOF candidate' and r['response']=='insect' and
                r['models']['specialist']['fixed_relation']=='no_detection' and
                rcnn[r['id']]['fixed_relation']=='same_class_only'):
            groups[r['id']] = cases[r['id']]['capture_group'][:-2]
    assert len(groups) == 6
    held = sorted(set(groups.values()),key=lambda g:hashlib.sha256(('native-probe-'+g).encode()).digest())[0]
    reserved = {held, cases[POST]['capture_group'][:-2]}
    teacher_fit = [uid for uid,group in groups.items() if group not in reserved]
    readers = {r['id']:r['response'] for r in reviewed['records']}
    return cases, readers, reserved, teacher_fit, [i for i,g in groups.items() if g in reserved]


def freeze():
    cases, responses, reserved, teaching, probes = withheld_and_teacher()
    train = [r for r in read(MANIFEST)['samples'] if r['partition']=='train']
    split = dict(read(FROZEN)['split'])
    for r in train:
        if r['same_second_group'][:-2] in reserved:
            assert split[r['image_id']]=='fit'
            split[r['image_id']]='reserved_minute'
    assert responses[POST]==''
    protocol = dict(architecture='Replace only existing last native one-to-one classifier coefficients for '
                    'each class/scale; scratch and insect both trainable, box regressions fixed.',
                    control='Human point chosen by highest native probability at local center.',
                    transfer='Identical target points and prior. In five teacher-agreed training fruit wounds '
                    'choose the anchor most aligned with the teacher footprint, only class supervision.',
                    point_export=sha(PACK/'组织者留存/source_key.json'),
                    answers_sha=read(AUDIT/'training_point_behavior.json')['export_sha'],
                    manifest=sha(MANIFEST),parent_scratch=read(replay.ROOT/'experiments/defect_continue_scratch_specialist/receipt.json')['weight_sha'],
                    teacher_sha=read(AUDIT/'rcnn_training_points.json')['teacher_sha'],
                    held_probes=sorted(set(probes+[POST])),teacher_fit=teaching,split=split,
                    counts=dict(Counter(split.values())),cuts=CUTS,
                    new_head_gate='Compared with native parent on 111 screen frames: >=1 net extra TP, '
                    'FP no larger, P no lower. Paired test-point queries kept independent of weight selection. '
                    'No fallback threshold. Parent saw all 596 train frames before specialization.',
                    loss='Separate convex weighted squared target logit increments at native final classifiers; '
                    'units normalized per feature column by RMS of stabilization anchors. '
                    'Original-score anchors 1.0, native true .08, overlapping wrong class .08, reviewed positive '
                    'and negative sites .15 each. L2 .03. Positive target probability .75 and negative .08.',
                    fit_seed=180413)
    path=OUT/'protocol.json'
    if path.exists() and read(path)!=protocol:
        raise ValueError('Protocol changed')
    atomic_json(path,protocol)
    return train, cases, responses, protocol


@torch.inference_mode()
def teacher_support(train, cases, protocol):
    from defect_continue_rcnn import initialized, load_checkpoint, SIDE, predict
    src=ROOT/'experiments/defect_continue_rcnn/transfer_v2'
    f=read(src/'frozen.json')['selected']
    assert f['sha256']==protocol['teacher_sha']
    model=initialized().cuda().eval()
    model.load_state_dict(load_checkpoint(src/'epoch18.pt')['model'])
    data={r['image_id']:r for r in train}
    boxes={}
    for uid in protocol['teacher_fit']:
        rec=cases[uid]; b=rec['box']; x,y=(np.asarray(b[:2])+b[2:])/2
        assert protocol['split'][rec['image']]=='fit'
        pred=predict(model,load_image(data[rec['image']]['rgb_path']))
        valid=pred[(pred[:,5]==1)&(pred[:,4]>=f['cut'][1])&(pred[:,0]<=x)&(x<=pred[:,2])&
                   (pred[:,1]<=y)&(y<=pred[:,3])]
        if not len(valid):
            raise ValueError('Missing teacher anchor support')
        ov=ious(valid[:,:4],[b])[:,0]
        j=ov.argmax()
        if ov[j]<.5:
            raise ValueError('Unclear spatial alignment of teacher')
        boxes[uid]=valid[j,:4]
    return boxes


@torch.inference_mode()
def dense(model, im):
    rgb, trans = replay.original.square(im)
    assert trans==(0,0,.75,.75)
    x=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
    fine, levels=model.locked.features(x)
    scratch=model.scratch
    sp=[levels[0]+scratch.fine(fine),*levels[1:]]
    warm=model.locked.warm
    q=flatten_quality([b(f) for b,f in zip(warm.quality,levels,strict=True)])
    out=[]
    for c, (head,pyramid) in enumerate([(scratch.head,sp),(warm.head,levels)]):
        feats=[tower[:-1](f) for tower,f in zip(head.one2one_cv3,pyramid,strict=True)]
        raw=head.forward_head(pyramid,**head.one2one)
        logits=raw['scores'] if c==0 else fused_logits(raw['scores'],q)
        raw_boxes=(head._get_decode_boxes(raw)[0].T/.75).clamp(0,1024)
        start=0; mapped=[]
        for level, (tower, feature) in enumerate(zip(head.one2one_cv3,feats,strict=True)):
            n=feature.shape[-2]*feature.shape[-1]
            # The linear coefficients changed by the solver are exactly the native last conv.
            terminal=tower[-1](feature)[0,c].flatten()
            if not torch.equal(terminal,raw['scores'][0,c,start:start+n]):
                raise ValueError('Native ROI anchor indexing differs')
            mapped.append((raw_boxes[start:start+n].cpu().numpy(), logits[0,c,start:start+n].cpu().numpy(),
                           feature[0].flatten(1).T))
            start+=n
        out.append(mapped)
    return out


def add(collection, name, level, f, z, ids, desired=None):
    if len(ids)==0:
        return
    ii=np.asarray(ids,dtype=np.int64)
    feature=f[torch.as_tensor(ii,device=f.device)].cpu().numpy()
    delta=np.zeros(len(ii),np.float32) if desired is None else np.broadcast_to(desired,(len(ii),))-z[ii]
    collection[(level,name)].append((feature,delta.astype(np.float32)))


def fit():
    train,cases,answers,protocol=freeze()
    if (OUT/'fit.json').exists():
        raise ValueError('Paired fit already completed')
    lookup={r['image']:r for r in cases.values() if r['source']=='unmatched OOF candidate'
            and answers[r['id']] in ('scratch','insect','other') and protocol['split'][r['image']]=='fit'}
    guide=teacher_support(train,cases,protocol)
    model=base_network()
    groups=defaultdict(list)
    rng=np.random.default_rng(protocol['fit_seed'])
    differences=[]
    for rec in train:
        if protocol['split'][rec['image_id']]!='fit':
            continue
        im=load_image(rec['rgb_path']); features=dense(model,im)
        gt=native_gt(rec); g=np.array([r['box'] for r in gt]).reshape(-1,4)
        cls=np.array([r['class'] for r in gt],dtype=int)
        point=lookup.get(rec['image_id'])
        for c, levels in enumerate(features):
            native_targets={}
            weak_candidates=[]
            for lev,(box,z,f) in enumerate(levels):
                j=(c,lev)
                top=np.argpartition(-z,min(55,len(z)-1))[:56]
                random=rng.choice(len(z),40,replace=False)
                idx=np.unique(np.concatenate((top,random)))
                if point is not None:
                    pb=np.asarray(point['box'])
                    cx,cy=(pb[:2]+pb[2:])/2
                    intersects=(box[:,0]<=cx)&(cx<=box[:,2])&(box[:,1]<=cy)&(cy<=box[:,3])
                    idx=idx[~intersects[idx]]
                    site=ious(box,[pb])[:,0]
                    positive=np.flatnonzero(intersects&(site>=.15))
                    if len(positive) and len(gt):
                        clean=ious(box[positive],g).max(1)<.10
                        positive=positive[clean]
                    for a in positive:
                        weak_candidates.append((lev,int(a),float(z[a]),float(site[a])))
                add(groups,'anchor',j,f,z,idx)
                if len(gt):
                    ov=ious(box,g)
                    for gi,category in enumerate(cls):
                        ids=np.flatnonzero(ov[:,gi]>=.5)
                        if not len(ids):continue
                        a=ids[z[ids].argmax()]
                        row=(z[a],lev,a)
                        if gi not in native_targets or row[0]>native_targets[gi][0]:
                            native_targets[gi]=row
            for gi,(log,lev,a) in native_targets.items():
                box,z,f=levels[lev]
                positive=cls[gi]==c
                # Correct classes only require floor support. Confident native predictions need no upward drift.
                goal=max(log,logit(.60)) if positive else min(log,logit(.08))
                add(groups,'native_positive' if positive else 'native_confusion',(c,lev),f,z,[a],goal)
            if point is None or not weak_candidates:continue
            target={'scratch':0,'insect':1,'other':-1}[answers[point['id']]]
            best=max(weak_candidates,key=lambda v:(v[2],v[3]))
            chosen=best
            if c==1 and point['id'] in guide:
                choices=np.asarray([levels[v[0]][0][v[1]] for v in weak_candidates])
                ov=ious(choices,[guide[point['id']]])[:,0]
                # Choose high-confidence native anchor among near-best teacher-overlap anchors.
                pool=[v for v,o in zip(weak_candidates,ov) if o>=max(.50,float(ov.max())-.06)]
                if pool:
                    chosen=max(pool,key=lambda v:v[2])
                differences.append(dict(case=point['id'],support_candidates=len(choices),best_teacher_iou=float(ov.max()),
                                        control_anchor=list(best[:2]),teacher_anchor=list(chosen[:2])))
            for tag,row in [('control',best),('transfer',chosen)]:
                lev,a,log,_=row
                b,z,f=levels[lev]
                positive=c==target
                desired=max(log,logit(.75)) if positive else min(log,logit(.08))
                add(groups,tag+('_positive' if positive else '_negative'),(c,lev),f,z,[a],desired)
    evidence=dict(teacher=[dict(case=i,box=list(map(float,b))) for i,b in guide.items()],
                  selected=differences,distinct_pairs=sum(v['control_anchor']!=v['teacher_anchor'] for v in differences))
    atomic_json(OUT/'anchor_evidence.json',evidence)
    # There is no teacher-specific experiment if its geometry selects the same native candidates.
    if evidence['distinct_pairs']<2:
        atomic_json(OUT/'fit.json',dict(status='insufficient_teacher_specific_support',**evidence))
        print('Gate before fit: only',evidence['distinct_pairs'],'different anchor assignments');return
    for mode in ('control','transfer'):
        changes={}
        for c in range(2):
            for lev in range(3):
                j=(c,lev)
                def matrix(k):
                    if not groups[j,k]:return None,None
                    fs,zs=zip(*groups[j,k]); x=np.concatenate(fs); y=np.concatenate(zs)
                    return np.column_stack([x,np.ones(len(x))]).astype(np.float64),y
                anchor,_=matrix('anchor')
                rms=np.sqrt(np.mean(anchor**2,axis=0)).clip(.05,None)
                norm=anchor/rms
                A=norm.T@norm/len(norm)+.03*np.eye(len(rms)); b=np.zeros(len(rms))
                for label,weight in [('native_positive',.08),('native_confusion',.08),
                                     (mode+'_positive',.15),(mode+'_negative',.15)]:
                    raw,tar=matrix(label)
                    if raw is None:continue
                    norm=raw/rms
                    A+=weight*(norm.T@norm)/len(norm); b+=weight*(norm.T@tar)/len(norm)
                changes[j]=torch.from_numpy((np.linalg.solve(A,b)/rms).astype(np.float32))
        folder=OUT/mode;folder.mkdir(exist_ok=True)
        replay.atomic_latest(folder/'weights.pt',changes)
        atomic_json(folder/'receipt.json',dict(weight_sha=sha(folder/'weights.pt'),thresholds=CUTS,
                    upstream_weights='unchanged scratch specialist and locked quality detector',
                    method='one closed-form regularized update of six terminal native convolutional rows'))
    atomic_json(OUT/'fit.json',dict(status='trained',native_counts={str(k):sum(len(z) for _,z in v) for k,v in groups.items()},**evidence))
    print('Trained transfer with',evidence['distinct_pairs'],'distinct teacher-selected native anchors',flush=True)


def model_for(mode):
    m=base_network()
    if mode=='parent':return m
    folder=OUT/mode
    assert sha(folder/'weights.pt')==read(folder/'receipt.json')['weight_sha']
    changes=torch.load(folder/'weights.pt',weights_only=True,map_location='cpu')
    with torch.no_grad():
        for (c,lev),delta in changes.items():
            head=m.scratch.head if c==0 else m.locked.warm.head
            conv=head.one2one_cv3[lev][-1]
            conv.weight[c,:,0,0].add_(delta[:-1].to(conv.weight))
            conv.bias[c].add_(delta[-1].to(conv.bias))
    return m


@torch.inference_mode()
def screen():
    train,cases,responses,proto=freeze()
    selected=[r for r in train if proto['split'][r['image_id']]=='screen']
    assert len(selected)==111
    results={}
    for name in ('parent','control','transfer'):
        model=model_for(name); data=[]
        for rec in selected:
            pred=deployment_boxes(model,load_image(rec['rgb_path']))
            keep=[p.tolist() for p in pred if p[4]>=CUTS[int(p[5])]]
            data.append(dict(image_id=rec['image_id'],pred=keep,GT=native_gt(rec)))
        results[name]=metrics(data)
    p=results['parent']['total']; t=results['transfer']['total']
    passed=t['TP']>p['TP'] and t['FP']<=p['FP'] and t['P']>=p['P']
    atomic_json(OUT/'gate.json',dict(passed=passed,results=results))
    print('Gate',passed,{n:r['total'] for n,r in results.items()},flush=True)


@torch.inference_mode()
def probe():
    train,cases,responses,proto=freeze()
    raw={r['image_id']:r for r in train}
    probes=proto['held_probes']
    methods=['parent']
    fit=read(OUT/'fit.json')
    if fit['status']=='trained':
        methods+=['control','transfer']
    summary={}
    for uid in probes:
        c=cases[uid]
        assert proto['split'][c['image']]=='reserved_minute'
        im=load_image(raw[c['image']]['rgb_path']); x,y=(np.array(c['box'][:2])+c['box'][2:])/2
        d={}
        def at_point(p):
            b=p[(p[:,0]<=x)&(x<=p[:,2])&(p[:,1]<=y)&(y<=p[:,3])]
            return {str(i):max((float(v[4]) for v in b if v[5]==i),default=None) for i in (0,1)}
        for name in methods:
            d[name]=at_point(deployment_boxes(model_for(name),im))
        # The already frozen residual pair provides the actual scores on the later answered point.
        m=base_network()
        p,features=extract(m,im)
        for name in ['control','review']:
            f=ROOT/'experiments/defect_blind_review_update_v1'/name
            r=read(f/'receipt.json');h=Residual(r['feature_dim']).cuda().eval()
            h.load_state_dict(torch.load(f/'weights.pt',map_location='cpu',weights_only=True))
            d['previous_residual_'+name]=at_point(h(features,torch.from_numpy(p).cuda()).cpu().numpy())
        summary[uid]=d
    atomic_json(OUT/'held_point_probe.json',dict(points=summary,scope='Point coverage only; full-minute exclusion '
               'from fitting the native modifications and from screening. The historical parent saw source images.'))
    print(summary,flush=True)


def evaluate(name):
    if not read(OUT/'gate.json')['passed']:
        raise ValueError('Frozen training-only precision/recall gate failed')
    original=replay.original
    old=original.deploy_model,original.boxes
    try:
        source=lambda:model_for(name)
        original.deploy_model,original.boxes=source,deployment_boxes
        with replay.evaluation_output(OUT/name):original.evaluate()
    finally:original.deploy_model,original.boxes=old


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze','fit','screen','probe','control','transfer'])
    a=parser.parse_args()
    configure(OUT,'cpu' if a.action=='freeze' else 'cuda')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    with replay.exclusive_run(OUT):
        if a.action in ('control','transfer'): evaluate(a.action)
        else: {'freeze':freeze,'fit':fit,'screen':screen,'probe':probe}[a.action]()
