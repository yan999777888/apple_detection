"""Matched native classification-only updates with or without the completed 21 weak judgments."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.special import logit
import torch

from audit_defect_causes_20261112 import ious
from class0_patch_refinement import ROOT, atomic_json, configure, load_image, read, sha
from defect_blind_review_pack import MANIFEST, native_gt
from defect_blind_review_update import SOURCE_SHA, base_network, replay
from defect_continue_scratch_net import WARM_SHA
from defect_review_completed_points import FINAL
from defect_review_native_transfer import CUTS, FROZEN, dense, add, deployment_boxes
from defect_review_point_behavior import point_evidence, relation
from run_defect_zero_train_20261112 import metrics

OUT = ROOT/'experiments/defect_review_terminal_points_v1'
ARMS = ('control','review')
GROUP_WEIGHTS = dict(anchor=1., high_anchor=1., native_pos=.08, native_neg=.08,
                     weak_pos=.15, weak_neg=.15)
SEED = 180413


def protocol():
    reviewed=read(FINAL)
    assert reviewed['answered']==34 and reviewed['export_answered']==33
    assert reviewed['supplement']['id']=='125f3b781628'
    weak={r['image_id']:r for r in reviewed['rows'] if r['kind']=='unmatched' and
          r['manual'] in ('other','insect','scratch')}
    assert Counter(r['manual'] for r in weak.values())=={'other':3,'scratch':5,'insect':13}
    split=read(FROZEN)['split']
    train=[r for r in read(MANIFEST)['samples'] if r['partition']=='train']
    assert set(split)=={r['image_id'] for r in train} and set(split.values())=={'fit','screen'}
    assert len(weak)==21 and all(split[k]=='fit' for k in weak)
    all_reviewed={r['image_id'] for r in reviewed['rows']}
    assert all(split[k]=='fit' for k in all_reviewed)
    mins={tag:{r['same_second_group'][:-2] for r in train if split[r['image_id']]==tag} for tag in ('fit','screen')}
    assert mins['fit'].isdisjoint(mins['screen'])
    p=dict(schema=1,split_counts=dict(Counter(split.values())),split_sha=sha(FROZEN),
           native_manifest=sha(MANIFEST),completed_reader_audit=sha(FINAL),
           earlier_export=reviewed['export_sha256'],supplement=reviewed['supplement']['sha256'],
           raw_reader_targets=dict(Counter(r['manual'] for r in weak.values())),
           parent_warm=WARM_SHA,parent_scratch=SOURCE_SHA,source_sha=sha(Path(__file__)),
           upstream_dense_sha=sha(Path(__file__).with_name('defect_review_native_transfer.py')),
           arms='Both minimize native GT logit margins and original-score drift on identical class/scale rows. '
           'Only A1 adds point classification loss. All GT boxes and regressor weights fixed. '
           'The teacher is not used in either arm.', thresholds=CUTS,
           model='Six final one-to-one classification Conv2d rows and biases. No extra inference op.',
           fit='Closed form ridge with uncentered column RMS and coefficient penalty 0.03. '
           'Target native confidence floors .60/.08, weak .75/.08, no update required beyond target. '
           'Preserve anchors include exact parent logits for top56 and seeded 40 random per level '
           'and all native fixed-threshold dense anchors in the additional high-anchor term.',
           objective_weights=GROUP_WEIGHTS,seed=SEED,
           weak_rule='Within original sampler support select strongest native same-branch hypothesis at the '
           'human-indicated pixel, centered within 0.55 of proposal extent and overlap >=.15 with the '
           'sampling proposal for identity only. Reject if any native GT IoU >=.10. Largest native '
           'probability across scales selects one row per class per reviewed site; no weak box supervision.',
           screen='Preexisting, reused 111 same-training-date frames held out by capture minute from these '
           'terminal updates; parent trained on all source frames. Must jointly add >=2 TP and no extra FP '
           'versus BOTH parent and A0; insect TP >= original parent. No threshold search. '
           'On failure stop before development inference/speed. Point replay is training behavior only.')
    dst=OUT/'protocol.json'
    if dst.exists() and read(dst)!=p:
        raise RuntimeError('Frozen experiment changed')
    atomic_json(dst,p)
    return train,split,reviewed,weak


def store_native(c, levels, boxes, truth, groups, rng, site):
    native, weak = {},[]
    for lev,(b,z,feat) in enumerate(levels):
        key=(c,lev)
        idx=np.unique(np.r_[np.argpartition(-z,55)[:56],rng.choice(len(z),40,replace=False)])
        high=np.flatnonzero(z>=logit(CUTS[c]))
        add(groups,'anchor',key,feat,z,idx)
        add(groups,'high_anchor',key,feat,z,high)
        ov=ious(b,boxes) if len(boxes) else None
        if ov is not None:
            for j in range(len(truth)):
                ids=np.flatnonzero(ov[:,j]>=.50)
                if len(ids):
                    a=int(ids[z[ids].argmax()])
                    if j not in native or z[a]>native[j][0]:
                        native[j]=(float(z[a]),lev,a)
        if site is not None:
            center=np.asarray(site['center_xy'])
            orig=np.asarray(site['box_xyxy']);size=(orig[2:]-orig[:2]).clip(1)
            inside=np.all((b[:,:2]<=center)&(center<=b[:,2:]),axis=1)
            within=np.max(np.abs((b[:,:2]+b[:,2:])/2-center)/size,axis=1)<=.55
            ids=np.flatnonzero(inside & within)
            if len(ids):
                identity=ious(b[ids],[orig])[:,0]
                ok=identity>=.15
                if ov is not None:ok &= ov[ids].max(1)<.10
                for a,v in zip(ids[ok],identity[ok]):
                    weak.append((float(z[a]),float(v),lev,int(a)))
    for j,(s,lev,a) in native.items():
        _,z,f=levels[lev];positive=truth[j]==c
        target=max(s,logit(.60)) if positive else min(s,logit(.08))
        add(groups,'native_pos' if positive else 'native_neg',(c,lev),f,z,[a],target)
    if weak:
        s,_,lev,a=max(weak);_,z,f=levels[lev]
        positive=site['manual']==('scratch','insect')[c]
        target=max(s,logit(.75)) if positive else min(s,logit(.08))
        add(groups,'weak_pos' if positive else 'weak_neg',(c,lev),f,z,[a],target)
        return dict(cls=c,positive=positive,selected=[lev,a],eligible=len(weak),
                    initial_logit=s,contains_native_GTs=False,response=site['manual'])
    return None


def solve(groups,mode):
    weights={}
    for c in (0,1):
        for lev in range(3):
            key=(c,lev)
            def matrix(label):
                if not groups[key,label]:
                    return None,None
                x,y=zip(*groups[key,label])
                x=np.concatenate(x);y=np.concatenate(y)
                return np.column_stack([x,np.ones(len(y))]).astype(np.float64),y
            A,b=None,None
            anchor,_=matrix('anchor')
            rms=np.maximum(.05,np.sqrt(np.mean(anchor**2,axis=0)))
            for label,w in GROUP_WEIGHTS.items():
                if label.startswith('weak') and mode=='control':
                    continue
                X,Y=matrix(label)
                if X is None:
                    continue
                U=X/rms
                if A is None:
                    A=.03*np.eye(len(rms));b=np.zeros(len(rms))
                A+=w*(U.T@U)/len(U);b+=w*(U.T@Y)/len(U)
            delta=np.linalg.solve(A,b)/rms
            if not np.all(np.isfinite(delta)):
                raise FloatingPointError('Nonfinite terminal update')
            weights[key]=torch.tensor(delta,dtype=torch.float32)
    return weights


def fit():
    rows,split,_,weak=protocol()
    if (OUT/'fit.json').exists():raise RuntimeError('Weights already fitted')
    model=base_network();rng=np.random.default_rng(SEED);groups=defaultdict(list);coverage=[]
    for r in rows:
        if split[r['image_id']]!='fit':
            continue
        gt=native_gt(r);truth=np.asarray([v['class'] for v in gt],int)
        boxes=np.asarray([v['box'] for v in gt]).reshape(-1,4)
        site=weak.get(r['image_id'])
        dense_rows=dense(model,load_image(r['rgb_path']))
        evidence=[]
        for c,levels in enumerate(dense_rows):
            signal=store_native(c,levels,boxes,truth,groups,rng,site)
            if signal:evidence.append(signal)
        if site:
            coverage.append(dict(id=site['id'],position=site['position'],manual=site['manual'],signals=evidence))
    assert len(coverage)==21
    for mode in ARMS:
        where=OUT/mode;where.mkdir(exist_ok=True)
        delta=solve(groups,mode)
        replay.atomic_latest(where/'weights.pt',delta)
        atomic_json(where/'receipt.json',dict(mode=mode,weight_sha=sha(where/'weights.pt'),
                    base_sha=[WARM_SHA,SOURCE_SHA],thresholds=CUTS,affine_updates=6,
                    summary='Apply learned deltas to original terminal rows for deployment',
                    protocol_sha=sha(OUT/'protocol.json')))
    stats=Counter()
    for v in coverage:
        tag=v['manual']; stats[f'{tag}/sites']+=1
        stats[f'{tag}/at_least_one_branch']+=bool(v['signals'])
        if tag!='other':stats[f'{tag}/positive_same_class']+=any(w['positive'] for w in v['signals'])
        else:stats['other/negative_constraints']+=len(v['signals'])
    atomic_json(OUT/'fit.json',dict(site_coverage=stats,coverage=coverage,counts={
        str(k):sum(len(x) for _,x in v) for k,v in groups.items()}))
    print('Weak-point availability',stats,flush=True)


def model_for(mode):
    m=base_network()
    if mode=='parent':return m
    where=OUT/mode;receipt=read(where/'receipt.json')
    assert sha(where/'weights.pt')==receipt['weight_sha'] and receipt['protocol_sha']==sha(OUT/'protocol.json')
    with torch.no_grad():
        for (c,lev),d in torch.load(where/'weights.pt',weights_only=True,map_location='cpu').items():
            h=m.scratch.head if c==0 else m.locked.warm.head
            conv=h.one2one_cv3[lev][-1]
            conv.weight[c,:,0,0].add_(d[:-1].to(conv.weight))
            conv.bias[c].add_(d[-1].to(conv.bias))
    return m


@torch.inference_mode()
def screen_and_points():
    rows,split,audit,weak=protocol()
    raw={r['image_id']:r for r in rows}
    screen_rows=[r for r in rows if split[r['image_id']]=='screen']
    assert len(screen_rows)==111
    stats={};counts={};ledger={}
    for mode in ('parent',*ARMS):
        m=model_for(mode); measured=[]; train_counts=Counter(); train_detailed=[]
        for r in screen_rows:
            full=deployment_boxes(m,load_image(r['rgb_path']))
            keep=[p.tolist() for p in full if p[4]>=CUTS[int(p[5])]]
            measured.append(dict(image_id=r['image_id'],pred=keep,GT=native_gt(r)))
        stats[mode]=metrics(measured)
        for s in audit['rows']:
            full=deployment_boxes(m,load_image(raw[s['image_id']]['rgb_path']))
            hits=point_evidence(full,s['box_xyxy'],s['center_xy'])
            kind=relation(s['manual'],hits)
            train_counts[f"{s['kind']}/{s['manual']}/{kind}"]+=1
            train_detailed.append(dict(id=s['id'],state=kind,score_by_class={
                tag:max((p['score'] for p in hits if p['cls']==tag),default=None) for tag in ('scratch','insect')}))
        counts[mode]=train_counts;ledger[mode]=train_detailed
    result=stats['review']
    tests={}
    for mode in ('parent','control'):
        reference=stats[mode]['total']
        tests['vs_'+mode]=dict(extra_tp=result['total']['TP']-reference['TP'],
                               extra_fp=result['total']['FP']-reference['FP'])
    insect=result['class']['1']['TP']-stats['parent']['class']['1']['TP']
    passed=all(r['extra_tp']>=2 and r['extra_fp']<=0 for r in tests.values()) and insect>=0
    atomic_json(OUT/'gate.json',dict(passed=passed,comparisons=tests,insect_tp_delta_to_parent=insect,
                scope='111 reused within-training-date screens only',measurements=stats))
    atomic_json(OUT/'reviewed_points.json',dict(scope='Seen training sites, not independent evaluation',
                counts=counts,ledger=ledger))
    print('Train-date gate',passed,tests,insect,{k:v['total'] for k,v in stats.items()},flush=True)


def evaluate(mode):
    if not read(OUT/'gate.json')['passed']:
        raise ValueError('Development evaluation did not qualify')
    old=replay.original.deploy_model,replay.original.boxes
    try:
        replay.original.deploy_model,replay.original.boxes=lambda:model_for(mode),deployment_boxes
        with replay.evaluation_output(OUT/mode):replay.original.evaluate()
    finally:replay.original.deploy_model,replay.original.boxes=old


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['freeze','fit','screen','control','review'])
    action=ap.parse_args().action
    configure(OUT,'cpu' if action=='freeze' else 'cuda')
    cv2.setNumThreads(1);torch.set_num_threads(1)
    with replay.exclusive_run(OUT):
        if action in ARMS:evaluate(action)
        else:{'freeze':protocol,'fit':fit,'screen':screen_and_points}[action]()
