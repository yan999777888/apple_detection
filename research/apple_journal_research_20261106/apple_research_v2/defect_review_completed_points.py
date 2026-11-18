"""Evaluate frozen GPU inference at the final 34 locations including the separately answered point."""

from collections import Counter, defaultdict

import cv2
import numpy as np
import torch

from class0_patch_refinement import ROOT, atomic_json, configure, load_image, read, sha
from defect_blind_review_pack import MANIFEST, OUT as PACK
from defect_blind_review_update import OUT as PAIR, Residual, base_network, extract, replay
from defect_continue_scratch_net import boxes as deployed
from defect_review_point_behavior import OUT, point_evidence, relation

FINAL = PACK/'核对结果/34处最终核对/cases_audit.json'
LATE_ID = '125f3b781628'


@torch.inference_mode()
def evaluate():
    final=read(FINAL)
    assert final['answered']==34 and final['export_answered']==33 and final['manifest_sha256']==sha(MANIFEST)
    assert len({r['image_id'] for r in final['rows']})==34
    rows={r['image_id']:r for r in read(MANIFEST)['samples'] if r['partition']=='train'}
    detector=base_network()
    heads={}
    for mode in ('control','review'):
        where=PAIR/mode
        receipt=read(where/'receipt.json')
        assert receipt['weight_sha']==sha(where/'weights.pt') and receipt['export_sha256']==final['export_sha256']
        net=Residual(receipt['feature_dim']).cuda().eval()
        net.load_state_dict(torch.load(where/'weights.pt',map_location='cpu',weights_only=True))
        heads[mode]=net
    details=[]
    sums=defaultdict(Counter)
    lookup={r['id']:r for r in read(OUT/'rcnn_training_points.json')['records']}
    weak=read(PAIR/'frozen_review.json')
    weak_ids={r['case'] for r in weak['cases']}
    assert LATE_ID not in weak_ids
    for case in final['rows']:
        im=load_image(rows[case['image_id']]['rgb_path'])
        p,z=extract(detector,im)
        old=deployed(detector,im)
        if not np.array_equal(old,p):
            raise AssertionError('Frozen proposal path differs from native inference')
        pred={'parent':p}
        for mode,h in heads.items():
            q=h(z,torch.from_numpy(p).cuda()).cpu().numpy()
            pred[mode]=q[np.argsort(-q[:,4],kind='stable')]
        bundles={k:point_evidence(q,case['box_xyxy'],case['center_xy']) for k,q in pred.items()}
        bundles['teacher_cpu']=lookup[case['id']]['evidence']
        where='weak_supervision_source' if case['id'] in weak_ids else 'no_weak_label'
        evidence={}
        for k,hits in bundles.items():
            state=relation(case['manual'],hits)
            top={cls:max((p['score'] for p in hits if p['cls']==cls),default=None) for cls in ('scratch','insect')}
            evidence[k]=dict(state=state,top_scores=top)
            sums[k][f"{case['kind']}/{case['manual']}/{state}"]+=1
        details.append(dict(position=case['position'],id=case['id'],group=where,predictions=evidence))
    late=next(v for v in details if v['id']==LATE_ID)
    result=dict(manifest_sha=sha(MANIFEST),final_audit_sha=sha(FINAL),late_weak_training=False,
                scope='Matched reader-designated training point at fixed original thresholds .4/.34; '
                'teacher independent CPU inference, .675/.65. Parent/control/review full-image GPU. '
                'Includes rows seen during native training; control and review share the original frozen 33-answer export. '
                'No localization ground-truth geometry from judgments.',
                counts=sums,late_probe=late,details=details)
    atomic_json(OUT/'all34_frozen_models_gpu.json',result)
    atomic_json(OUT/'all34_point_aggregate.json',{k:v for k,v in result.items() if k not in ('details','late_probe')})
    print('late',late)
    for tag,stats in sums.items():
        print(tag,{k:v for k,v in stats.items() if k.startswith('unmatched/')})


if __name__=='__main__':
    configure(OUT,'cuda')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    with replay.exclusive_run(OUT): evaluate()
