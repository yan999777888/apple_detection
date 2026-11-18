"""Replay the two frozen detectors on exactly the train-only locations seen by a reader."""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_defect_causes_20261112 import ious
from class0_patch_refinement import ROOT, atomic_json, load_image, read, sha
from defect_blind_review_pack import MANIFEST, OUT as PACK
from defect_continue_scratch_net import LockedInsect, ScratchHead, WARM, WARM_SHA
from defect_continue_scratch_specialist import OUT as SPECIALIST
from defect9485_small import nms, square

OUT = ROOT / 'experiments/defect_review_impact_v1'
THRESHOLDS = (0.4, 0.34)
LABEL = {0: 'scratch', 1: 'insect'}


def native(raw, hw, transform):
    x, y, sx, sy = transform
    h, w = hw
    values = raw[0]
    rows = values[values[:, 4] >= .001].cpu().numpy().copy()
    rows[:, [0, 2]] = ((rows[:, [0, 2]] - x) / sx).clip(0, w)
    rows[:, [1, 3]] = ((rows[:, [1, 3]] - y) / sy).clip(0, h)
    return nms(rows)


def point_evidence(rows, box, center, thresholds=THRESHOLDS):
    """Reader identified a point. IoU is with the sampling proposal, never an expert GT."""
    cx, cy = center
    near = [
        r for r in rows
        if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]
    ]
    hits = []
    for r in sorted(near, key=lambda v: -v[4]):
        hits.append(dict(cls=LABEL[int(r[5])], score=float(r[4]),
                         sampler_iou=float(ious([r[:4]], [box])[0, 0]),
                         fixed=bool(r[4] >= thresholds[int(r[5])])))
    return hits


def relation(response, evidence):
    classes = {r['cls'] for r in evidence if r['fixed']}
    if response == 'other':
        return 'response_other_with_detection' if classes else 'response_other_suppressed'
    if response in ('scratch', 'insect'):
        if response in classes:
            return 'both_classes' if len(classes) > 1 else 'same_class_only'
        return 'other_class_only' if classes else 'no_detection'
    return 'not_interpretable'


@torch.inference_mode()
def create():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    key = read(PACK / '组织者留存/source_key.json')
    reader_path = max((PACK / '盲审材料/导出结果').glob('*.json'), key=lambda p: p.stat().st_mtime_ns)
    reader = read(reader_path)
    centers = {r['id']: r['center'] for r in read(PACK / '盲审材料/cases.json')['cases']}
    assert key['case_manifest_sha'] == reader['case_manifest_sha256'] == sha(PACK / '盲审材料/cases.json')
    assert sha(MANIFEST) == key['ssda_manifest_sha'] and sha(WARM) == WARM_SHA
    annotations = {r['id']: r['response'] for r in reader['cases']}
    assert set(annotations) == set(centers) == {r['id'] for r in key['cases']}
    train = {r['image_id']: r for r in read(MANIFEST)['samples'] if r['partition'] == 'train'}
    specialist_sha = sha(SPECIALIST / 'weights.pt')
    assert specialist_sha == read(SPECIALIST/'receipt.json')['weight_sha']
    locked = LockedInsect('cpu').eval().requires_grad_(False)
    specialist = ScratchHead().eval().requires_grad_(False)
    specialist.load_state_dict(torch.load(SPECIALIST/'weights.pt', weights_only=True, map_location='cpu'))
    counts = Counter()
    out = []
    for rec in key['cases']:
        uid = rec['id']
        photo = train[rec['image']]
        assert rec['raw_sha'] == sha(Path(photo['rgb_path']))
        frame = load_image(photo['rgb_path'])
        image, transform = square(frame)
        x = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
        p2, pyramid = locked.features(x)
        raw = native(locked.detections(pyramid), frame.shape[:2], transform)
        extra = native(specialist(p2, pyramid), frame.shape[:2], transform)
        combined = np.concatenate((raw[raw[:, 5] == 1], extra[extra[:, 5] == 0]))
        by_model = {}
        for name, rows in (('warm', raw), ('specialist', combined)):
            hit = point_evidence(rows, rec['box'], centers[uid])
            state = relation(annotations[uid], hit)
            by_model[name] = dict(at_point=hit, fixed_relation=state)
            counts[(rec['source'], name, state)] += 1
        out.append(dict(id=uid, source=rec['source'], response=annotations[uid], models=by_model))
    return dict(experiment='Full image FP32 CPU single thread, train-only replay, predefined thresholds; '
                'readers only labeled a location, image dates were seen during training, no generalization metric.',
                split_sha=sha(MANIFEST), packet_sha=key['case_manifest_sha'], export_sha=sha(reader_path),
                warm_sha=WARM_SHA, specialist_sha=specialist_sha, thresholds=THRESHOLDS,
                counts=[dict(source=s, model=m, relation=r, n=n) for (s,m,r), n in sorted(counts.items())],
                records=out)


@torch.inference_mode()
def compare_rcnn():
    from defect_continue_rcnn import initialized, load_checkpoint, split_train, SIDE
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records = read(OUT/'training_point_behavior.json')
    cases = {r['id']:r for r in read(PACK/'组织者留存/source_key.json')['cases']}
    centers = {r['id']:r['center'] for r in read(PACK/'盲审材料/cases.json')['cases']}
    xx, cc = split_train()
    training = {v['image_id']:v for v in xx}
    calibration = {v['image_id']:v for v in cc}
    rows = {**training, **calibration}
    weights = ROOT/'experiments/defect_continue_rcnn/transfer_v2/epoch18.pt'
    f = read(weights.parent/'frozen.json')['selected']
    assert f['epoch'] == 18 and f['sha256'] == sha(weights)
    model = initialized().eval().requires_grad_(False)
    model.load_state_dict(load_checkpoint(weights)['model'])
    report, counts = [], Counter()
    for subject in records['records']:
        uid = subject['id']
        lookup = cases[uid]
        raw = rows[lookup['image']]
        assert sha(raw['rgb_path']) == lookup['raw_sha']
        frame = load_image(raw['rgb_path'])
        h,w = frame.shape[:2]
        im = cv2.resize(frame,(SIDE,SIDE),interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(im[:,:,::-1])).permute(2,0,1).float()/255
        pred = model([x])[0]
        boxes = pred['boxes']*torch.tensor([w/SIDE,h/SIDE,w/SIDE,h/SIDE])
        all_preds = torch.cat((boxes,pred['scores'][:,None],pred['labels'][:,None]-1),1).numpy()
        hits = point_evidence(all_preds,lookup['box'],centers[uid], f['cut'])
        state = relation(subject['response'],hits)
        part = 'teacher_fit' if lookup['image'] in training else 'teacher_calibration'
        if lookup['source'] == 'unmatched OOF candidate':
            base_state = subject['models']['specialist']['fixed_relation']
            counts[(part,subject['response'],base_state,state)] += 1
        report.append(dict(id=uid, fit_status=part, fixed_relation=state, evidence=hits))
    return dict(scope='Native training partition, teacher-student diagnostic only, '
                'teacher selected using its own train/calibration samples; CPU FP32 single thread',
                teacher_sha=f['sha256'], packet_sha=records['packet_sha'], export_sha=records['export_sha'],
                thresholds=f['cut'], records=report,
                counts=[dict(fit=s, response=r, parent=p, teacher=t, n=n)
                        for (s,r,p,t),n in sorted(counts.items())])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rcnn',action='store_true')
    args = parser.parse_args()
    result = compare_rcnn() if args.rcnn else create()
    filename = 'rcnn_training_points.json' if args.rcnn else 'training_point_behavior.json'
    atomic_json(OUT / filename, result)
    for row in result['counts']:
        print(row)
