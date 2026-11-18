"""Recompute held-date results and paired localization changes for reviewed-point training."""

import argparse
from collections import Counter
from pathlib import Path

from audit_defect_causes_20261112 import match
from class0_patch_refinement import ROOT, atomic_json, read, sha
from compare_defect_target9485 import inspect, delta
from defect_scratch_specialist_audit import (
    CANDIDATE, MANIFEST, OLD29, canonical, expect_prefix, timing, validate,
)

RUN = ROOT / 'experiments/defect_blind_review_update_v1'
OUTPUT = ROOT / 'experiments/defect_review_impact_v1'


def evaluate(folder, dev, canonical_rows):
    fixed = validate(read(folder/'fixed_predictions.json'), dev)
    low = validate(read(folder/'low_predictions.json'), dev)
    expect_prefix(fixed, low, (0.40, 0.34))
    stats, hits, reasons, _ = inspect({k: v['pred'] for k, v in fixed.items()}, canonical_rows)
    work = read(folder/'metrics.json')['fixed']['total']
    if any(work[k] != stats['total'][k] for k in ('TP', 'FP', 'FN')):
        raise ValueError(f'Metrics differ from independently matched canonical GT in {folder}')
    # Give every fixed FN exactly one cause, independent of the number of nearby boxes.
    causes = {}
    for image, row in low.items():
        predictions = row['pred']
        _, overlaps = match(predictions, row['GT'])
        for i, truth in enumerate(row['GT']):
            if (image, i) in hits:
                continue
            cls = truth['class']
            same = [float(overlaps[j, i]) for j, p in enumerate(predictions) if p[5] == cls]
            opposite = [float(overlaps[j, i]) for j, p in enumerate(predictions) if p[5] != cls]
            if max(same, default=0) >= .5:
                compatible = [j for j, p in enumerate(predictions) if p[5] == cls and overlaps[j, i] >= .5]
                if max(predictions[j][4] for j in compatible) >= (.40, .34)[cls]:
                    cause = 'competing_GT_consumed_fixed_candidate'
                else:
                    cause = 'matching_candidate_below_threshold'
            elif max(opposite, default=0) >= .5:
                cause = 'only_other_class_reaches_IoU50'
            elif max(same, default=0) >= .1:
                cause = 'partial_same_class_overlap'
            else:
                cause = 'no_nearby_same_class_candidate'
            causes[f'{image}#{i}'] = dict(cls=cls, cause=cause)
    if len(causes) != stats['total']['FN']:
        raise ValueError('FN ledger mismatch')
    total = stats['total']
    speed = timing(read(folder/'speed.json'))
    result = dict(
        checkpoint_sha256=read(folder/'receipt.json').get('weight_sha') if (folder/'receipt.json').exists() else None,
        output_sha256={n: sha(folder/n) for n in ('fixed_predictions.json','low_predictions.json','speed.json')},
        class_and_date=stats, FP_geometry=reasons, FN_counts=Counter(
            f"class{r['cls']}/{r['cause']}" for r in causes.values()),
        FN_cases=causes, speed=speed, target_P_R=(total['P'] >= .94 and total['R'] >= .85),
    )
    return result, hits, fixed


def crossed_workpoint(new_fixed, reference_fixed):
    counts = Counter()
    for image, row in new_fixed.items():
        streams = []
        for stream in (reference_fixed[image], row):
            pairs, _ = match(stream['pred'], stream['GT'])
            ids = {}
            for pi, p in enumerate(stream['pred']):
                uid = (int(p[5]), *p[:4])
                if uid in ids:
                    raise ValueError('Duplicate exact output box')
                ids[uid] = 'TP' if pi in pairs else 'FP'
            streams.append(ids)
        old, new = streams
        for label, ids, values in [('entered',new.keys()-old.keys(),new),
                                   ('left',old.keys()-new.keys(),old)]:
            for uid in ids:
                counts[f'{label}/class{uid[0]}/{values[uid]}'] += 1
        for uid in new.keys() & old.keys():
            counts[f'shared/class{uid[0]}/{old[uid]}_to_{new[uid]}'] += 1
    return dict(exact_key='(image, class, native xyxy), scores excluded', counts=counts)


def audit(parent=CANDIDATE, control=RUN/'control', candidate=RUN/'review'):
    dev, train_dates = canonical()
    if any(date in train_dates for date, _ in dev.values()):
        raise ValueError('Development date overlaps training dates')
    history = read(OLD29)['cases']['miss']
    if len(history) != 29:
        raise ValueError('Historical scratch cohort changed')
    canon = {k: dict(GT=v[1]) for k,v in dev.items()}
    result, hits, fixed = {}, {}, {}
    for label, folder in [('parent', parent), ('control', control), ('review', candidate)]:
        result[label], hits[label], fixed[label] = evaluate(folder, dev, canon)
    comparisons = {}
    for label in ['parent', 'control']:
        comparisons[f'review_vs_{label}'] = delta(hits['review'], hits[label], canon)
        comparisons[f'review_vs_{label}']['retained_detections'] = crossed_workpoint(fixed['review'], fixed[label])
    old_cases = []
    for r in history:
        i = (r['image'],r['index'])
        if canon[i[0]]['GT'][i[1]]['box'] != r['box'] or canon[i[0]]['GT'][i[1]]['class'] != 0:
            raise ValueError('Historic target identity changed')
        old_cases.append(dict(id=r['code'], **{tag:i in cases for tag,cases in hits.items()}))
    return dict(scope='Unchanged held-date 113 images/230 original boxes; class-specific one-to-one IoU>=0.5; '
                'no human point edits applied to development GT.', native_manifest_sha256=sha(MANIFEST),
                fixed_workpoint=[.4,.34], measurements=result, paired_GT_changes=comparisons,
                historic_29=old_cases)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent',type=Path,default=CANDIDATE)
    ap.add_argument('--control',type=Path,default=RUN/'control')
    ap.add_argument('--review',type=Path,default=RUN/'review')
    ap.add_argument('--self-check',action='store_true')
    args = ap.parse_args()
    if args.self_check:
        out = audit(args.parent,args.parent,args.parent)
        assert all(x['gained']['count'] == x['lost']['count'] == 0 for x in out['paired_GT_changes'].values())
        print(out['measurements']['parent']['class_and_date']['total'])
    else:
        out = audit(args.parent,args.control,args.review)
        atomic_json(OUTPUT/'paired_development.json',out)
        print({k:v['class_and_date']['total'] for k,v in out['measurements'].items()})
