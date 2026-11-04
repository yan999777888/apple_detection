"""CPU-only independent arithmetic, provenance and sampled-pixel audit of local VAL.

Does not import the evaluator or Torch, run inference, alter GT, or choose a model.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/local_formal_20261104/final_audit'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return abs(float(a) - float(b)) < 1e-10
    return a == b


def main():
    checks, nodes, ledgers = {}, {}, {}
    manifest_path = ROOT / 'cache/datasets/local_apple_v1/val_manifest.json'
    manifest = read(manifest_path)
    by_image = {row['image_id']: row for row in manifest['samples']}
    prior_truth = None
    for epoch in (0, 1, 5, 10):
        p = ROOT / f'reports/local_formal_20261104/val_epoch{epoch:03d}'
        summary = read(p / 'validation_summary.json')
        frames = [json.loads(s) for s in (p / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
        truth = {t['key']: t for f in frames for t in f['truth']}
        preds = [x for f in frames for x in f['predictions']]
        def check(name, value):
            checks[f'{epoch:03d}/{name}'] = bool(value)
        check('814_unique_images', len(frames) == len({f['image_id'] for f in frames}) == 814)
        check('3449_unique_GT', len(truth) == sum(len(f['truth']) for f in frames) == 3449)
        check('13_empty_images', sum(not f['truth'] for f in frames) == summary['empty_GT_images'] == 13)
        check('same_truth_across_epochs', prior_truth is None or truth == prior_truth)
        prior_truth = truth
        check('manifest_image_set', set(by_image) == {f['image_id'] for f in frames})
        check('input_hashes', all(f['rgb_sha256'] == by_image[f['image_id']]['rgb_sha256'] and
            f['GT_mask_sha256'] == by_image[f['image_id']]['mask_sha256'] for f in frames))
        check('frame_GT_counts_match_manifest', all(len(f['truth']) == by_image[f['image_id']]['instance_count'] for f in frames))
        check('predictions_count_and_domain', len(preds) == summary['predictions'] and
            all(x['score'] > .001 for x in preds) and all(len(f['predictions']) <= 300 for f in frames))
        check('one_to_one_valid_matches', all(len([x['mask_GT'] for x in f['predictions'] if x['mask_GT']]) ==
            len({x['mask_GT'] for x in f['predictions'] if x['mask_GT']}) and
            all(x['mask_GT'] in truth and x['mask_IoU'] >= .5 for x in f['predictions'] if x['mask_GT']) for f in frames))
        groups = {}
        for pred in preds:
            row = groups.setdefault(pred['score'], [0, 0])
            row[0 if pred['mask_GT'] else 1] += 1
        curve, tp, fp = [], 0, 0
        for score in sorted(groups, reverse=True):
            tp += groups[score][0]; fp += groups[score][1]
            curve.append({'threshold': score, 'TP': tp, 'FP': fp, 'FN': 3449-tp,
                'precision': tp/(tp+fp), 'recall': tp/3449})
        check('full_PR_whole_score_groups', close(curve, summary['mask']['curve']))
        fixed_preds = [x for x in preds if x['score'] >= .25]
        tp = sum(x['mask_GT'] is not None for x in fixed_preds); fp = len(fixed_preds)-tp
        fixed = {'threshold': .25, 'TP': tp, 'FP': fp, 'FN': 3449-tp, 'precision':tp/(tp+fp), 'recall':tp/3449}
        check('fixed_metrics', close(fixed, summary['mask']['fixed']))
        candidates = [x for x in curve if x['recall'] >= .9]
        p90 = max(candidates, key=lambda x:x['precision']) if candidates else None
        check('P90_complete_tie_groups', close(p90, summary['mask']['P_at_R90']))
        ap = sum(max([x['precision'] for x in curve if x['recall'] >= i/100], default=0) for i in range(101))/101
        check('AP50_101point', close(ap, summary['mask']['AP50_101point']))
        check('max_recall', close(curve[-1]['recall'], summary['mask']['maximum_recall']))
        fixed_hits = {x['mask_GT']:x['mask_IoU'] for x in fixed_preds if x['mask_GT']}
        gt_saved = read(p/'GT_matches_fixed.json')
        check('fixed_GT_ledger', gt_saved['threshold']==.25 and len(gt_saved['GT'])==3449 and
            all(x == {**truth[x['key']], 'matched': x['key'] in fixed_hits} for x in gt_saved['GT']))
        for label, threshold in [('fixed', .25), ('P_at_R90', p90['threshold'] if p90 else None)]:
            if threshold is None:
                check('P90_strata_absent', label not in summary['occlusion_mask'] or summary['occlusion_mask'][label] is None)
                continue
            hits = {x['mask_GT']:x['mask_IoU'] for x in preds if x['score']>=threshold and x['mask_GT']}
            for stratum in {t['visibility_stratum'] for t in truth.values()}:
                keys={k for k,t in truth.items() if t['visibility_stratum']==stratum}; found=keys & hits.keys()
                actual={'GT':len(keys),'TP':len(found),'FN':len(keys-found),'recall':len(found)/len(keys),
                    'matched_mean_iou':sum(hits[k] for k in found)/len(found) if found else None,
                    'missed_GT_keys':sorted(keys-found)}
                check(f'{label}/{stratum}', close(actual,summary['occlusion_mask'][label][stratum]))
        empty_fp=sum(x['score']>=.25 for f in frames if not f['truth'] for x in f['predictions'])
        check('empty_FP', empty_fp==summary['empty_image_FP_at_fixed'])
        check('matched_mean_iou', close(sum(fixed_hits.values())/len(fixed_hits),summary['matched_mask_mean_IoU_at_fixed']))
        check('manifest_SHA',sha(manifest_path)==summary['manifest_sha256'])
        check('weights_SHA',sha(summary['weights'])==summary['weights_sha256'])
        source = ROOT / ('src/apple_local_rgbd/local_evaluation.py' if epoch>=5 else
            'reports/local_formal_20261104/recovery_v3/source_before/local_evaluation.py')
        check('matching_historical_evaluator_SHA',sha(source)==summary['source_sha256'])
        speed=read(p/'speed_benchmark.json'); speed_clean=[]
        check('speed_working_point',speed['conf']==.25 and speed['max_det']==64 and speed['precision']=='FP16' and speed['batch']==1)
        for case in speed['cases']:
            ms=[f['ms'] for f in case['frames']]
            check(f'speed_{case["imgsz"]}_40_frames',len(ms)==40 and len({f['image_id'] for f in case['frames']})==8 and case['warmup']==5 and case['rounds']==5)
            check(f'speed_{case["imgsz"]}_arithmetic',close(float(np.mean(ms)),case['mean_ms']) and
                close(float(np.percentile(ms,95)),case['p95_ms']) and close(1000/np.mean(ms),case['fps_from_mean']))
            speed_clean.append({k:v for k,v in case.items() if k!='frames'})
        pixel_stats={'images':0,'predictions':0,'matches':0,'max_iou_error':0.0}
        if epoch>=5:
            frame_by_id={f['image_id']:f for f in frames}
            for saved in sorted((p/'diagnostic_native_masks').glob('*.npz')):
                image_id=saved.stem+'.png'; frame=frame_by_id[image_id]; row=by_image[image_id]
                with np.load(saved,allow_pickle=False) as arr:
                    pm=np.unpackbits(arr['visible_packed'],axis=-1,bitorder='little')[...,:int(arr['original_hw'][1])].astype(bool)
                    boxes=arr['boxes'].copy()
                with np.load(row['mask_path'],allow_pickle=False) as arr:
                    gm=np.unpackbits(arr['visible_packed'],axis=-1,bitorder='little')[...,:int(arr['original_hw'][1])].astype(bool)
                    ids=[str(x) for x in arr['instance_ids']]
                expected=[x for x in frame['predictions'] if x['score']>=.25]
                check('pixel_count_'+image_id,len(expected)==len(pm)==len(boxes))
                used=set()
                for j,mask in enumerate(pm):
                    areas=[float(np.count_nonzero(mask & g))/max(1,np.count_nonzero(mask | g)) if i not in used else -1 for i,g in enumerate(gm)]
                    best=int(np.argmax(areas)) if areas else -1
                    hit=best>=0 and areas[best]>=.5
                    key=f'{image_id}::{ids[best]}' if hit else None
                    check(f'pixel_match_{image_id}_{j}', key==expected[j]['mask_GT'] and close(float(boxes[j,4]),expected[j]['score']))
                    if hit:
                        used.add(best);err=abs(areas[best]-expected[j]['mask_IoU'])
                        pixel_stats['max_iou_error']=max(pixel_stats['max_iou_error'],err);pixel_stats['matches']+=1
                        check(f'pixel_IoU_{image_id}_{j}',err<1e-12)
                pixel_stats['images']+=1;pixel_stats['predictions']+=len(pm)
        nodes[epoch]={'fixed':fixed,'AP50_101point':ap,'P_at_R90':p90,'maximum_recall':curve[-1]['recall'],
            'empty_FP':empty_fp,'matched_mean_IoU':summary['matched_mask_mean_IoU_at_fixed'],
            'occlusion':{k:{s:{z:v for z,v in stats.items() if z!='missed_GT_keys'} for s,stats in val.items()}
                         for k,val in summary['occlusion_mask'].items() if val is not None},
            'speed':speed_clean,'pixel_checks':pixel_stats,'weights':summary['weights'],'weights_sha256':summary['weights_sha256'],
            'source_sha256':summary['source_sha256'],'prediction_count':len(preds),'input_summary_sha256':sha(p/'validation_summary.json')}
        ledgers[epoch]={'truth':truth,'fixed':set(fixed_hits),'all':{x['mask_GT'] for x in preds if x['mask_GT']}}
    transitions={}
    for old,new in [(0,1),(1,5),(5,10),(1,10),(0,10)]:
        transitions[f'{old}_to_{new}']={}
        for domain in ['fixed','all']:
            lost=ledgers[old][domain]-ledgers[new][domain]; recovered=ledgers[new][domain]-ledgers[old][domain]
            count=lambda keys:dict(sorted(Counter(ledgers[new]['truth'][k]['visibility_stratum'] for k in keys).items()))
            transitions[f'{old}_to_{new}'][domain]={'new_FN':len(lost),'recovered':len(recovered),'net_TP':len(recovered)-len(lost),
                'new_FN_by_stratum':count(lost),'recovered_by_stratum':count(recovered),
                'new_FN_keys':sorted(lost),'recovered_keys':sorted(recovered)}
    report={'schema':'independent_final_local_visible_metrics_audit_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'method':'Independent standard library/NumPy only; saved-prediction arithmetic and full native Boolean pixel checks for saved first8 images at epochs5/10; no Torch or inference.',
        'passed':all(checks.values()),'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],
        'nodes':nodes,'GT_transitions':transitions,'limitations':['VAL development includes309 fruit identities shared with training; test sealed.',
            'GT match tables checked for all814; independent pixel matching limited to saved first8 images for epoch5/10.',
            'P90 score point not speed-tested; deployment speed .25/maxdet64 differs from PR .001/maxdet300.',
            '512 full-view speed only; no512 fullVAL or D405 RGBD/3D/servo validation.']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'metrics_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 局部模型最终指标独立审计','',f'检查通过：{report["passed"]}；检查项{len(checks)}；失败项：{report["failed_checks"]}。','',
        '|epoch|P固定0.25|R固定0.25|TP/FP/FN|AP50|最高Recall|P@R90|heavy R|640均ms/P95/FPS|',
        '|---|---:|---:|---|---:|---:|---|---:|---|']
    for e,n in nodes.items():
        f=n['fixed'];pr=n['P_at_R90'];h=n['occlusion']['fixed']['heavy_visible_lt_0.40'];v=n['speed'][0]
        lines.append(f'|{e}|{100*f["precision"]:.4f}%|{100*f["recall"]:.4f}%|{f["TP"]}/{f["FP"]}/{f["FN"]}|{100*n["AP50_101point"]:.4f}%|{100*n["maximum_recall"]:.4f}%|'+
            (f'{100*pr["precision"]:.4f}% / {100*pr["recall"]:.4f}%' if pr else '不可达')+f'|{100*h["recall"]:.4f}%|{v["mean_ms"]:.4f}/{v["p95_ms"]:.4f}/{v["fps_from_mean"]:.4f}|')
    lines+=['','第10轮固定阈值P/R较高，但AP50和最高Recall均低于第5轮；不能宣布全面优于第5轮。最高Recall仅针对当前保留候选域。','',
        '## 逐GT变化','', '|比较|域|新增FN|恢复|heavy新增FN|heavy恢复|','|---|---|---:|---:|---:|---:|']
    for name,parts in transitions.items():
        for domain,x in parts.items():
            lines.append(f'|{name}|{domain}|{x["new_FN"]}|{x["recovered"]}|{x["new_FN_by_stratum"].get("heavy_visible_lt_0.40",0)}|{x["recovered_by_stratum"].get("heavy_visible_lt_0.40",0)}|')
    lines+=['','fixed为score≥0.25；all为score>0.001、max_det300的整个保留候选域。新增和恢复逐GT名单在JSON。','',*report['limitations']]
    (OUT/'metrics_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'check_count':len(checks),'failed_checks':report['failed_checks']},ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
