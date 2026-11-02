"""Original-weight proposal audit and real pre-mask ablations, no training.

The fixed stratified subset is diagnostic, not full VAL or independent test.
GT is used only after inference for matching, strata and mask measurements.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'strong_runtime/src'), str(REPO/'strong_runtime/tmp')]
from evaluate_fast_deployment import sha, write, now, source_inventory, summarize, policy

POLICIES = {
    'none': {}, 'k256': {'topk':256}, 'k128': {'topk':128},
    'k96': {'topk':96}, 'k64': {'topk':64},
    'score003': {'score_threshold':.03}, 'nms080': {'box_nms_iou':.8},
    'nms080_k64': {'box_nms_iou':.8, 'topk':64},
    'score003_nms080_k64': {'score_threshold':.03, 'box_nms_iou':.8, 'topk':64},
}


def sources():
    return {**source_inventory(), **{p:sha(REPO/p) for p in (
        'scripts/audit_premask_candidates.py','deployment/pre_mask.py','deployment/candidate_policy.py')}}


def selected_prediction(prediction, threshold=.75):
    selected = [i for i,s in enumerate(prediction['scores']) if s >= threshold]
    return {**prediction, 'boxes':prediction['boxes'][selected],
            'scores':[prediction['scores'][i] for i in selected],
            'visible':prediction['visible'][selected], 'amodal':prediction['amodal'][selected]}


def diagnostic(prediction, truth, target, metrics):
    from apple_seg.od_evaluate import _roi_mask_overlaps
    from apple_seg.evaluate import _greedy_assignment
    pred = selected_prediction(prediction)
    match = metrics.match_image(truth['boxes'], pred['boxes'].numpy(), pred['scores'], .5, .75)
    assigned = [-1]*len(pred['scores'])
    for p,g in zip(match['pred_indices'],match['gt_indices'],strict=True):
        assigned[int(p)] = int(g)
    row = {'image_id':target['source']['image_id'], 'gt_count':len(truth['boxes']),
           'scores':pred['scores'], 'box_gt_indices_by_prediction':assigned}
    for branch,matrix in _roi_mask_overlaps(pred,truth,tuple(target['original_hw'])).items():
        row[branch+'_gt_indices_by_prediction'] = _greedy_assignment(matrix,pred['scores'])
    return row


def subset_contract(contract, samples):
    names = {s['image_id'] for s in samples}
    truth = {k:v for k,v in contract['truth'].items() if k in names}
    return {**contract, 'truth':truth,
            'strata':dict(Counter(row['occlusion_stratum'] for rows in truth.values() for row in rows))}


def audit_policies(trace, prediction, gt_boxes, rows, metrics):
    import numpy as np
    import torch
    from deployment.candidate_policy import select_candidates
    boxes = trace['decoded_boxes'].cpu()
    initial = trace['initial_scores'].cpu()
    final = trace['final_scores'].cpu()
    output = trace['output_indices'].cpu().tolist()
    match = metrics.match_image(gt_boxes,prediction['boxes'].numpy(),prediction['scores'],.5,.75)
    tp = [(int(p),int(g),output[int(p)]) for p,g in
          zip(match['pred_indices'],match['gt_indices'],strict=True) if g>=0]
    rank = torch.empty(len(initial),dtype=torch.int64)
    rank[torch.argsort(initial,descending=True,stable=True)] = torch.arange(1,len(initial)+1)
    report = {'final_high_initial_below_003':int(((final>=.75)&(initial<.03)).sum()),
        'baseline_TP':len(tp), 'TP_initial_scores':[float(initial[j]) for _,_,j in tp],
        'TP_initial_ranks':[int(rank[j]) for _,_,j in tp], 'policies':{}}
    for name,spec in POLICIES.items():
        keep = select_candidates(boxes,initial,**spec).tolist()
        kept = set(keep)
        lost = [{'gt_index':g,'apple_id':rows[g]['apple_id'],
                 'stratum':rows[g]['occlusion_stratum'], 'proposal_index':j,
                 'initial_score':float(initial[j]), 'final_score':float(final[j]),
                 'initial_rank':int(rank[j])} for _,g,j in tp if j not in kept]
        report['policies'][name] = {'mask_candidates':len(keep), 'lost_original_TP_proposals':lost,
            'warning':'Loss of the original TP proposal; alternative retained proposals may recover its GT. Actual post-filter inference still required.'}
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'configs/strong_baseline_4060.json')
    parser.add_argument('--samples',type=Path,default=REPO/'configs/premask_audit_samples_20261102.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['audit','screen'],required=True)
    parser.add_argument('--models',nargs='+',choices=['base','small'],default=['base','small'])
    parser.add_argument('--policies',nargs='+',choices=list(POLICIES),default=['k64','score003','nms080','nms080_k64','score003_nms080_k64'])
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args(argv)
    out=args.output.resolve()
    if out.drive.upper()!='F:':raise ValueError('All outputs must stay on F:')
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    root=(args.config.resolve().parent/cfg['path_base']).resolve()
    paths={k:(root/v).resolve() for k,v in cfg['paths'].items()}
    samples=json.loads(args.samples.read_text(encoding='utf-8'))['samples']
    if args.limit:samples=samples[:args.limit]
    status={'schema':'premask_subset_v1','phase':'initializing','created_utc':now(),
            'mode':args.mode,'sample_manifest':str(args.samples.resolve()),'sample_sha256':sha(args.samples),
            'samples':samples,'scope':'Stratified diagnostic subset; not full VAL or independent test',
            'config_sha256':sha(args.config),'source_sha256':sources(), 'models':{},
            'training':False,'gt_used_for_inference':False,'precision':'fp16',
            'roi_chunk':64,'pre_nms_threshold':None,'fixed_confidence':.75}
    write(out/'status.json',status)
    try:
        import numpy as np
        import torch
        import flip_tta
        from apple_seg.evaluate import load_papple_contract,_audit_target,_audit_rgb,_mask_pr,_occlusion
        from apple_seg.data import PAppleDataset
        from apple_seg.od_data import ODInputDataset
        from apple_seg.od_training import load_checkpoint
        from apple_seg.od_model import build_od_amodal_detector
        from apple_seg.od_inference import _strict_restore
        from apple_seg.od_evaluate import _validate_prediction
        from deployment.pre_mask import PreMaskDeploymentAdapter
        from deployment.strong_fast import FastDeploymentAdapter
        metrics=flip_tta.metrics
        torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        status.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)
        contract=load_papple_contract(paths['provenance'])
        native=PAppleDataset(paths['papple_root'],'val',32,augment=False)
        if native.annotation_sha256!=contract['provenance']['binding']['current_annotation_sha256']:
            raise RuntimeError('Annotations differ')
        dataset=ODInputDataset(native,short_sides=(1024,),max_size=1280)
        small_contract=subset_contract(contract,samples)
        for name in args.models:
            checkpoint=paths[name+'_checkpoint']
            if sha(checkpoint)!=cfg[name+'_checkpoint_sha256']:raise RuntimeError('Original weights differ')
            payload=load_checkpoint(checkpoint)
            model=build_od_amodal_detector(weights=None,**payload['constructor_config'])
            restored=_strict_restore(model,payload['model'])
            if model.constructor_config!=payload['constructor_config']:raise RuntimeError('Architecture changed')
            del payload;gc.collect()
            model.requires_grad_(False).eval().to('cuda')
            versions={k:v._version for k,v in model.state_dict(keep_vars=True).items()}
            info={'checkpoint_sha256':cfg[name+'_checkpoint_sha256'],'strict_restore':restored,'variants':{}}
            status['models'][name]=info
            active=['none'] if args.mode=='audit' else args.policies
            for variant in active:
                records=[];diagnostics=[];audits=[];times=[]
                directory=out/name/variant;directory.mkdir(parents=True)
                entry={'phase':'running','policy':POLICIES[variant],'images':0}
                info['variants'][variant]=entry
                status.update(phase='inference',current_model=name,current_variant=variant)
                write(out/'status.json',status)
                adapter=PreMaskDeploymentAdapter(model,candidate_policy=POLICIES[variant],
                    capture_candidates=args.mode=='audit',roi_chunk_size=64,roi_precision='autocast')
                # Explicit disabled-path equivalence guard, same loaded weights/input.
                if args.mode=='audit':
                    pixels=dataset[samples[0]['image_index']]['pixel_values'][None].to('cuda')
                    with (FastDeploymentAdapter(model,roi_chunk_size=64,roi_precision='autocast') as original,
                          torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16)):
                        reference=original.predict_views(pixels)[0]
                    with adapter,torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                        actual=adapter.predict_views(pixels)[0]
                    diffs={key:float((actual[key]-reference[key]).abs().max()) if actual[key].numel() else 0.
                        for key in ('boxes','scores','roi_visible_masks','roi_amodal_masks')}
                    if any(diffs.values()):raise RuntimeError('Disabled adapter changed output: '+str(diffs))
                    info['disabled_path_exact_first_image']=diffs
                    del reference,actual,pixels
                with adapter,torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    for seq,sample in enumerate(samples):
                        idx=sample['image_index'];frozen=contract['images'][idx]
                        if frozen['image_id']!=sample['image_id']:raise RuntimeError('Sample identity differs')
                        item=dataset[idx];target=item['target'];hw=tuple(target['original_hw'])
                        rows=contract['truth'][frozen['image_id']]
                        truth=_audit_target(target,frozen,rows)
                        _audit_rgb(Path(target['source']['rgb_path']),frozen)
                        pixels=item['pixel_values'];input_hw=tuple(pixels.shape[-2:])
                        torch.cuda.synchronize();started=time.perf_counter()
                        raw=adapter.predict_views(pixels[None].to('cuda'))
                        pred=_validate_prediction(raw,input_hw,hw,mask_chunk_size=8)
                        torch.cuda.synchronize()
                        times.append({'image_id':sample['image_id'],'ms':1000*(time.perf_counter()-started),
                                      'stats':deepcopy(adapter.last_stats)})
                        records.append({'image_id':sample['image_id'],'gt':{'boxes':truth['boxes']},
                            'pred':{'boxes':pred['boxes'].tolist(),'scores':pred['scores']}})
                        diagnostics.append(diagnostic(pred,truth,target,metrics))
                        if args.mode=='audit':
                            trace=adapter.traces[0]
                            row=audit_policies(trace,pred,truth['boxes'],rows,metrics)
                            row['image_id']=sample['image_id'];audits.append(row)
                            np.savez_compressed(directory/f'candidates_{idx:04d}.npz',
                                **{k:v.cpu().numpy() for k,v in trace.items()})
                        entry.update(images=seq+1,updated_utc=now())
                        write(out/'status.json',status)
                        print(f'PREMASK {args.mode} {name} {variant} {seq+1}/{len(samples)}',flush=True)
                        del item,target,truth,pixels,raw,pred
                evaluation_policy=policy(None)
                evaluation_policy.update(threshold_selection_split='32-image stratified diagnostic subset; not model acceptance',
                    curve_scope='All returned candidates under this pre-mask policy; subset only')
                summary,curve=summarize(records,metrics,evaluation_policy)
                total=sum(len(r['gt']['boxes']) for r in records)
                entry.update(phase='complete',summary=summary,GT=total,
                    mask_fixed075={branch:_mask_pr(diagnostics,branch+'_mask',.75,total,metrics)
                                   for branch in ('visible','amodal')},
                    occlusion_fixed075=_occlusion(diagnostics,small_contract,.75)['groups'])
                for branch in entry['mask_fixed075'].values():
                    branch['threshold_policy']='Fixed .75 on diagnostic subset; independent mask IoU>=.5 matching'
                write(directory/'predictions.json',records);write(directory/'diagnostics.json',diagnostics)
                write(directory/'summary.json',entry);write(directory/'timing.json',{'scope':'Unwarmed validation path, not FPS benchmark','frames':times})
                if curve:write(directory/'curve.json',curve)
                if audits:
                    totals={p:{'lost_original_TP_proposals':sum(len(r['policies'][p]['lost_original_TP_proposals']) for r in audits),
                               'mean_mask_candidates':float(np.mean([r['policies'][p]['mask_candidates'] for r in audits]))}
                            for p in POLICIES}
                    write(directory/'candidate_audit.json',{'images':audits,'policy_totals':totals,
                        'final_high_initial_below_003':sum(r['final_high_initial_below_003'] for r in audits),
                        'baseline_TP':sum(r['baseline_TP'] for r in audits)})
                    entry['candidate_audit_totals']=totals
                write(out/'status.json',status)
            if versions!={k:v._version for k,v in model.state_dict(keep_vars=True).items()}:
                raise RuntimeError('Model tensors were modified')
            if sha(checkpoint)!=cfg[name+'_checkpoint_sha256']:raise RuntimeError('Checkpoint modified')
            del adapter,model;gc.collect();torch.cuda.empty_cache()
        if sources()!=status['source_sha256']:raise RuntimeError('Source changed during audit')
        status.update(phase='complete',completed_utc=now());write(out/'status.json',status)
    except BaseException as error:
        status.update(phase='failed',error=str(error),traceback=traceback.format_exc(),updated_utc=now())
        write(out/'status.json',status);raise


if __name__=='__main__':main()
