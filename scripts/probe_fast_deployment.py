"""Small, original-weight deployment compatibility/speed probe, never VAL acceptance."""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time
import traceback

from reproduce_strong_baseline import REPO, sha, write, source_inventory
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sample-dir', type=Path, required=True)
    p.add_argument('--contour-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=['base','small'], required=True)
    p.add_argument('--limit', type=int, default=3)
    p.add_argument('--variants', nargs='+', default=['fp32_filter','fp32_chunk32','fp16_chunk32','fp16_chunk64','fp16_pair64'])
    args = p.parse_args()
    cfgpath = args.config.resolve()
    cfg = json.loads(cfgpath.read_text(encoding='utf-8-sig'))
    root = (cfgpath.parent/cfg['path_base']).resolve()
    checkpoint = (root/cfg['paths'][args.model+'_checkpoint']).resolve()
    out = args.output.resolve()
    assert out.drive.upper() == 'F:'
    out.mkdir(parents=True, exist_ok=False)
    sampled, contoured = args.sample_dir.resolve(), args.contour_dir.resolve()
    manifest = json.loads((sampled/'sample_manifest.json').read_text(encoding='utf-8'))
    samples = manifest['samples'][:args.limit]
    assert samples and 1 <= args.limit <= len(manifest['samples'])
    native_records = {r['image_id']:r for r in json.loads((contoured/'contour_predictions.json').read_text(encoding='utf-8'))['models'][args.model]}
    flipped_ref = {r['image_id']:r['pred'] for r in json.loads((sampled/args.model/'hflip_predictions.json').read_text(encoding='utf-8'))}
    inventory = source_inventory()
    adapter_sha = sha(REPO/'deployment/strong_fast.py')
    status = dict(phase='initializing', model=args.model, checkpoint_sha256=sha(checkpoint),
                  source_sha256=inventory, adapter_sha256=adapter_sha, script_sha256=sha(__file__),
                  samples=samples, full_VAL=False, original_weights=True, variants={})
    write(out/'probe_status.json',status)
    assert status['checkpoint_sha256'] == cfg[args.model+'_checkpoint_sha256']
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    from apple_seg.od_training import load_checkpoint
    from apple_seg.od_model import build_od_amodal_detector
    from apple_seg.od_inference import _strict_restore
    from apple_seg.od_data import legacy_rgb_tensor
    from apple_seg.od_evaluate import _validate_prediction
    from apple_seg.od_roi import paste_roi_mask
    from deployment.strong_fast import FastDeploymentAdapter
    cases = {
        'fp32_filter': dict(precision='fp32',chunk=8,threshold=.75,roi_precision='fp32',pair=False),
        'fp32_chunk32': dict(precision='fp32',chunk=32,threshold=.75,roi_precision='fp32',pair=False),
        'fp16_chunk32': dict(precision='fp16',chunk=32,threshold=.35,roi_precision='autocast',pair=False),
        'fp16_chunk64': dict(precision='fp16',chunk=64,threshold=.35,roi_precision='autocast',pair=False),
        'fp16_pair64': dict(precision='fp16',chunk=64,threshold=.35,roi_precision='autocast',pair=True),
        'bf16_chunk64': dict(precision='bf16',chunk=64,threshold=.35,roi_precision='autocast',pair=False),
        'fp16_roi32_fallback': dict(precision='fp16',chunk=32,threshold=.35,roi_precision='fp32',pair=False),
    }
    assert set(args.variants) <= set(cases)
    rgbs, references = [], []
    for sample in samples:
        with Image.open(sample['rgb_path']) as image:
            rgbs.append(image.convert('RGB'))
        path=contoured/native_records[sample['image_id']]['compact_roi_probabilities']
        with np.load(path,allow_pickle=False) as archive:
            references.append({key:archive[key].copy() for key in archive.files})
    payload=load_checkpoint(checkpoint)
    model=build_od_amodal_detector(weights=None,**payload['constructor_config'])
    status['strict_restore']=_strict_restore(model,payload['model'])
    status['constructor_config']=payload['constructor_config']
    model.requires_grad_(False).eval().to('cuda')
    status['gpu']=torch.cuda.get_device_name(0)
    def compare(pred,reference):
        selected=[i for i,s in enumerate(pred['scores']) if s>=.75]
        ref_scores=np.asarray(reference['scores'])
        expected=np.flatnonzero(ref_scores>=.75)
        arrays={'boxes':pred['boxes'].numpy(), 'scores':np.asarray(pred['scores'])}
        for branch in ('visible','amodal'):
            if branch in reference:
                arrays[branch]=pred[branch].numpy()
        result={'reference_at_0_75':len(expected),'candidate_at_0_75':len(selected),
                'same_count':len(selected)==len(expected),'direct_order_comparison':True}
        diffs={}
        if len(selected)==len(expected):
            for key,arr in arrays.items():
                ref=np.asarray(reference[key])
                if key=='boxes':ref=ref.reshape(-1,4)
                a,b=arr[selected],ref[expected]
                diffs[key]=float(np.max(np.abs(a-b),initial=0)) if a.shape==b.shape else None
            result['bitwise_equal']=all(np.array_equal(arr[selected],np.asarray(reference[key]).reshape(-1,4)[expected] if key=='boxes' else np.asarray(reference[key])[expected]) for key,arr in arrays.items())
        else:
            result['bitwise_equal']=False
        result['max_abs_differences']=diffs
        result['note']='Raw probabilities are compared in retained score order; not GT accuracy or a spatial matching metric.'
        return result
    for case in args.variants:
        spec=cases[case]
        result={'phase':'running','spec':spec,'frames':[]}
        status['variants'][case]=result
        status.update(phase='running',current=case)
        write(out/'probe_status.json',status)
        print('PROBE_START '+args.model+' '+case,flush=True)
        try:
            with FastDeploymentAdapter(model,pre_nms_threshold=spec['threshold'],roi_chunk_size=spec['chunk'],
                                       skip_dead_amodal=False,roi_precision=spec['roi_precision']) as adapter:
                dtype=torch.float16 if spec['precision']=='fp16' else torch.bfloat16
                def run(rgb):
                    torch.cuda.synchronize();start=time.perf_counter()
                    pixels=legacy_rgb_tensor(rgb,1024,1280)
                    hw=(rgb.height,rgb.width);input_hw=tuple(pixels.shape[-2:])
                    inputs=pixels[None].to('cuda')
                    if spec['pair']:inputs=torch.cat((inputs,inputs.flip(-1)),0)
                    mask=torch.ones((len(inputs),*input_hw),device='cuda',dtype=torch.bool)
                    with torch.inference_mode(),torch.autocast('cuda',enabled=spec['precision']!='fp32',dtype=dtype):
                        raw=adapter.predict_views(inputs,mask)
                    predictions=[]
                    for vi,output in enumerate(raw):
                        pred=_validate_prediction([output],input_hw,hw,mask_chunk_size=8)
                        if vi:
                            old=pred['boxes'].clone()
                            pred['boxes'][:,0]=hw[1]-old[:,2];pred['boxes'][:,2]=hw[1]-old[:,0]
                            for branch in ('visible','amodal'):pred[branch]=pred[branch].flip(-1).contiguous()
                        # Native contours only for the final display threshold.
                        rendered=[paste_roi_mask(pred[branch][i],pred['boxes'][i],hw,mask_threshold=.5)
                                  for i,s in enumerate(pred['scores']) if s>=.75 for branch in ('visible','amodal')]
                        predictions.append(pred)
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    return predictions,elapsed,deepcopy(adapter.last_stats)
                warm,_,_=run(rgbs[0]);del warm
                torch.cuda.reset_peak_memory_stats()
                for i,rgb in enumerate(rgbs):
                    predictions,ms,counts=run(rgb)
                    comparison=compare(predictions[0],references[i])
                    if spec['pair']:comparison['flipped_boxes_scores']=compare(predictions[1],flipped_ref[samples[i]['image_id']])
                    row=dict(image_id=samples[i]['image_id'],end_to_end_ms=ms,adapter_stats=counts,comparison=comparison)
                    result['frames'].append(row)
                    save={'image_id':samples[i]['image_id'],'views':[]}
                    for vi,pred in enumerate(predictions):
                        save['views'].append({'boxes':pred['boxes'].tolist(),'scores':pred['scores']})
                    write(out/case/(Path(samples[i]['image_id']).stem+'.json'),save)
                    print(f'PROBE {args.model} {case} {i+1}/{len(rgbs)} {ms:.2f}ms same={comparison["bitwise_equal"]}',flush=True)
                    del predictions
                times=[row['end_to_end_ms'] for row in result['frames']]
                result.update(phase='complete',mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),
                              max_ms=max(times),min_ms=min(times),fps_source_frames=1000/float(np.mean(times)),
                              all_bitwise_equal_at_0_75=all(r['comparison']['bitwise_equal'] for r in result['frames']),
                              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        except BaseException as error:
            result.update(phase='failed',error=str(error),traceback=traceback.format_exc())
            print('PROBE_FAIL '+case+' '+str(error),flush=True)
        write(out/'probe_status.json',status)
        gc.collect();torch.cuda.empty_cache()
    model.cpu()
    assert all(torch.equal(value,payload['model'][key]) for key,value in model.state_dict().items())
    assert source_inventory()==inventory and sha(REPO/'deployment/strong_fast.py')==adapter_sha
    status.update(phase='complete',weight_values_unchanged=True,source_before_after_identical=True)
    write(out/'probe_status.json',status)
    print('PROBE_COMPLETE '+json.dumps({name:{k:v for k,v in row.items() if k not in ('frames','traceback')} for name,row in status['variants'].items()},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
