"""CPU-only frozen original-visible GT and four explicit historical ROI branches.
Does not reconstruct, tune, or claim any fused historical segmentation model.
"""
from __future__ import annotations
import argparse, collections, contextlib, datetime, hashlib, importlib.util, io, json, os
from pathlib import Path
import sys

ROOT = Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106')
BASE = ROOT.parent
OLD = BASE / 'apple_detection'
RUN = OLD / 'runs/stage2_t15_paired_20261102'
VENDOR = ROOT / 'runtime/coco_eval_2_0_11/site-packages'
OUT = ROOT / 'experiments/original_visible_gt_coco_v1'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, x):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def configure():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
    for k in ['TEMP','TMP','TMPDIR']: os.environ[k]=str(ROOT/'runtime/tmp')
    sys.path.insert(0,str(VENDOR))
def rle_json(mask,mu):
    import numpy as np
    r=mu.encode(np.asfortranarray(mask,dtype=np.uint8));r['counts']=r['counts'].decode('ascii'); return r

def evaluate(gt_path, predictions, out_prefix):
    """COCO bbox and segm independently; segm loadRes receives no bbox (mask area)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    results={}
    for task in ('bbox','segm'):
        rows=[{k:r[k] for k in ['image_id','category_id','score', 'bbox' if task=='bbox' else 'segmentation']} for r in predictions]
        s=io.StringIO()
        with contextlib.redirect_stdout(s):
            g=COCO(str(gt_path));d=g.loadRes(rows);ev=COCOeval(g,d,task)
            ev.params.imgIds=sorted(g.imgs); ev.params.catIds=[1]; ev.params.maxDets=[1,10,100]
            ev.evaluate();ev.accumulate();ev.summarize()
        names=['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
        results[task]={'stats':dict(zip(names,map(float,ev.stats))), 'iou_thresholds':ev.params.iouThrs.tolist(),
            'recall_thresholds':ev.params.recThrs.tolist(),'maxDets':ev.params.maxDets,'area_ranges':ev.params.areaRng,
            'area_range_labels':ev.params.areaRngLbl,'images':len(g.imgs),'GT':len(g.anns),'predictions':len(rows),
            'category_ids':[1],'category_name':'apple','category_collapsed':True,'score_filter_added':False,
            'precision_shape':list(ev.eval['precision'].shape),'recall_shape':list(ev.eval['recall'].shape),
            'stats_not_available_sentinel':-1, 'segm_detection_area_from_RLE':task=='segm'}
        with Path(str(out_prefix)+'_'+task+'_cocoeval.txt').open('x',encoding='utf-8') as f:f.write(s.getvalue())
        with Path(str(out_prefix)+'_'+task+'_arrays.npz').open('xb') as f:
            np.savez_compressed(f,precision=ev.eval['precision'],recall=ev.eval['recall'],scores=ev.eval['scores'])
    write(Path(str(out_prefix)+'_metrics.json'),results)
    return results

def prepare():
    configure()
    import numpy as np, torch
    from PIL import Image,ImageDraw
    from pycocotools import mask as mu
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('GPU must remain unused')
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'source').mkdir();bound={}
    def bind(p, expected=None):
        p=Path(p).resolve()
        if p.drive.upper()!='F:':raise ValueError('All inputs must be on F')
        h=sha(p)
        if expected is not None and h!=expected:raise ValueError('Hash mismatch: '+str(p))
        bound[str(p)]=h
        return h
    protocol={
      'schema':'original_visible_gt_coco_protocol_v1','created_utc':now(),'source_split':'original PApple validation; no test',
      'independent_fruit_holdout':False,'evaluation_role':'development-only; original west/east split shares 309 derived fruit IDs; namespace not fully established',
      'GT':'3449 original author visible instances, 814 full native images, 13 empty images; includes 14 regions absent from later binary-stage labels',
      'GT_masks':'Original frozen packed visible masks, additionally checked against original VIA polygons rasterized with historical Pillow ImageDraw convention; disconnected polygons unioned by apple_ID within image',
      'GT_bbox':'Tight visible-raster support [min_x,min_y,max_x+1,max_y+1], converted to COCO xywh',
      'GT_categories':{'1':'apple'},'pred_categories':'All original apple detections -> category 1; later ripe/unripe collapse -> category 1 without GT-guided selection',
      'prediction_ROI':'Original native-oriented probabilities on amodal ROI support; no second resize/flip; original od_roi.paste_roi_mask at threshold >= 0.5, pixel-center bilinear grid_sample align_corners=False',
      'prediction_bbox':'Tight support of predicted visible raster; no GT used to derive mask, box, score or selection',
      'empty_predicted_masks':'Retained as zero-area RLE and [0,0,0,0] bbox, never silently dropped',
      'four_branches':['base_original','base_hflip','small_original','small_hflip'],
      'historical_fusion':'Only fused amodal boxes exist. These four new visible-mask adapter reports are separate branch results, NOT Small+Base fused segmentation.',
      'scores':'All saved detections, original FP32 scores from FP16 T15 inference; no added threshold/NMS/calibration. Already truncated by historical pre-mask score .15, NMS .8, topk 64 and final dual-mask NMS. This is not untruncated PR.',
      'COCO':'Official pycocotools 2.0.11; IoU .50:.05:.95, 101 recall points, maxDets [1,10,100], one category; segm loadRes receives only RLE to prevent box-area substitution',
      'future_model_rule':'Use same 814 image hashes and original visible GT. Full native visible binary masks at fixed inference policy, category-collapse ripe/unripe to apple, derive tight boxes from same mask. No GT-threshold search or cherry-picking. Report native YOLO detection-head boxes only as separate geometry definition.',
      'historical_ROI_integrity_limitation':'Old metadata did not store per-NPZ SHA. Now hash every NPZ, require boxes and scores bit-exact to hash-bound prediction JSON and verify producer source/metadata, freeze current ROI bytes; cannot retroactively prove ROI archive hashes at original creation.',
      'original_GT_is_not_physically_independent_test':True,'no_model_loaded':True,'GPU_used':False}
    write(OUT/'protocol.json',protocol)
    bind(Path(__file__))
    provenance=BASE/'Apple_DualBranch_3080_20261022/assets/papple_val_provenance.json'
    p=read(provenance);bind(provenance,'fcd96d08c75e1663c83efecb8eb321bbc10220dc74314c8227f1f8565a2ccb3a')
    frozen_images=p['image_metadata'];assert len(frozen_images)==814
    truth_by_image=collections.defaultdict(list)
    for row in p['ground_truth_instances']:truth_by_image[row['image_id']].append(row)
    visible_file=BASE/'apple_vision/local_amodal/data/papple/data/gt_json/val/via_region_data_instance.json'
    bind(visible_file,p['binding']['current_annotation_sha256']['visible'])
    raw=read(visible_file);via={x['filename']:x for x in raw.get('_via_img_metadata',raw).values()}
    assert set(via)=={x['image_id'] for x in frozen_images}
    metadata={}
    for model in ['base','small']:
        mp=RUN/model/'prediction_metadata.json';m=read(mp);bind(mp);metadata[model]=m
        assert m['phase']=='complete' and m['roi_masks_saved'] and m['images']==814 and m['GT']==3449
        assert m['provenance_sha256']==sha(provenance)
        for rel,h in m['source_sha256'].items():
            path=OLD/rel
            if str(path.resolve()) not in bound:bind(path,h)
            else:assert bound[str(path.resolve())]==h
        for view,h in m['predictions_sha256'].items():bind(RUN/model/(view+'_predictions.json'),h)
        assert m['roi_mask_archive_convention']=='Native image-oriented ROI probabilities, with native XYXY support; hflip inverted for BOTH boxes and masks'
    cfg=read(OLD/'configs/strong_baseline_4060.json');bind(OLD/'configs/strong_baseline_4060.json')
    for model,m in metadata.items():
        weight=(OLD/cfg['paths'][model+'_checkpoint']).resolve();bind(weight,m['checkpoint_sha256'])
    bind(RUN/'fused_predictions.json','bdbd7c921044a13a4548af06b61c65aece9c8148cea6a6758f4063e37983c3b5')
    module_path=OLD/'strong_runtime/src/apple_seg/od_roi.py'
    snapshot=OUT/'source/od_roi.py';snapshot.write_bytes(module_path.read_bytes())
    spec=importlib.util.spec_from_file_location('_frozen_old_roi_cpu',snapshot);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    # Frozen wrapper reads only probabilities/boxes; no model/checkpoint execution.
    images=[];anns=[];input_manifest=[];gt_cache_manifest=[];regions=0
    gtroot=BASE/'apple_vision/local_amodal/cache/targets/papple_v1_20261019/val'
    rgbroot=BASE/'apple_vision/local_amodal/data/papple/data/images/val'
    newroot=ROOT/'apple_research_v2/datasets/papple_visible_stage_seg_v1/images/val'
    for index,frozen in enumerate(frozen_images):
        name=frozen['image_id'];h,w=frozen['height'],frozen['width'];assert(h,w)==(1300,1300)
        image=rgbroot/name;bind(image,frozen['RGB_sha256']);bind(newroot/name,frozen['RGB_sha256'])
        input_manifest.append({'image_id':index+1,'filename':name,'image_path':str(image),'new_model_image_path':str(newroot/name),'sha256':frozen['RGB_sha256'],'height':h,'width':w})
        images.append({'id':index+1,'file_name':name,'height':h,'width':w})
        cp=gtroot/f'{index:06d}.npz';bind(cp,frozen['cache_sha256'])
        gt_cache_manifest.append({'index':index,'path':str(cp),'sha256':bound[str(cp.resolve())]})
        rows=truth_by_image[name];ids=frozen['apple_ids_in_target_order'];assert ids==[r['apple_id'] for r in rows]
        regs=via[name]['regions'];polys=collections.defaultdict(list)
        for reg in (regs.values() if isinstance(regs,dict) else regs):
            attr=reg['shape_attributes'];poly=np.asarray(list(zip(attr['all_points_x'],attr['all_points_y'])),dtype=np.float32)
            assert poly.ndim==2 and poly.shape[1]==2 and len(poly)>=3 and np.isfinite(poly).all()
            polys[str(reg['region_attributes']['apple_ID']).strip()].append(poly)
        assert set(ids)==set(polys)
        with np.load(cp,allow_pickle=False) as z:
            assert tuple(z['mask_shape'])==(len(ids),h,w)
            masks=np.unpackbits(z['visible_masks_packed'],bitorder='little',count=len(ids)*h*w).reshape(len(ids),h,w)
            for j,(instance_id,tr) in enumerate(zip(ids,rows,strict=True)):
                v=masks[j];assert hashlib.sha256(v.tobytes()).hexdigest()==tr['visible_mask_uint8_sha256']
                canvas=Image.new('L',(w,h),0);draw=ImageDraw.Draw(canvas)
                for poly in polys[instance_id]:draw.polygon([tuple(map(float,point)) for point in poly],fill=1)
                assert np.array_equal(v,np.asarray(canvas,dtype=np.uint8)),(name,instance_id,'original VIA differs')
                enc=rle_json(v,mu);area=int(mu.area(enc));assert area==tr['visible_area_pixels'] and area>0
                bbox=mu.toBbox(enc).tolist();regions+=1
                anns.append({'id':regions,'image_id':index+1,'category_id':1,'segmentation':enc,'area':area,'bbox':bbox,'iscrowd':0,'original_apple_id':instance_id})
        if (index+1)%100==0:print(f'GT {index+1}/814 checked',flush=True)
    assert len(anns)==3449 and sum(not truth_by_image[x['image_id']] for x in frozen_images)==13
    gt={'info':{'description':'Frozen original visible-only PApple development GT; not maturity or amodal GT','version':'1.0'},'licenses':[], 'images':images,'annotations':anns,'categories':[{'id':1,'name':'apple','supercategory':'fruit'}]}
    gt_path=OUT/'original_visible_gt.coco.json';write(gt_path,gt);write(OUT/'image_manifest.json',input_manifest);write(OUT/'gt_cache_manifest.json',gt_cache_manifest)
    summaries={};cache_manifest=[]
    for model in ['base','small']:
      for view in ['original','hflip']:
        key=model+'_'+view; branch=RUN/model/'roi_masks'/view;records=read(RUN/model/(view+'_predictions.json'))
        assert len(records)==814 and len(list(branch.glob('*.npz')))==814
        predictions=[];empty=0;total=0;score_min=1.;score_max=0.
        for index,(record,frozen) in enumerate(zip(records,frozen_images,strict=True)):
            assert record['image_id']==frozen['image_id'];cp=branch/f'{index:06d}.npz';h=bind(cp)
            cache_manifest.append({'branch':key,'index':index,'image_id':index+1,'filename':frozen['image_id'],'path':str(cp),'sha256':h,'bytes':cp.stat().st_size})
            with np.load(cp,allow_pickle=False) as z:
                assert set(z.files)=={'boxes','scores','visible','amodal'}
                boxes,scores,v,a=(z[x] for x in ['boxes','scores','visible','amodal']);n=len(scores)
                assert boxes.dtype==scores.dtype==v.dtype==a.dtype==np.float32
                assert boxes.shape==(n,4) and scores.shape==(n,) and v.shape==a.shape==(n,56,56)
                assert np.array_equal(boxes,np.asarray(record['pred']['boxes'],dtype=np.float32).reshape(-1,4))
                assert np.array_equal(scores,np.asarray(record['pred']['scores'],dtype=np.float32))
                assert np.isfinite(boxes).all() and (boxes[:,2:]>boxes[:,:2]).all()
                for x in [scores,v,a]:assert np.isfinite(x).all() and (x>=0).all() and (x<=1).all()
                for j in range(n):
                    tile,window=mod.paste_roi_mask(torch.from_numpy(v[j]),torch.from_numpy(boxes[j]),(1300,1300),mask_threshold=.5)
                    l,t,r,b=window;full=np.zeros((1300,1300),dtype=np.uint8,order='F');full[t:b,l:r]=tile.numpy()
                    enc=rle_json(full,mu);area=int(mu.area(enc));empty+=int(area==0);total+=1;score=float(scores[j]);score_min=min(score_min,score);score_max=max(score_max,score)
                    predictions.append({'image_id':index+1,'category_id':1,'score':score,'bbox':mu.toBbox(enc).tolist(),'segmentation':enc,'visible_area_pixels':area,'source_detection_index':j})
            if (index+1)%200==0:print(f'{key} {index+1}/814 converted',flush=True)
        pred_path=OUT/(key+'_visible_predictions.coco.json');write(pred_path,predictions)
        metrics=evaluate(gt_path,predictions,OUT/key)
        summaries[key]={'images':814,'detections':total,'empty_masks_retained':empty,'score_min':score_min,'score_max':score_max,'predictions_sha256':sha(pred_path),'metrics':metrics,'model_checkpoint_sha256':metadata[model]['checkpoint_sha256'],'cache_projection_is_new_adapter':True,'historical_fused_result':False}
        print(key,json.dumps({'count':total,'empty':empty,'bboxAP':metrics['bbox']['stats']['AP'],'segmAP':metrics['segm']['stats']['AP']}),flush=True)
    write(OUT/'roi_cache_manifest.json',cache_manifest)
    # Recheck input files after evaluation, including every source, image and ROI archive.
    for filename,h in bound.items():
        if sha(filename)!=h:raise ValueError('Input changed while evaluating '+filename)
    wheel=ROOT/'runtime/coco_eval_2_0_11/pycocotools-2.0.11-cp311-cp311-win_amd64.whl'
    report={'schema':'original_visible_gt_four_branch_coco_audit_v1','phase':'complete','completed_utc':now(),'script_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'gt_sha256':sha(gt_path),'images':814,'GT_original_visible':3449,'empty_GT_images':13,'GT_masks_match_original_polygon_rasterization':3449,'new_model_image_copies_match_original_SHA256':814,'roi_archives':len(cache_manifest),'roi_bytes':sum(x['bytes'] for x in cache_manifest),'input_files_hashed_before_after':len(bound),'all_input_bytes_unchanged':True,'gpu_initialized':torch.cuda.is_initialized(),'torch_threads':torch.get_num_threads(),'numpy':np.__version__,'torch':torch.__version__,'pycocotools':'2.0.11','wheel_sha256':sha(wheel),'branches':summaries,'historical_fused_masks_exist':False,'limitations':protocol}
    write(OUT/'bound_inputs_sha256.json',bound);write(OUT/'summary.json',report)
    print(json.dumps({'output':str(OUT),'phase':'complete','images':814,'GT':3449,'archives':len(cache_manifest)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evaluate-predictions',type=Path);ap.add_argument('--output-prefix',type=Path);args=ap.parse_args()
    if args.evaluate_predictions:
        configure()
        if not args.output_prefix or args.output_prefix.resolve().drive.upper()!='F:':ap.error('F output-prefix required')
        predictions=read(args.evaluate_predictions)
        known={r['image_id'] for r in read(OUT/'image_manifest.json')}
        for p in predictions:
            if p['image_id'] not in known or p['category_id']!=1:raise ValueError('Use frozen image map and apple=1')
            if p['segmentation']['size']!=[1300,1300]:raise ValueError('Only native original masks')
        evaluate(OUT/'original_visible_gt.coco.json',predictions,args.output_prefix)
    else:prepare()
