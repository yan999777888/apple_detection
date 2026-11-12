"""CPU evaluation of newly exported native visible masks against frozen original GT.
Requires full image coverage in a hash-bound provenance sidecar; no inference/training.
"""
from __future__ import annotations
import argparse,importlib.util,json,math,os,sys
from pathlib import Path
ROOT=Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106')
FROZEN=ROOT/'experiments/original_visible_gt_coco_v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--provenance',type=Path,required=True);p.add_argument('--output-directory',type=Path,required=True);a=p.parse_args()
    for x in [a.predictions,a.provenance,a.output_directory]:
        if x.resolve().drive.upper()!='F:':raise ValueError('All inputs/outputs must be on F')
    module_path=ROOT/'apple_research_v2/prepare_original_visible_coco.py'
    spec=importlib.util.spec_from_file_location('_frozen_visible_coco',module_path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.configure()
    import numpy as np
    from pycocotools import mask as mu
    report=m.read(FROZEN/'summary.json')
    assert report['phase']=='complete' and report['GT_original_visible']==3449 and report['images']==814
    assert m.sha(module_path)==report['script_sha256']
    assert m.sha(FROZEN/'original_visible_gt.coco.json')==report['gt_sha256']
    assert m.sha(FROZEN/'protocol.json')==report['protocol_sha256']
    prov=m.read(a.provenance);pred=m.read(a.predictions);manifest=m.read(FROZEN/'image_manifest.json')
    assert prov['predictions_sha256']==m.sha(a.predictions)
    assert prov['source_image_manifest_sha256']==m.sha(FROZEN/'image_manifest.json')
    assert prov['ground_truth_sha256']==report['gt_sha256']
    assert prov['ground_truth_used_for_inference'] is False
    assert sorted(prov['evaluated_image_ids'])==list(range(1,815))
    for key in ['mask_binary_rule','confidence_policy','inference_resize_policy','source_class_mapping','duplicate_handling_policy']:
        if not prov.get(key):raise ValueError('Missing explicit policy: '+key)
    weight=Path(prov['model_weights_path']).resolve()
    assert weight.drive.upper()=='F:' and m.sha(weight)==prov['model_weights_sha256']
    known={r['image_id']:r for r in manifest};assert set(known)==set(range(1,815))
    for r in manifest:
        assert m.sha(r['new_model_image_path'])==r['sha256']
    empty=0;source_counts={}
    for r in pred:
        assert r['image_id'] in known and r['category_id']==1
        assert isinstance(r['score'],(int,float)) and not isinstance(r['score'],bool) and math.isfinite(r['score']) and 0<=r['score']<=1
        enc=r['segmentation'];assert enc['size']==[1300,1300] and isinstance(enc['counts'],str)
        assert mu.decode(enc).shape==(1300,1300)
        box=mu.toBbox(enc);assert np.allclose(np.asarray(r['bbox']),box,atol=1e-9,rtol=0),'Box must derive from same visible mask'
        assert np.isfinite(box).all();empty+=int(mu.area(enc)==0)
        source=str(r['source_category_id']);assert str(prov['source_class_mapping'][source])=='1'
        source_counts[source]=source_counts.get(source,0)+1
    a.output_directory.mkdir(parents=True,exist_ok=False)
    result=m.evaluate(FROZEN/'original_visible_gt.coco.json',pred,a.output_directory/'new_model_original_visible')
    m.write(a.output_directory/'evaluation_provenance.json',{'schema':'new_model_original_visible_gt_coco_v1','phase':'complete','predictions_path':str(a.predictions),'predictions_sha256':m.sha(a.predictions),'export_provenance':prov,'source_image_manifest_sha256':m.sha(FROZEN/'image_manifest.json'),'GT_sha256':report['gt_sha256'],'protocol_sha256':report['protocol_sha256'],'visible_GT':3449,'images':814,'predictions':len(pred),'empty_predictions_retained':empty,'source_class_counts':source_counts,'category_collapsed_to_apple':True,'image_hashes_checked':814,'GPU_used':False,'independent_fruit_holdout':False,'no_added_score_filter_or_NMS':True,'script_sha256':m.sha(__file__),'metrics':result})
    print(json.dumps({'phase':'complete','output':str(a.output_directory),'bboxAP':result['bbox']['stats']['AP'],'segmAP':result['segm']['stats']['AP']}))
if __name__=='__main__':main()
