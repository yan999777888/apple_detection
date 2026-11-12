from pathlib import Path
import hashlib,json,sys,zipfile,numpy as np,collections
R=Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106');O=R/'experiments/original_visible_gt_coco_v1';OLD=R.parent/'apple_detection';V=R/'runtime/coco_eval_2_0_11'
sys.path.insert(0,str(V/'site-packages'))
from pycocotools import mask as mu
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
s=read(O/'summary.json');checked={};wheel=V/'pycocotools-2.0.11-cp311-cp311-win_amd64.whl'
with zipfile.ZipFile(wheel) as z:
 for n in z.namelist():
  if n.startswith('pycocotools/') and not n.endswith('/'):
   path=V/'site-packages'/n;assert path.read_bytes()==z.read(n);checked[n]=sha(path)
allstats={};candidate={}
for branch,b in s['branches'].items():
 ps=read(O/(branch+'_visible_predictions.coco.json'));assert len(ps)==b['detections'];assert sum(mu.area(r['segmentation'])==0 for r in ps)==b['empty_masks_retained']
 assert all(np.array_equal(mu.toBbox(r['segmentation']),r['bbox']) for r in ps)
 framecounts=collections.Counter(r['image_id'] for r in ps)
 for task in ['bbox','segm']:
  z=np.load(O/(branch+'_'+task+'_arrays.npz'),allow_pickle=False);a=z['precision'];rr=z['recall'];mean=lambda x:float(x[x>-1].mean()) if (x>-1).any() else -1.
  recalcs={'AP':mean(a[:,:,:,0,2]),'AP50':mean(a[0,:,:,0,2]),'AP75':mean(a[5,:,:,0,2]),'AP_small':mean(a[:,:,:,1,2]),'AP_medium':mean(a[:,:,:,2,2]),'AP_large':mean(a[:,:,:,3,2]),'AR1':mean(rr[:,:,0,0]),'AR10':mean(rr[:,:,0,1]),'AR100':mean(rr[:,:,0,2]),'AR_small':mean(rr[:,:,1,2]),'AR_medium':mean(rr[:,:,2,2]),'AR_large':mean(rr[:,:,3,2])}
  assert all(abs(v-b['metrics'][task]['stats'][k])<1e-12 for k,v in recalcs.items());allstats[branch+'_'+task]=recalcs
 model,view=branch.split('_');m=read(OLD/'runs/stage2_t15_paired_20261102'/model/'prediction_metadata.json')
 candidate[branch]={'upstream_candidate_policy':m['deployment']['candidate_policy_semantics'],'upstream_stage':m['deployment']['candidate_selection_stage'],'final_head_score_comparison':'> 0.001 on reclassified ROI softmax foreground score','initial_score_is_not_final_score':True,'initial_classifier':'base_logits softmax(-1)[:,1]','final_classifier':'ROI class_logits softmax(-1)[:,1]','source_evidence':'deployment/pre_mask.py lines 49-64; hash bound in original metadata and summary','pre_mask_nms_topk':m['constructor_config']['pre_mask_nms_topk'],'detections_per_img_cap':m['constructor_config']['detections_per_img'],'effective_pre_ROI_cap':64,'final_nms':{'method':'dual mask','suppress_when_both':'amodal IoU > .5 AND visible IoU > .45','mask_probability_threshold':.5},'final_cache_detections':len(ps),'actual_final_score_min':min(r['score'] for r in ps),'actual_final_score_max':max(r['score'] for r in ps),'actual_per_image_max':max(framecounts.values()),'actual_zero_detection_images':814-len(framecounts),'COCO_maxDets':[1,10,100],'unavailable_discarded_candidates_cannot_be_recovered':True,'not_strictly_matched_candidate_policy_with_new_YOLO_conf001_max300':True}
result={'schema':'visible_coco_numeric_and_candidate_verification_v1','summary_sha256':sha(O/'summary.json'),'all_96_COCO_stats_recomputed_from_saved_arrays':True,'all_18125_prediction_bboxes_recomputed_from_RLE':sum(x['detections'] for x in s['branches'].values()),'all_empty_masks_retained':sum(x['empty_masks_retained'] for x in s['branches'].values()),'official_wheel_installed_file_bytes_verified':checked,'wheel_sha256':sha(wheel),'recomputed_stats':allstats,'candidate_policy_audit':candidate,'no_GPU_or_model_loaded':True,'limitations':['This verifies saved standard COCO metrics and visible geometry, not independent inference reproduction.','Four independent branch outputs do not constitute old Small+Base fused segmentation.','Different upstream candidate policies limit attribution of quality differences even when GT and COCO maxDets match.']}
with (O/'numeric_and_candidate_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'passed':True,'stats_checked':96,'predictions_checked':sum(x['detections'] for x in s['branches'].values()),'wheel_files':len(checked),'per_image_max':{k:v['actual_per_image_max'] for k,v in candidate.items()}}))
