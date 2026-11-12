"""Export full PApple predictions for original-visible COCO comparison.
Only --run starts CUDA. --check-only validates geometry on synthetic CPU tensors.
Never reads GT polygons/masks or labels during inference. Does not measure FPS.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys,traceback
from pathlib import Path
ROOT=Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106')
FROZEN=ROOT/'experiments/original_visible_gt_coco_v1'
ULTRA=Path(r'F:\anaconda\envs\falldet\Lib\site-packages\ultralytics')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def environment(out,cpu):
 if cpu:os.environ['CUDA_VISIBLE_DEVICES']=''
 runtime=out/'runtime'
 for n in ['tmp','yolo','mpl','torch','hf','cache','cuda','extensions','pip']:(runtime/n).mkdir(parents=True,exist_ok=True)
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='False',YOLO_OFFLINE='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',YOLO_CONFIG_DIR=str(runtime/'yolo'),MPLCONFIGDIR=str(runtime/'mpl'),TORCH_HOME=str(runtime/'torch'),HF_HOME=str(runtime/'hf'),XDG_CACHE_HOME=str(runtime/'cache'),CUDA_CACHE_PATH=str(runtime/'cuda'),TORCH_EXTENSIONS_DIR=str(runtime/'extensions'),PIP_CACHE_DIR=str(runtime/'pip'),TEMP=str(runtime/'tmp'),TMP=str(runtime/'tmp'))
 sys.path.insert(0,str(ROOT/'runtime/coco_eval_2_0_11/site-packages'))
 os.chdir(ROOT/'apple_research_v2')
def source_hashes():
 return {str(p):sha(p) for p in [ULTRA/'models/yolo/segment/predict.py',ULTRA/'models/yolo/detect/predict.py',ULTRA/'utils/ops.py',ULTRA/'utils/nms.py',ULTRA/'engine/predictor.py']}
def predictor_class():
 from ultralytics.models.yolo.segment.predict import SegmentationPredictor
 from ultralytics.engine.results import Results
 from ultralytics.utils import ops
 class RetainEmptyNativePredictor(SegmentationPredictor):
  """Official retina restoration, retaining zero masks instead of deleting candidates."""
  def construct_result(self,pred,img,orig_img,img_path,proto):
   if not self.args.retina_masks:raise ValueError('This adapter requires native retina masks')
   if pred.shape[0]==0:masks=None
   else:
    pred[:,:4]=ops.scale_boxes(img.shape[2:],pred[:,:4],orig_img.shape)
    masks=ops.process_mask_native(proto,pred[:,6:],pred[:,:4],orig_img.shape[:2])
   return Results(orig_img,path=img_path,names=self.model.names,boxes=pred[:,:6],masks=masks)
 return RetainEmptyNativePredictor

def cpu_check(out):
 environment(out,True)
 import torch,numpy as np,ultralytics
 from types import SimpleNamespace
 from ultralytics.models.yolo.segment.predict import SegmentationPredictor
 from pycocotools import mask as mu
 torch.set_num_threads(2)
 assert ultralytics.__version__=='8.4.115' and not torch.cuda.is_initialized()
 cls=predictor_class();old=object.__new__(SegmentationPredictor);new=object.__new__(cls)
 for obj in [old,new]:obj.args=SimpleNamespace(retina_masks=True);obj.model=SimpleNamespace(names={0:'ripe',1:'unripe'})
 pred=torch.tensor([[1,1,7,7,.9,0,1,1,1,1],[1,1,7,7,.8,1,-1,-1,-1,-1]],dtype=torch.float32)
 img=torch.zeros((1,3,8,8));orig=np.zeros((8,8,3),np.uint8);proto=torch.ones((4,4,4))
 native=old.construct_result(pred.clone(),img,orig,'cpu.png',proto)
 adapted=new.construct_result(pred.clone(),img,orig,'cpu.png',proto)
 assert len(native.boxes)==1 and len(adapted.boxes)==2
 assert torch.equal(native.boxes.data[0],adapted.boxes.data[0]) and torch.equal(native.masks.data[0],adapted.masks.data[0])
 assert adapted.masks.data[1].sum()==0
 enc=mu.encode(np.asfortranarray(adapted.masks.data[0].numpy(),dtype=np.uint8));assert np.array_equal(mu.decode(enc),adapted.masks.data[0].numpy())
 report={'schema':'new_visible_export_cpu_preflight_v1','passed':True,'GPU_initialized':torch.cuda.is_initialized(),'model_loaded':False,'synthetic_check':'Official native predictor yields one foreground result; adapter keeps same result byte-exact plus zero-mask candidate; no filtering or GT selection','source_sha256':source_hashes(),'script_sha256':sha(__file__),'original_visible_GT_sha256':read(FROZEN/'summary.json')['gt_sha256'],'image_manifest_sha256':sha(FROZEN/'image_manifest.json'),'input_images':len(read(FROZEN/'image_manifest.json'))}
 write(out/'cpu_preflight.json',report);print(json.dumps({'preflight_passed':True,'GPU_initialized':False,'output':str(out)}))

def run(a,out):
 environment(out,False)
 import numpy as np,cv2,torch,ultralytics
 from ultralytics import YOLO
 from pycocotools import mask as mu
 torch.set_num_threads(2);cv2.setNumThreads(1)
 assert ultralytics.__version__=='8.4.115'
 check=read(a.preflight);assert check['passed'] and check['script_sha256']==sha(__file__) and check['source_sha256']==source_hashes()
 weights=a.weights.resolve();assert weights.drive.upper()=='F:' and weights.is_file() and sha(weights)==a.weights_sha256
 manifest=read(FROZEN/'image_manifest.json');gt_sha=read(FROZEN/'summary.json')['gt_sha256']
 assert len(manifest)==814 and sha(FROZEN/'image_manifest.json')==check['image_manifest_sha256']
 assert gt_sha==check['original_visible_GT_sha256']
 for row in manifest:assert sha(row['new_model_image_path'])==row['sha256']
 device=torch.device(a.device);assert device.type=='cuda' and torch.cuda.is_available()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=YOLO(str(weights));assert model.task=='segment';model.to(device);model.model.float().eval()
 names={int(k):str(v) for k,v in model.names.items()};assert names=={0:'ripe',1:'unripe'},names
 arguments={'device':a.device,'imgsz':640,'batch':1,'quantize':32,'conf':.001,'iou':.7,'max_det':300,'rect':False,'retina_masks':True,'augment':False,'agnostic_nms':False,'classes':None,'verbose':False,'save':False,'save_txt':False,'save_conf':False,'save_crop':False,'show':False,'project':str(out/'predict'),'name':'no_saved_images'}
 predictions=[];frames=[];empty=0;cls=predictor_class();sources_before=source_hashes()
 with torch.inference_mode():
  for row in manifest:
   image=cv2.imdecode(np.fromfile(row['new_model_image_path'],dtype=np.uint8),cv2.IMREAD_COLOR)
   assert image is not None and image.shape==(1300,1300,3)
   res=model.predict(source=image,predictor=cls,stream=False,**arguments);assert len(res)==1
   result=res[0];boxes=result.boxes.data.detach().cpu().numpy().copy();n=len(boxes)
   assert boxes.shape==(n,6) and np.isfinite(boxes).all() and np.isin(boxes[:,5],[0,1]).all()
   assert not model.predictor.model.fp16 and model.predictor.args.quantize==32
   assert getattr(model.predictor,'_feats',None) is None
   if n:
    masks=result.masks.data.detach().cpu().numpy();assert masks.shape==(n,1300,1300) and np.isin(masks,[0,1]).all()
    for j in range(n):
     enc=mu.encode(np.asfortranarray(masks[j],dtype=np.uint8));enc['counts']=enc['counts'].decode('ascii');area=int(mu.area(enc));empty+=int(area==0)
     predictions.append({'image_id':row['image_id'],'category_id':1,'source_category_id':int(boxes[j,5]),'score':float(boxes[j,4]),'bbox':mu.toBbox(enc).tolist(),'segmentation':enc,'visible_area_pixels':area,'source_detection_index':j,'source_head_box_xyxy':boxes[j,:4].tolist()})
   else:assert result.masks is None or len(result.masks)==0
   frames.append({'image_id':row['image_id'],'filename':row['filename'],'image_sha256':row['sha256'],'predictions':n})
   if len(frames)%50==0:print(f'EXPORT {len(frames)}/814 candidates={len(predictions)}',flush=True)
 torch.cuda.synchronize(device)
 assert sha(weights)==a.weights_sha256 and source_hashes()==sources_before
 assert all(sha(row['new_model_image_path'])==row['sha256'] for row in manifest)
 pred_path=out/'visible_predictions.coco.json';write(pred_path,predictions);write(out/'frame_manifest.json',frames)
 provenance={'schema':'new_PApple_visible_export_v1','phase':'complete','predictions_sha256':sha(pred_path),'source_image_manifest_sha256':sha(FROZEN/'image_manifest.json'),'ground_truth_sha256':gt_sha,'ground_truth_used_for_inference':False,'evaluated_image_ids':[r['image_id'] for r in frames],'model_weights_path':str(weights),'model_weights_sha256':a.weights_sha256,'source_class_mapping':{'0':1,'1':1},'source_class_names':names,'mask_binary_rule':'Pinned Ultralytics ops.process_mask_native: native pixel mask logits >0, uint8 then native head-box crop; empty masks retained by local adapter. No extra mask threshold or GT used.','confidence_policy':{'requested_conf':.001,'comparison':'>','max_det':300,'iou_argument':.7,'agnostic_nms_argument':False,'end2end_head':bool(getattr(model.model,'end2end',False)),'if_end2end':'Pinned non_max_suppression takes score>conf then first max_det rows; does not execute IoU NMS in that path','old_cached_policy_different':True},'inference_resize_policy':{'input':'full native 1300x1300 BGR','imgsz':640,'rect':False,'batch':1,'retina_masks':True,'external_preresize':False},'duplicate_handling_policy':'Native predictor candidate policy before category collapse; after collapse, no extra deduplication/NMS or score changes. Local adapter retains empty masks normally removed by official SegmentationPredictor.','predicted_bbox_rule':'Tight native visible binary-mask support, COCO xywh; native head boxes also retained as separate fields','predict_arguments':arguments,'precision':'FP32, TF32 disabled','images':814,'predictions':len(predictions),'empty_masks_retained':empty,'source_sha256':sources_before,'source_unchanged_after':True,'image_hashes_unchanged_after':814,'script_sha256':sha(__file__),'ultralytics':ultralytics.__version__,'torch':torch.__version__,'GPU':torch.cuda.get_device_name(device),'timing_measured':False,'independent_fruit_holdout':False,'old_fused_segmentation_baseline_exists':False,'comparison_limitation':'Common original-visible GT and COCO metrics; upstream candidate cutoff/caps/NMS differ from historical T15 caches. No claim of equal candidate coverage or causal architecture-only gain.'}
 provenance['native_API_would_drop_empty_masks']=empty
 provenance['evaluation_adapter_differs_from_deployment_timing_pipeline']='Only candidate empty-mask retention; deployment benchmark uses standard native API and excludes those masks. This export is not a timing measurement.'
 write(out/'export_provenance.json',provenance)
 filtered=[r for r in predictions if r['visible_area_pixels']>0];native_path=out/'native_filter_visible_predictions.coco.json';write(native_path,filtered)
 native_prov=dict(provenance);native_prov.update(predictions_sha256=sha(native_path),predictions=len(filtered),empty_masks_retained=0,duplicate_handling_policy='Derived from retained export by the pinned standard native API rule mask.amax>0; no further NMS or changes. Not a second inference run.',native_filter_view_derived_without_second_inference=True)
 write(out/'native_filter_export_provenance.json',native_prov)
 print(json.dumps({'phase':'complete','images':814,'predictions':len(predictions),'empty_masks_retained':empty,'output':str(out)}))

def main():
 ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--check-only',action='store_true');group.add_argument('--run',action='store_true');ap.add_argument('--output-directory',type=Path,required=True);ap.add_argument('--weights',type=Path);ap.add_argument('--weights-sha256');ap.add_argument('--preflight',type=Path);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
 out=a.output_directory.resolve()
 if out.drive.upper()!='F:':ap.error('Output must be on F')
 if a.run and not all([a.weights,a.weights_sha256,a.preflight]):ap.error('--run requires weights, weights-sha256 and preflight')
 out.mkdir(parents=True,exist_ok=False)
 try:
  if a.check_only:cpu_check(out)
  else:run(a,out)
 except BaseException as exc:
  write(out/'FAILED.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
