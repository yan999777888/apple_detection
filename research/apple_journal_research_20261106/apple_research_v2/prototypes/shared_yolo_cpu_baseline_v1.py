"""CPU-only engineering prototype: one official COCO trunk, two fresh task heads.

This is hard parameter sharing with source-aware loss gating, an established
multi-task baseline, not a trained joint model or a claimed methodological novelty.
No optimizer, GPU, download, dataset training, model export, or accuracy evaluation.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
sys.dont_write_bytecode=True
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',YOLO_AUTOINSTALL='False',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
ROOT=Path(__file__).resolve().parents[2]
TASK_FOR_SOURCE={'mendeley_stage3':'stage','ssda_defect2':'defect'}
EXPECTED_HASHES={'stage':'f6874118f0e8aaaa4ee45d137e23c16fa32627a18aa4772776d81266ae9344e3','defect':'1d10a4a02a852c85bdd4ea5a34eed11f72c70367445ec6176cd411b9b8b674bc','official_seg':'361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f','official_detect':'9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'}
RUNTIME=ROOT/'runtime/shared_yolo_cpu_prototype_v1'
for variable in ('TEMP','TMP','YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR'):
 path=RUNTIME/variable.lower();path.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(path)
import torch
from torch import nn
torch.set_num_threads(2)
from ultralytics.nn.modules.head import Detect,Segment26
from ultralytics.utils.loss import E2ELoss,v8SegmentationLoss

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def fingerprint(state):
 h=hashlib.sha256()
 for name,tensor in sorted(state.items()):
  a=tensor.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def graph_signature(model):
 return [(type(layer).__name__,layer.i,layer.f,[(key,tuple(value.shape)) for key,value in layer.state_dict().items()]) for layer in model.model[:-1]]
def parameter_count(module):return sum(p.numel() for p in module.parameters())

class LossHeadView(nn.Module):
 """Temporary device/head/hyperparameter interface expected by upstream losses."""
 def __init__(self,head):
  super().__init__();self.model=nn.ModuleList([head]);self.args=SimpleNamespace(box=7.5,cls=.5,dfl=1.5,overlap_mask=False,epochs=40)

class SharedYOLOTaskPrototype(nn.Module):
 """Share backbone AND neck (layers0..22); preserve distinct label spaces."""
 def __init__(self,official_segment_model,stage_reference,defect_reference):
  super().__init__()
  reference=official_segment_model
  require(graph_signature(reference)==graph_signature(stage_reference)==graph_signature(defect_reference),'Trunk graph/shapes differ')
  self.trunk=nn.ModuleList(copy.deepcopy(list(reference.model[:-1]))).float().requires_grad_(True)
  self.save_indices=set(reference.save);self.feature_indices=tuple(reference.model[-1].f)
  require(self.feature_indices==(16,19,22),'Unexpected feature routing')
  channels=tuple(branch[0].conv.in_channels for branch in reference.model[-1].cv2)
  self.heads=nn.ModuleDict({'stage':Segment26(nc=3,nm=32,npr=64,reg_max=1,end2end=True,ch=channels),'defect':Detect(nc=2,reg_max=1,end2end=True,ch=channels)})
  self.names={task:{int(k):v for k,v in model.names.items()} for task,model in [('stage',stage_reference),('defect',defect_reference)]}
  self.head_weight_shape_checks={}
  for task,model in [('stage',stage_reference),('defect',defect_reference)]:
   head=self.heads[task];head.stride=reference.model[-1].stride.detach().clone().float();head.bias_init()
   saved=model.model[-1].state_dict();fresh=head.state_dict()
   require(set(saved)==set(fresh) and all(saved[key].shape==fresh[key].shape for key in saved),'Task head checkpoint shape incompatibility')
   self.head_weight_shape_checks[task]={'keys_checked':len(saved),'all_keys_and_shapes_match':True,'trained_weights_loaded_into_prototype':False,'head_initialization':'fresh random module plus standard bias_init; no trained head transplant','class_names':self.names[task]}
  # These are plain criterion objects, not duplicated backbone/head modules.
  self.criteria={'stage':E2ELoss(LossHeadView(self.heads['stage']),v8SegmentationLoss),'defect':E2ELoss(LossHeadView(self.heads['defect']))}
  self.trunk_calls=0;self.head_calls={'stage':0,'defect':0};self.criterion_calls={'stage':0,'defect':0}
 def shared_features(self,images):
  require(images.device.type=='cpu','Prototype is CPU-only');require(images.ndim==4 and images.shape[1]==3,'Expected BCHW RGB')
  require(images.shape[-1]%32==0 and images.shape[-2]%32==0,'Dimensions must be divisible by32')
  self.trunk_calls+=1;x=images;outputs=[]
  for layer in self.trunk:
   if layer.f!=-1:x=outputs[layer.f] if isinstance(layer.f,int) else [x if index==-1 else outputs[index] for index in layer.f]
   x=layer(x);outputs.append(x if layer.i in self.save_indices else None)
  features=[outputs[i] for i in self.feature_indices]
  require(all(feature is not None for feature in features),'Missing routed feature');return features
 def head_forward(self,task,features):
  require(task in self.heads,'Unknown head');self.head_calls[task]+=1
  return self.heads[task](list(features))
 def validate_batch(self,source,batch):
  require(source in TASK_FOR_SOURCE,'Unknown source: task labels cannot be inferred')
  task=TASK_FOR_SOURCE[source];expected={name:name==task for name in self.heads}
  require(batch.get('annotation_available')==expected,'Missing or mixed task annotation declaration')
  require(batch.get('source')==source,'Source declaration mismatch')
  for key in ('img','batch_idx','cls','bboxes'):require(key in batch,'Missing active target field '+key)
  images=batch['img'];require(images.device.type=='cpu' and images.is_floating_point(),'CPU float images required')
  require(torch.isfinite(images).all().item() and images.min().item()>=0 and images.max().item()<=1,'Images must be finite RGB in [0,1]')
  n=len(batch['cls']);require(batch['cls'].shape==(n,1) and batch['bboxes'].shape==(n,4) and batch['batch_idx'].numel()==n,'Target shape mismatch')
  require(torch.isfinite(batch['cls']).all().item() and torch.isfinite(batch['bboxes']).all().item(),'Nonfinite labels')
  require(torch.all(batch['cls']==batch['cls'].floor()).item() and torch.all((batch['cls']>=0)&(batch['cls']<len(self.names[task]))).item(),'Wrong source class IDs')
  require(torch.all((batch['batch_idx']>=0)&(batch['batch_idx']<len(images))).item(),'Invalid local batch indices')
  require(torch.all((batch['bboxes']>=0)&(batch['bboxes']<=1)).item() and torch.all(batch['bboxes'][:,2:]>0).item(),'Invalid normalized boxes')
  if task=='stage':
   require('masks' in batch and 'sem_masks' in batch,'Stage requires actual instance+semantic masks')
   require(batch['masks'].shape[0]==n and batch['sem_masks'].shape[0]==len(images),'Stage mask indexing mismatch')
  else:require('masks' not in batch and 'sem_masks' not in batch,'Do not fabricate segmentation labels for detection source')
  return task
 def source_loss(self,source,batch):
  require(self.training,'Training-mode loss required');task=self.validate_batch(source,batch)
  features=self.shared_features(batch['img']);preds=self.head_forward(task,features)
  self.criterion_calls[task]+=1;vector,logged_one2one=self.criteria[task](preds,batch)
  require(torch.isfinite(vector).all().item(),'Nonfinite source loss')
  return vector.sum(),{'active_task':task,'inactive_tasks_skipped':[x for x in self.heads if x!=task],'loss_vector':vector,'one2one_only_logging':logged_one2one}
 def predict_both(self,images):
  require(not self.training,'Eval mode required for prediction');features=self.shared_features(images)
  return {task:{'task':task,'class_names':self.names[task],'raw_prediction':self.head_forward(task,features)} for task in self.heads}

def build_prototype():
 paths={'stage':ROOT/'runs/stage_yolo26n_seg_640_e40_s20261106/weights/last.pt','defect':ROOT/'runs/ssda_yolo26n_detect_640_e40_s20261106/weights/last.pt','official_seg':ROOT/'assets/ultralytics_yolo26_coco_20261106/yolo26n-seg.pt','official_detect':ROOT/'assets/ultralytics_yolo26_coco_20261106/yolo26n.pt'}
 models={};receipts={}
 for name,path in paths.items():
  actual=sha(path);require(actual==EXPECTED_HASHES[name],'Unexpected checkpoint bytes: '+name);receipts[str(path)]=actual
  payload=torch.load(path,map_location='cpu',weights_only=False);models[name]=payload.get('ema') or payload['model']
 signatures={name:graph_signature(model) for name,model in models.items()}
 require(all(value==signatures['official_seg'] for value in signatures.values()),'Official/final trunk architecture differs')
 torch.manual_seed(20261106)
 prototype=SharedYOLOTaskPrototype(models['official_seg'],models['stage'],models['defect'])
 reference_trunk=nn.ModuleList(list(models['official_seg'].model[:-1])).float()
 expected_hash=fingerprint(reference_trunk.state_dict());actual_hash=fingerprint(prototype.trunk.state_dict());require(expected_hash==actual_hash,'Shared trunk not exact chosen official initialization')
 metadata={'source_checkpoint_sha256_before':receipts,'trunk_graph_shapes_match_all_four_models':True,'feature_indices':[16,19,22],'feature_channels':[64,128,256],'strides':[8,16,32],'shared_trunk_initialization':'Only official COCO yolo26n-seg.pt, converted to FP32; no averaging or trained checkpoint trunk transfer','official_trunk_tensor_sha256':actual_hash,'heads':prototype.head_weight_shape_checks,'parameter_counts':{'shared_trunk':parameter_count(prototype.trunk),'stage_head':parameter_count(prototype.heads['stage']),'defect_head':parameter_count(prototype.heads['defect']),'shared_total':parameter_count(prototype),'separate_reference_total':sum(parameter_count(models[k]) for k in ('stage','defect'))}}
 metadata['parameter_counts']['reduction_fraction']=1-metadata['parameter_counts']['shared_total']/metadata['parameter_counts']['separate_reference_total']
 return prototype,metadata

def fixture(source,size=96):
 """Explicit synthetic unit fixture; not an apple annotation or research sample."""
 task=TASK_FOR_SOURCE[source];torch.manual_seed(41 if task=='stage' else 43)
 batch={'source':source,'annotation_available':{x:x==task for x in ('stage','defect')},'img':torch.rand(1,3,size,size),'batch_idx':torch.tensor([0]),'cls':torch.tensor([[1.]]),'bboxes':torch.tensor([[.5,.5,.5,.5]])}
 if task=='stage':
  mask=torch.zeros(1,size//4,size//4);mask[:,size//16:3*size//16,size//16:3*size//16]=1
  batch['masks']=mask;batch['sem_masks']=(mask>0).long()
 return batch

def gradients(module):
 tensors=[p.grad for p in module.parameters() if p.grad is not None]
 return {'parameters_with_grad':len(tensors),'all_finite':all(torch.isfinite(g).all().item() for g in tensors),'nonzero_gradient_tensors':sum(bool(torch.count_nonzero(g).item()) for g in tensors),'gradient_abs_sum':sum(float(g.detach().abs().sum()) for g in tensors)}
def tensor_contract(value):
 if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'dtype':str(value.dtype),'all_finite':bool(torch.isfinite(value).all())}
 if isinstance(value,dict):return {str(k):tensor_contract(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [tensor_contract(v) for v in value]
 return str(type(value).__name__)

def self_check(model):
 results=[];parameter_hash_before=fingerprint(dict(model.named_parameters()))
 for source in TASK_FOR_SOURCE:
  task=TASK_FOR_SOURCE[source];inactive='defect' if task=='stage' else 'stage';model.train();model.zero_grad(set_to_none=True)
  inactive_before=fingerprint(model.heads[inactive].state_dict());before=(model.trunk_calls,dict(model.head_calls),dict(model.criterion_calls))
  loss,log=model.source_loss(source,fixture(source));loss.backward()
  shared=gradients(model.trunk);active=gradients(model.heads[task]);idle=gradients(model.heads[inactive])
  require(shared['all_finite'] and shared['nonzero_gradient_tensors']>0,'Shared trunk did not receive finite positive gradient')
  require(active['all_finite'] and active['nonzero_gradient_tensors']>0,'Active head gradient missing/nonfinite')
  require(idle['parameters_with_grad']==0,'Inactive head received fake-negative/other gradient')
  require(fingerprint(model.heads[inactive].state_dict())==inactive_before,'Inactive head values or BN buffers changed')
  require(model.trunk_calls-before[0]==1 and model.head_calls[task]-before[1][task]==1 and model.head_calls[inactive]==before[1][inactive],'Incorrect source routing')
  require(model.criterion_calls[task]-before[2][task]==1 and model.criterion_calls[inactive]==before[2][inactive],'Inactive criterion was called')
  results.append({'source':source,'active_task':task,'loss':float(loss.detach()),'loss_vector_length':log['loss_vector'].numel(),'shared_gradient':shared,'active_head_gradient':active,'inactive_head_gradient':idle,'inactive_parameters_and_BN_unchanged':True,'trunk_calls':1,'active_head_calls':1,'inactive_head_calls':0,'inactive_criterion_calls':0})
 # Bad annotations must fail BEFORE any feature or head forward.
 rejected=[]
 for case,source,batch in [('unknown_source','unknown',fixture('mendeley_stage3')),('missing_task_metadata','mendeley_stage3',fixture('mendeley_stage3')),('fake_segmentation_zeros','ssda_defect2',fixture('ssda_defect2'))]:
  if case=='missing_task_metadata':batch.pop('annotation_available')
  if case=='fake_segmentation_zeros':batch['masks']=torch.zeros(1,24,24)
  calls=model.trunk_calls
  try:model.source_loss(source,batch)
  except ValueError:require(model.trunk_calls==calls,'Invalid batch already executed trunk');rejected.append(case)
  else:raise AssertionError('Unsafe batch accepted: '+case)
 require(fingerprint(dict(model.named_parameters()))==parameter_hash_before,'Parameter values changed despite no optimizer')
 model.zero_grad(set_to_none=True);model.eval();calls=model.trunk_calls
 with torch.no_grad():prediction=model.predict_both(fixture('mendeley_stage3')['img'])
 require(model.trunk_calls-calls==1,'Both-head inference duplicated the trunk')
 stage_decoded=prediction['stage']['raw_prediction'][0][0];defect_decoded=prediction['defect']['raw_prediction'][0]
 require(stage_decoded.shape[-1]==38 and defect_decoded.shape[-1]==6,'Task output contract mismatch')
 require(torch.isfinite(stage_decoded).all().item() and torch.isfinite(defect_decoded).all().item(),'Nonfinite decoded output')
 return {'synthetic_source_gradient_tests':results,'rejected_before_forward':rejected,'inference_shared_trunk_calls_for_both_heads':1,'inference_output_contracts':{name:tensor_contract(output['raw_prediction']) for name,output in prediction.items()},'names':model.names,'optimizer_steps':0,'all_parameter_values_unchanged':True,'BN_note':'Active trunk/head BN buffers may update during the synthetic training-mode forward; no model checkpoint is saved. Inactive-head BN buffers are verified unchanged.'}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();output=args.output.resolve()
 require(output.drive.lower()=='f:' and output.is_relative_to(ROOT) and not output.exists(),'Output must be a NEW directory within the F project')
 torch.set_num_interop_threads(1);output.mkdir(parents=True)
 model,metadata=build_prototype();checks=self_check(model)
 after={path:sha(path) for path in metadata['source_checkpoint_sha256_before']};require(after==metadata['source_checkpoint_sha256_before'],'Original checkpoints changed')
 require(not torch.cuda.is_initialized(),'CUDA must remain uninitialized')
 report={'schema':'shared_yolo_cpu_engineering_prototype_v1','phase':'checks_passed','completed_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'runtime':{'torch':str(torch.__version__),'CPU_threads':torch.get_num_threads(),'CPU_interop_threads':torch.get_num_interop_threads(),'CUDA_initialized':False},'architecture':metadata,'checks':checks,'source_checkpoint_sha256_after':after,'trained_joint_model':False,'production_ready':False,'accuracy_evaluated':False,'speed_measured':False,'novelty_claim':False,'model_checkpoint_saved':False,'limitations':['Established hard-sharing/source-aware-loss baseline; no methodological novelty claim.','Fresh heads plus chosen official trunk are untrained for these joint tasks; trained checkpoint heads were inspected for shapes only.','Source-aware gating prevents missing TASK labels becoming negatives; it cannot fix omissions within the active task.','Mendeley three visible stage labels and SSDA two numeric defect-region labels have separate semantics; neither is physiological maturity or market grade.','SSDA harvested/black-background domain differs from field apples; shared BN/representation may cause negative transfer.','Current PApple author labels are two classes, not the Mendeley three-class ontology; future PApple integration requires a separately specified stage head/protocol.','Parameter sharing does not establish real latency or accuracy; larger real training/evaluation must be coordinated separately.']}
 (output/'prototype_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 print(json.dumps({'phase':report['phase'],'output':str(output/'prototype_report.json'),'parameter_counts':metadata['parameter_counts'],'source_tests':[{k:r[k] for k in ['source','loss','inactive_head_calls','inactive_criterion_calls','inactive_parameters_and_BN_unchanged']} for r in checks['synthetic_source_gradient_tests']],'optimizer_steps':0,'CUDA_initialized':False},ensure_ascii=False,indent=2))
if __name__=='__main__':main()