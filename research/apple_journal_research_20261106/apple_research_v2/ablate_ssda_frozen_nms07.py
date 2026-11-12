"""One declared postprocessing ablation on frozen SSDA dev predictions; CPU only."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import json,hashlib,collections,datetime
ROOT=Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106')
SRC=ROOT/'experiments/resolution_pairs/ssda_v1'
ERR=ROOT/'experiments/error_analysis/ssda_frozen_v1'
OUT=ROOT/'experiments/postprocess_ablation/ssda_nms07_v1'
CONF=.25;MATCH=.5;NMS=.7

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def iou(a,b):
 inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
 return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1e-12)
def order(ps):return sorted(range(len(ps)),key=lambda k:(-ps[k]['score'],ps[k]['prediction_index']))
def nms_ref(ps):
 active=order(ps);keep=[];removed={}
 while active:
  pi=active.pop(0);keep.append(pi);rest=[]
  for pj in active:
   v=iou(ps[pi]['box_xyxy'],ps[pj]['box_xyxy'])
   if ps[pi]['class_id']==ps[pj]['class_id'] and v>NMS:removed[pj]={'suppressor_list_index':pi,'suppression_IoU':v}
   else:rest.append(pj)
  active=rest
 return keep,removed

def matching(ps,gs):
 used=set();matches=[]
 for pi in order(ps):
  candidates=[j for j,g in enumerate(gs) if j not in used and g['class_id']==ps[pi]['class_id']]
  if candidates:
   gi=max(candidates,key=lambda j:iou(ps[pi]['box_xyxy'],gs[j]['box_xyxy']))
   v=iou(ps[pi]['box_xyxy'],gs[gi]['box_xyxy'])
   if v>=MATCH:used.add(gi);matches.append({'prediction_index':ps[pi]['prediction_index'],'GT_index':gi,'GT_id':gs[gi]['GT_id'],'IoU':v})
 mid={x['prediction_index'] for x in matches};fp=[p['prediction_index'] for p in ps if p['prediction_index'] not in mid];fn=[i for i in range(len(gs)) if i not in used]
 per={}
 for c in [0,1]:
  n=sum(g['class_id']==c for g in gs);tp=sum(gs[x['GT_index']]['class_id']==c for x in matches);np=sum(p['class_id']==c for p in ps)
  per[str(c)]={'support':n,'TP':tp,'FP':np-tp,'FN':n-tp}
 return {'support':len(gs),'TP':len(matches),'FP':len(fp),'FN':len(fn),'per_class':per,'matches':matches,'false_positive_prediction_indices':fp,'missed_GT_indices':fn,'retained_predictions':len(ps)}

def main():
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'predictions').mkdir()
 protocol={'schema':'ssda_nms07_prespecified_engineering_ablation_v1','written_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':str(SRC),'resolutions':[640,960],'same_checkpoint':True,'all_images':113,'all_GT_each_resolution':230,'confidence_inclusive':CONF,'NMS_iou':NMS,'NMS_semantics':'Class-aware greedy descending unchanged confidence, original prediction_index breaks ties; suppress iff pair IoU > 0.7 (standard torchvision semantics), no maximum new detection limit.','match_iou_inclusive':MATCH,'matching':'All GT first; descending confidence stable original-index ties; best unused same-class GT.','reason':'Exploratory source review found same-class duplicate boxes; iou=.7 was originally requested in predict but inactive for end2end head. One engineering ablation, no method novelty claim.','baseline':'Original frozen NMS-free predictions; not overwritten.','no_other_NMS_thresholds':True,'no_confidence_tuning':True,'score_order_unchanged':True,'AP_not_computable_from_confidence_truncated_predictions':True,'CPU_threads_max':2,'CUDA_VISIBLE_DEVICES':'','source_labels_unmodified':True,'dataset_role':'Date-heldout development, physical fruit identity unknown; not independent test.'}
 write('protocol.json',protocol)
 # Protocol is persisted before reading per-image results or executing NMS.
 import torch,torchvision
 from torchvision.ops import nms
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 assert not torch.cuda.is_initialized()
 pairs=read(SRC/'paired_images.json');frozen=read(SRC/'summary.json');assert len(pairs)==113
 errorrows=read(ERR/'all_prediction_records.json');errmap={(x['image_id'],x['imgsz'],x['prediction_index']):x for x in errorrows}
 reports=[];audit=[];GTchanges=[];checks=[];inputhash={str(SRC/n):sha(SRC/n) for n in ['paired_images.json','summary.json','input_ground_truth.json']}
 # Scalar reference and Torchvision agree for overlapping/different-class/high-low-score known cases.
 synthetic=[{'prediction_index':i,'class_id':c,'score':s,'box_xyxy':b} for i,(c,s,b) in enumerate([(0,.9,[0,0,100,100]),(0,.8,[1,1,101,101]),(1,.7,[1,1,101,101]),(0,.6,[200,200,300,300]),(0,.5,[15,0,115,100]),(0,.4,[18,0,118,100])])]
 def tv_check(ps,tag):
  actual,_=nms_ref(ps);expected=[];checked=0;ties=0
  for c in [0,1]:
   ix=[i for i,p in enumerate(ps) if p['class_id']==c]
   if not ix:continue
   if len({ps[i]['score'] for i in ix})!=len(ix):ties+=1;continue
   boxes=torch.tensor([ps[i]['box_xyxy'] for i in ix],dtype=torch.float32,device='cpu');scores=torch.tensor([ps[i]['score'] for i in ix],dtype=torch.float32,device='cpu')
   out=[ix[k] for k in nms(boxes,scores,NMS).tolist()]
   got=[k for k in actual if ps[k]['class_id']==c]
   assert out==got,(tag,c,out,got)
   checked+=1
  return {'tag':tag,'distinct_score_class_groups_checked':checked,'tied_score_class_groups_skipped':ties}
 checks.append(tv_check(synthetic,'synthetic_distinct_scores'))
 # Explicit scalar tie behavior is tested without asserting torchvision's undefined tied-score order.
 tied=[dict(p) for p in synthetic[:2]];tied[0]['score']=.9;tied[1]['score']=.9;tied[0]['prediction_index']=9;tied[1]['prediction_index']=2
 assert nms_ref(tied)[0]==[1]
 for idx,pair in enumerate(pairs):
  for size in [640,960]:
   src=SRC/pair['resolutions'][str(size)]['prediction_file'];h=sha(src);assert h==pair['resolutions'][str(size)]['prediction_file_sha256'];inputhash[str(src)]=h
   d=read(src);ps=d['predictions'];gs=d['ground_truth'];assert all(p['score']>=CONF for p in ps)
   before=matching(ps,gs)
   assert {(m['prediction_index'],m['GT_index']) for m in before['matches']}=={(m['prediction_index'],m['GT_index']) for m in d['box_matching']['matches']}
   for key in ['support','TP','FP','FN']:assert before[key]==d['box_matching'][key]
   keep,removed=nms_ref(ps);retained=[ps[i] for i in range(len(ps)) if i in set(keep)];after=matching(retained,gs)
   checks.append(tv_check(ps,f"{idx}_{size}"))
   bg={x['GT_id']:x for x in before['matches']};ag={x['GT_id']:x for x in after['matches']};bp={x['prediction_index']:x for x in before['matches']};ap={x['prediction_index']:x for x in after['matches']}
   lost=sorted(set(bg)-set(ag));recovered=sorted(set(ag)-set(bg))
   for g in gs:
    GTchanges.append({'image_id':d['image_id'],'imgsz':size,'GT_id':g['GT_id'],'class_id':g['class_id'],'before_status':'TP' if g['GT_id'] in bg else 'FN','after_status':'TP' if g['GT_id'] in ag else 'FN','before_match':bg.get(g['GT_id']),'after_match':ag.get(g['GT_id'])})
   rankbefore={ps[i]['prediction_index']:rank+1 for rank,i in enumerate(order(ps))};rankafter={retained[i]['prediction_index']:rank+1 for rank,i in enumerate(order(retained))}
   for pi,p in enumerate(ps):
    suppressor=ps[removed[pi]['suppressor_list_index']] if pi in removed else None;er=errmap[(d['image_id'],size,p['prediction_index'])]
    audit.append({'image_id':d['image_id'],'imgsz':size,'prediction_index':p['prediction_index'],'class_id':p['class_id'],'confidence':p['score'],'box_xyxy':p['box_xyxy'],'before_confidence_rank':rankbefore[p['prediction_index']],'after_confidence_rank':rankafter.get(p['prediction_index']),'retained':pi not in removed,'suppressed_by_prediction_index':suppressor['prediction_index'] if suppressor else None,'suppressor_confidence':suppressor['score'] if suppressor else None,'suppression_IoU':removed[pi]['suppression_IoU'] if pi in removed else None,'before_status':'TP' if p['prediction_index'] in bp else 'FP','after_status':('TP' if p['prediction_index'] in ap else 'FP') if pi not in removed else 'suppressed','before_error_category':er['error_category'],'before_GT_match':bp.get(p['prediction_index']),'after_GT_match':ap.get(p['prediction_index'])})
   dest=f'predictions/{idx:05d}_{size}.json';write(dest,{'image_id':d['image_id'],'imgsz':size,'source_file':str(src),'source_sha256':h,'weights_sha256':d['weights_sha256'],'postprocess':'class_aware_NMS_IoU_gt_0.7','ground_truth':gs,'predictions':retained,'box_matching':after})
   reports.append({'image_id':d['image_id'],'imgsz':size,'before':before,'after':after,'suppressed_prediction_indices':[ps[i]['prediction_index'] for i in removed],'GT_TPs_lost':lost,'GTs_recovered':recovered,'postprocess_prediction_file':dest})
 total={}
 for size in [640,960]:
  rr=[r for r in reports if r['imgsz']==size];aa=[r for r in audit if r['imgsz']==size];tot={'images':len(rr)}
  for phase in ['before','after']:
   tot[phase]={key:sum(r[phase][key] for r in rr) for key in ['support','TP','FP','FN']};tot[phase]['per_class']={str(c):{key:sum(r[phase]['per_class'][str(c)][key] for r in rr) for key in ['support','TP','FP','FN']} for c in [0,1]}
  dropped=[a for a in aa if not a['retained']]
  tot['suppressed_total']=len(dropped);tot['suppressed_baseline_TP_predictions']=sum(a['before_status']=='TP' for a in dropped);tot['suppressed_baseline_FP_predictions']=sum(a['before_status']=='FP' for a in dropped);tot['suppressed_baseline_error_categories']=dict(collections.Counter(a['before_error_category'] for a in dropped));tot['GT_TPs_lost']=[g for r in rr for g in r['GT_TPs_lost']];tot['GTs_recovered']=[g for r in rr for g in r['GTs_recovered']]
  assert all(tot['before'][k]==frozen['counts'][str(size)][k] for k in ['support','TP','FP','FN']);assert tot['after']['support']==230
  total[str(size)]=tot
 assert not torch.cuda.is_initialized()
 write('all_prediction_suppression_audit.json',audit);write('all_GT_transitions.json',GTchanges);write('all_image_results.json',reports);write('reference_tests.json',{'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,'device':'cpu','num_threads':torch.get_num_threads(),'all_checks_passed':True,'deterministic_tie_test_passed':True,'tests':checks})
 result={'schema':'ssda_nms07_frozen_result_v1','status':'complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.json'),'source_predictions_unchanged':True,'input_file_hashes':inputhash,'counts':total,'reference_tests':'reference_tests.json','CPU_only':True,'GPU_initialized':False,'torchvision_distinct_score_class_groups_checked':sum(x['distinct_score_class_groups_checked'] for x in checks),'tied_score_class_groups_skipped':sum(x['tied_score_class_groups_skipped'] for x in checks),'scores_or_boxes_changed':False,'new_thresholds_tried':1,'AP_computed':False,'limitations':['Exploratory engineering ablation on previously inspected development data, not independent confirmation.','Only exported predictions with score>=0.25 are available; no AP claim or operating-threshold selection.','Postprocessing NMS can affect dense real orchard fruit/defect instances differently; these SSDA images do not establish deployment generalization.','No end-to-end device latency measured; this CPU audit does not establish NMS deployment overhead.']}
 write('summary.json',result)
 print(json.dumps({'status':'complete','output':str(OUT),'counts':total,'reference_groups':result['torchvision_distinct_score_class_groups_checked']},ensure_ascii=False))
if __name__=='__main__':main()
