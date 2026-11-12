"""Exploratory development-only SSDA failure analysis from frozen predictions.
No inference/training/threshold fitting. CPU <=2 threads; source files read-only.
"""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,collections,math,datetime,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106')
SRC=ROOT/'experiments/resolution_pairs/ssda_v1'
DATA=ROOT/'apple_research_v2/datasets/ssda_date_v2'
OUT=ROOT/'experiments/error_analysis/ssda_frozen_v1'
CONF=.25;MATCH=.5

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def ious(a,b):
 a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
 if not len(a) or not len(b):return np.zeros((len(a),len(b)))
 x=np.maximum(a[:,None,:2],b[None,:,:2]);y=np.minimum(a[:,None,2:],b[None,:,2:]);inter=np.maximum(0,y-x).prod(axis=2)
 aa=(a[:,2:]-a[:,:2]).prod(axis=1);bb=(b[:,2:]-b[:,:2]).prod(axis=1)
 return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)
def geometry(b,hw):
 h,w=hw;x1,y1,x2,y2=b;bw=x2-x1;bh=y2-y1;assert bw>0 and bh>0
 area=bw*bh/(w*h);aspect=max(bw/bh,bh/bw);edge=min(x1/w,y1/h,1-x2/w,1-y2/h);s=640/max(h,w);side=math.sqrt(bw*bh)*s;thin=min(bw,bh)*s
 return {'normalized_area':area,'major_minor_aspect_ratio':aspect,'normalized_edge_clearance':edge,'sqrt_area_at_canonical640':side,'min_side_at_canonical640':thin,'area_bin':'<0.1%' if area<.001 else '0.1%-1%' if area<.01 else '1%-5%' if area<.05 else '>=5%','aspect_bin':'[1,2)' if aspect<2 else '[2,4)' if aspect<4 else '>=4','edge_bin':'<2%' if edge<.02 else '2%-10%' if edge<.1 else '>=10%','canonical640_size_bin':'<16px' if side<16 else '16-32px' if side<32 else '32-96px' if side<96 else '>=96px','canonical640_thin_bin':'minside<4px' if thin<4 else 'minside>=4px'}
def ioubin(x):return '[0,.1)' if x<.1 else '[.1,.3)' if x<.3 else '[.3,.5)' if x<.5 else '[.5,.75)' if x<.75 else '[.75,1]'
def summarize(rows,isgt):
 fields=['class_id','area_bin','aspect_bin','edge_bin','canonical640_size_bin','canonical640_thin_bin','camera','date']
 out={}
 for field in fields:
  cells=[]
  for cl in [None,0,1]:
   selected=[r for r in rows if cl is None or r['class_id']==cl]
   for value in sorted({str(r[field]) for r in selected}):
    rr=[r for r in selected if str(r[field])==value];tp=sum(x['status']=='TP' for x in rr);err=len(rr)-tp
    cells.append({'class_filter':cl,'bin':value,'support_or_predictions':len(rr),'TP':tp,'FN' if isgt else 'FP':err,'recall' if isgt else 'precision':tp/len(rr),'error_categories':dict(collections.Counter(x['error_category'] for x in rr if x['status']!='TP'))})
  out[field]=cells
 return out

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 protocol={'role':'Exploratory analysis of date-held-out development; not an independent test.','frozen_source':str(SRC),'confidence_inclusive':CONF,'match_iou_inclusive':MATCH,'matching':'descending prediction confidence, stable prediction-index ties, best unused same-class GT; all GT matched before stratification.','gt_diagnostics':'For unmatched GT only: same-class IoU>=.5 already claimed => matching competition; otherwise wrong-class IoU>=.5 => class disagreement; otherwise best all-class IoU [.1,.5) => localization candidate; <.1 => no near retained candidate.','fp_diagnostics':'Unmatched prediction: same-class GT IoU>=.5 => duplicate/competition; otherwise wrong-class GT IoU>=.5 => class disagreement; [.1,.5) => localization candidate; <.1 => no nearby annotated GT.','fixed_iou_bins':['[0,.1)','[.1,.3)','[.3,.5)','[.5,.75)','[.75,1]'],'area_bins_normalized':[0,.001,.01,.05,1],'aspect_bins':[1,2,4,'infinity'],'edge_clearance_bins':[0,.02,.1,1],'canonical640_size_bins_sqrtarea':[0,16,32,96,'infinity'],'canonical640_thin_min_side_px':4,'class_names':['class_0','class_1'],'no_semantic_class_mapping_inferred':True,'predictions_below_confidence_not_available':True,'no_GT_relabeling':True,'no_training_or_inference':True,'no_threshold_tuning':True,'CPU_threads_max':2,'no_GPU':True,'AP_recomputed':False,'bins_chosen_before_running_this_analysis':True,'visual_artifacts':'debug only, not publication figures'}
 write('analysis_protocol.json',protocol)
 summary=read(SRC/'summary.json');paired=read(SRC/'paired_images.json');gtinputs=read(SRC/'input_ground_truth.json');manifest=read(DATA/'manifest.json');preflight=read(SRC/'preflight.json')
 assert len(gtinputs)==len(paired)==summary['images']==113
 assert sha(DATA/'manifest.json')==preflight['source_paths_sha256'][str(DATA/'manifest.json')]
 ms={x['image_id']:x for x in manifest['samples']};gtmap={x['image_id']:x for x in gtinputs};assert set(gtmap)=={x['image_id'] for x in manifest['samples'] if x['partition']=='dev'}
 inputs={str(SRC/n):sha(SRC/n) for n in ['summary.json','paired_images.json','input_ground_truth.json','preflight.json']};inputs[str(DATA/'manifest.json')]=sha(DATA/'manifest.json')
 gtrows=[];prrows=[];imrows=[];loaded={};matches_by_size={640:{},960:{}}
 for pair in paired:
  image_id=pair['image_id'];info=gtmap[image_id];meta=ms[image_id];hw=info['native_hw'];assert hw==meta['native_hw']
  assert sha(Path(info['image_path']))==info['image_sha256']==meta['rgb_sha256'];assert sha(Path(info['label_path']))==info['label_sha256']==meta['label_sha256']
  assert len(info['ground_truth'])==len(meta['instances'])
  for g,s in zip(info['ground_truth'],meta['instances']):
   cx,cy,bw,bh=s['xywh_normalized'];h,w=hw;expect=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
   assert g['class_id']==s['class_id'] and np.allclose(expect,g['box_xyxy'],atol=1e-8,rtol=0)
  camera=meta['capture_filename_tokens']['camera'];date=meta['capture_filename_tokens']['date']
  for size in [640,960]:
   index=pair['resolutions'][str(size)];p=SRC/index['prediction_file'];assert sha(p)==index['prediction_file_sha256'];inputs[str(p)]=sha(p)
   d=read(p);assert d['image_id']==image_id and d['weights_sha256']==summary['weights_sha256'];assert d['ground_truth']==info['ground_truth'];loaded[(image_id,size)]=d
   gs=d['ground_truth'];ps=d['predictions'];assert all(x['score']>=CONF for x in ps)
   mat=ious([x['box_xyxy'] for x in ps],[x['box_xyxy'] for x in gs]);used=set();mp={};mg={}
   for pi in sorted(range(len(ps)),key=lambda k:(-ps[k]['score'],ps[k]['prediction_index'])):
    cand=[j for j,g in enumerate(gs) if j not in used and g['class_id']==ps[pi]['class_id']]
    if cand:
     gi=max(cand,key=lambda j:mat[pi,j])
     if mat[pi,gi]>=MATCH:used.add(gi);mp[pi]=gi;mg[gi]=pi
   observed={(ps[pi]['prediction_index'],gi) for pi,gi in mp.items()};expected={(x['prediction_index'],x['GT_index']) for x in d['box_matching']['matches']};assert observed==expected
   assert [ps[i]['prediction_index'] for i in range(len(ps)) if i not in mp]==d['box_matching']['false_positive_prediction_indices'];assert [i for i in range(len(gs)) if i not in mg]==d['box_matching']['missed_GT_indices']
   matches_by_size[size][image_id]={g['GT_id'] for i,g in enumerate(gs) if i in mg}
   common={'image_id':image_id,'image_path':info['image_path'],'image_sha256':info['image_sha256'],'label_sha256':info['label_sha256'],'known_same_second_group':info['known_group_id'],'physical_fruit_id':None,'camera':camera,'date':date,'imgsz':size}
   for gi,g in enumerate(gs):
    best=int(np.argmax(mat[:,gi])) if len(ps) else None;bestv=float(mat[best,gi]) if best is not None else 0.;same=[pi for pi,pred in enumerate(ps) if pred['class_id']==g['class_id']];other=[pi for pi,pred in enumerate(ps) if pred['class_id']!=g['class_id']];samev=max((float(mat[pi,gi]) for pi in same),default=0);otherv=max((float(mat[pi,gi]) for pi in other),default=0)
    if gi in mg:status='TP';cat='matched'
    else:
     status='FN';cat='same_class_matching_competition' if samev>=.5 else 'class_disagreement_high_overlap' if otherv>=.5 else 'localization_IoU_.3_to_.5' if bestv>=.3 else 'localization_IoU_.1_to_.3' if bestv>=.1 else 'no_near_retained_candidate_IoU_below_.1'
    gtrows.append(dict(common,GT_id=g['GT_id'],GT_index=gi,class_id=g['class_id'],box_xyxy=g['box_xyxy'],status=status,error_category=cat,matched_prediction_index=ps[mg[gi]]['prediction_index'] if gi in mg else None,matched_IoU=float(mat[mg[gi],gi]) if gi in mg else None,best_class_agnostic_IoU=bestv,best_class_agnostic_IoU_bin=ioubin(bestv),best_candidate_prediction_index=ps[best]['prediction_index'] if best is not None else None,best_candidate_class=ps[best]['class_id'] if best is not None else None,best_same_class_IoU=samev,best_other_class_IoU=otherv,**geometry(g['box_xyxy'],hw)))
   for pi,pred in enumerate(ps):
    best=int(np.argmax(mat[pi,:])) if len(gs) else None;bestv=float(mat[pi,best]) if best is not None else 0.;same=[gi for gi,g in enumerate(gs) if g['class_id']==pred['class_id']];other=[gi for gi,g in enumerate(gs) if g['class_id']!=pred['class_id']];samev=max((float(mat[pi,gi]) for gi in same),default=0);otherv=max((float(mat[pi,gi]) for gi in other),default=0)
    if pi in mp:status='TP';cat='matched'
    else:
     status='FP';cat='same_class_duplicate_or_competition' if samev>=.5 else 'class_disagreement_high_overlap' if otherv>=.5 else 'localization_IoU_.3_to_.5' if bestv>=.3 else 'localization_IoU_.1_to_.3' if bestv>=.1 else 'no_near_annotated_GT_IoU_below_.1'
    prrows.append(dict(common,prediction_index=pred['prediction_index'],class_id=pred['class_id'],score=pred['score'],box_xyxy=pred['box_xyxy'],status=status,error_category=cat,matched_GT_id=gs[mp[pi]]['GT_id'] if pi in mp else None,best_class_agnostic_IoU=bestv,best_class_agnostic_IoU_bin=ioubin(bestv),best_candidate_GT_id=gs[best]['GT_id'] if best is not None else None,best_candidate_class=gs[best]['class_id'] if best is not None else None,best_same_class_IoU=samev,best_other_class_IoU=otherv,**geometry(pred['box_xyxy'],hw)))
   imrows.append(dict(common,support=len(gs),prediction_count=len(ps),TP=len(mp),FP=len(ps)-len(mp),FN=len(gs)-len(mp),FP_plus_FN=len(ps)+len(gs)-2*len(mp)))
 write('all_GT_records.json',gtrows);write('all_prediction_records.json',prrows);write('all_image_records.json',imrows)
 tables={};totals={}
 for size in [640,960]:
  gg=[x for x in gtrows if x['imgsz']==size];pp=[x for x in prrows if x['imgsz']==size]
  totals[str(size)]={'support':len(gg),'TP':sum(x['status']=='TP' for x in gg),'FN':sum(x['status']=='FN' for x in gg),'FP':sum(x['status']=='FP' for x in pp),'FN_categories':dict(collections.Counter(x['error_category'] for x in gg if x['status']=='FN')),'FP_categories':dict(collections.Counter(x['error_category'] for x in pp if x['status']=='FP')),'matched_IoU_bins':dict(collections.Counter(ioubin(x['matched_IoU']) for x in gg if x['status']=='TP')),'matched_mean_IoU':float(np.mean([x['matched_IoU'] for x in gg if x['status']=='TP'])),'per_class':{}}
  for key in ['support','TP','FP','FN']:assert totals[str(size)][key]==summary['counts'][str(size)][key]
  for cl in [0,1]:
   cg=[x for x in gg if x['class_id']==cl];cp=[x for x in pp if x['class_id']==cl]
   totals[str(size)]['per_class'][str(cl)]={'support':len(cg),'TP':sum(x['status']=='TP' for x in cg),'FN':sum(x['status']=='FN' for x in cg),'FP':sum(x['status']=='FP' for x in cp),'FN_categories':dict(collections.Counter(x['error_category'] for x in cg if x['status']=='FN')),'FP_categories':dict(collections.Counter(x['error_category'] for x in cp if x['status']=='FP'))}
  tables[str(size)]={'GT_recall_by_GT_geometry_and_metadata':summarize(gg,True),'prediction_precision_by_prediction_geometry_and_metadata':summarize(pp,False)}
 write('stratified_tables.json',tables)
 train=[]
 for x in manifest['samples']:
  if x['partition']!='train':continue
  h,w=x['native_hw']
  for i,g in enumerate(x['instances']):
   cx,cy,bw,bh=g['xywh_normalized'];b=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
   train.append({'image_id':x['image_id'],'class_id':g['class_id'],'camera':x['capture_filename_tokens']['camera'],'date':x['capture_filename_tokens']['date'],'box_xyxy':b,**geometry(b,[h,w])})
 trdist={k:dict(collections.Counter(str(x[k]) for x in train)) for k in ['class_id','camera','date','area_bin','canonical640_size_bin','aspect_bin','edge_bin']};write('train_geometry_reference.json',{'image_count':596,'box_count':len(train),'distributions':trdist,'records':train,'no_train_predictions_used':True})
 transitions=[]
 for g in [x for x in gtrows if x['imgsz']==640]:
  z=next(x for x in gtrows if x['imgsz']==960 and x['GT_id']==g['GT_id']);transitions.append({'GT_id':g['GT_id'],'class_id':g['class_id'],'640':g['status'],'960':z['status'],'640_category':g['error_category'],'960_category':z['error_category'],'area_bin':g['area_bin'],'canonical640_size_bin':g['canonical640_size_bin'],'camera':g['camera'],'date':g['date']})
 write('paired_GT_transitions.json',transitions)
 report={'schema':'ssda_frozen_exploratory_error_analysis_v1','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_summary':str(SRC/'summary.json'),'source_summary_sha256':sha(SRC/'summary.json'),'weights_sha256':summary['weights_sha256'],'fixed_operating_point_totals':totals,'paired_GT_transition_counts':dict(collections.Counter(x['640']+'->'+x['960'] for x in transitions)),'train_geometry_reference':trdist,'development_support':{'images':113,'boxes':230,'class_counts':{'0':87,'1':143}},'input_file_hashes':inputs,'input_verification':{'prediction_files_verified':226,'image_sha_verified':113,'label_sha_verified':113,'GT_boxes_reconciled_against_source_manifest':230,'all_frozen_matches_and_counts_reproduced':True},'protocol':'analysis_protocol.json','all_samples_retained_in_outputs':True,'GT_or_prediction_labels_modified':False,'thresholds_tuned':False,'training_performed':False,'inference_performed':False,'GPU_used':False,'limitations':['Fixed-confidence counts are not AP and cannot reconstruct the full PR curve.','No predictions below 0.25 were retained; no-near-candidate is conditional on this exported operating point.','Numeric class disagreement is relative to author GT and is not evidence of a mislabeled biological category.','Unmatched predictions with no annotated GT are not automatically confirmed background errors; annotation completeness needs a separate review.','Camera/date comparisons are observational and confounded by acquisition/class/geometry.','All sizes use the same checkpoint trained at640; this is not a640-vs960 training experiment.','No verified physical fruit identities; date-heldout development is not independent orchard validation.','This analysis may motivate prospective ablations; it does not establish a new method or tune on dev.']}
 with (ROOT/'runs/ssda_yolo26n_detect_640_e40_s20261106/results.csv').open(encoding='utf-8') as f:last=list(csv.DictReader(f))[-1]
 report['training_log_reference']={'epoch':int(last['epoch']),'mAP50':float(last['metrics/mAP50(B)']),'mAP50_95':float(last['metrics/mAP50-95(B)']),'meaning':'Official training validation metrics; different protocol from fixed threshold counts above.'}
 write('error_analysis_summary.json',report)
 # Fixed deterministic example selection: up to 2/image-distinct cases for each class and error category at640, plus worst960 changes.
 chosen=[];seen=set()
 for kind,rr in [('FN',gtrows),('FP',prrows)]:
  for cl in [0,1]:
   for cat in sorted({x['error_category'] for x in rr if x['imgsz']==640 and x['class_id']==cl and x['status']==kind}):
    cand=sorted([x for x in rr if x['imgsz']==640 and x['class_id']==cl and x['status']==kind and x['error_category']==cat],key=lambda x:(x['image_id'],x.get('GT_index',x.get('prediction_index',0))))
    n=0
    for x in cand:
     key=(kind,cl,cat,x['image_id'])
     if key in seen:continue
     chosen.append({'kind':kind,'class_id':cl,'category':cat,'image_id':x['image_id'],'focus_box':x['box_xyxy'],'record_id':x.get('GT_id',str(x.get('prediction_index')))});seen.add(key);n+=1
     if n==2:break
 for delta in sorted(paired,key=lambda x:(x['delta']['risk_reduction_640_minus_960'],x['image_id']))[:4]:
  z=next(x for x in gtrows if x['image_id']==delta['image_id'] and x['imgsz']==640);chosen.append({'kind':'960_worse','class_id':None,'category':'risk_delta='+str(delta['delta']['risk_reduction_640_minus_960']),'image_id':delta['image_id'],'focus_box':z['box_xyxy'],'record_id':None})
 def panel(image_id,size,focus=None):
  info=gtmap[image_id];d=loaded[(image_id,size)]
  with Image.open(info['image_path']) as im:im=im.convert('RGB')
  if focus is None:crop=(0,0,im.width,im.height)
  else:
   x1,y1,x2,y2=focus;cx=(x1+x2)/2;cy=(y1+y2)/2;side=min(max(im.width,im.height),max(256,2*max(x2-x1,y2-y1)));left=max(0,min(im.width-side,cx-side/2));top=max(0,min(im.height-side,cy-side/2));crop=(left,top,left+side,top+side)
  im=im.crop(tuple(round(x) for x in crop)).resize((320,320),Image.Resampling.LANCZOS);dr=ImageDraw.Draw(im);left,top,right,bottom=crop;sx=320/(right-left);sy=320/(bottom-top)
  def box(b,color,text):
   xy=[(b[0]-left)*sx,(b[1]-top)*sy,(b[2]-left)*sx,(b[3]-top)*sy];dr.rectangle(xy,outline=color,width=2);tx=max(0,min(280,xy[0]));ty=max(0,min(304,xy[1]));dr.text((tx,ty),text,fill=color,stroke_width=1,stroke_fill='black')
  miss=set(d['box_matching']['missed_GT_indices']);fps=set(d['box_matching']['false_positive_prediction_indices'])
  for i,g in enumerate(d['ground_truth']):box(g['box_xyxy'],'magenta' if i in miss else 'lime',f"G{g['class_id']}#{i}")
  for p in d['predictions']:box(p['box_xyxy'],'orange' if p['prediction_index'] in fps else 'cyan',f"P{p['class_id']} {p['score']:.2f}")
  return im,crop
 pages=[]
 for page in range((len(chosen)+2)//3):
  cc=chosen[page*3:(page+1)*3];sheet=Image.new('RGB',(1280,380*len(cc)),'white');dr=ImageDraw.Draw(sheet)
  for ri,c in enumerate(cc):
   dr.text((5,ri*380+3),c['image_id']+' | '+c['kind']+' | '+str(c['class_id'])+' | '+c['category'],fill='black')
   for col,(size,focus) in enumerate([(640,None),(960,None),(640,c['focus_box']),(960,c['focus_box'])]):
    im,cr=panel(c['image_id'],size,focus);sheet.paste(im,(col*320,ri*380+55));dr.text((col*320+5,ri*380+21),str(size)+(' full' if focus is None else ' focus'),fill='black')
   dr.text((5,ri*380+36),'GT matched=green missed=magenta | prediction TP=cyan FP=orange',fill='black')
  name=f'failure_contact_sheet_{page+1:02d}.jpg';sheet.save(OUT/name,quality=91);pages.append({'path':name,'cases':cc})
 write('visual_review_index.json',{'debug_artifact_not_publication':True,'selection':'Up to2 lexicographically first image-distinct cases per640 class/errorcategory plus4 largest960 risk increases; no label modifications.','pages':pages})
 print(json.dumps({'totals':totals,'transitions':report['paired_GT_transition_counts'],'train_distributions':trdist,'debug_pages':len(pages),'output':str(OUT)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
