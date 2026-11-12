import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import json,collections,hashlib,datetime
R=Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106')
P=R/'experiments/error_analysis/ssda_frozen_v1';S=R/'experiments/resolution_pairs/ssda_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def wr(n,x):
 p=P/n
 assert not p.exists()
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def inter(a,b):return max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
def area(b):return (b[2]-b[0])*(b[3]-b[1])
gg=read(P/'all_GT_records.json');pairs=read(S/'paired_images.json');ps={}
for row in pairs:
 for res in [640,960]:ps[(row['image_id'],res)]=read(S/row['resolutions'][str(res)]['prediction_file'])['predictions']
rows=[]
for g in gg:
 if g['error_category']!='no_near_retained_candidate_IoU_below_.1':continue
 b=g['box_xyxy'];cand=[]
 for p in ps[(g['image_id'],g['imgsz'])]:
  a=p['box_xyxy'];ia=inter(a,b);cx=(a[0]+a[2])/2;cy=(a[1]+a[3])/2
  cand.append({'prediction_index':p['prediction_index'],'class_id':p['class_id'],'score':p['score'],'IoU':ia/(area(a)+area(b)-ia),'prediction_center_inside_GT':b[0]<=cx<=b[2] and b[1]<=cy<=b[3],'intersection_over_prediction_area':ia/area(a),'intersection_over_GT_area':ia/area(b),'prediction_to_GT_area_ratio':area(a)/area(b)})
 rows.append({'image_id':g['image_id'],'imgsz':g['imgsz'],'GT_id':g['GT_id'],'class_id':g['class_id'],'canonical640_size_bin':g['canonical640_size_bin'],'candidates':cand,'any_prediction_center_inside_GT':any(p['prediction_center_inside_GT'] for p in cand),'any_same_class_prediction_center_inside_GT':any(p['prediction_center_inside_GT'] and p['class_id']==g['class_id'] for p in cand)})
sumry={}
for res in [640,960]:
 rr=[r for r in rows if r['imgsz']==res]
 sumry[str(res)]={'IoU_below_point1_FN_count':len(rr),'with_any_prediction_center_inside_GT':sum(r['any_prediction_center_inside_GT'] for r in rr),'with_same_class_prediction_center_inside_GT':sum(r['any_same_class_prediction_center_inside_GT'] for r in rr),'by_size':{b:{'count':sum(r['canonical640_size_bin']==b for r in rr),'center_inside':sum(r['canonical640_size_bin']==b and r['any_prediction_center_inside_GT'] for r in rr),'same_class_center_inside':sum(r['canonical640_size_bin']==b and r['any_same_class_prediction_center_inside_GT'] for r in rr)} for b in ['16-32px','32-96px','>=96px']}}
tr=read(P/'train_geometry_reference.json')['records']
class_size=[]
for partition,rr in [('train',tr),('dev',[g for g in gg if g['imgsz']==640])]:
 for c in [0,1]:
  for b in ['<16px','16-32px','32-96px','>=96px']:
   class_size.append({'partition':partition,'class_id':c,'size_bin':b,'count':sum(r['class_id']==c and r['canonical640_size_bin']==b for r in rr)})
wr('IoU_low_containment_supplement.json',{'role':'Supplementary exploratory geometric diagnostic, no matching/status changes.','reason':'IoU<.1 does not imply no local response: a small prediction may lie inside a much larger GT region.','no_tuned_thresholds':'Continuous intersection/area ratios and exact point-inside-box predicate; all candidates preserved.','summary':sumry,'records':rows,'class_size_support_train_dev':class_size})
print(json.dumps(sumry))
