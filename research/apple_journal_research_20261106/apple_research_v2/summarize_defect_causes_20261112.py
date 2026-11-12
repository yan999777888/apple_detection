"""CPU-only supplemental label, scale, score and evidence audit. No GT edits."""
from pathlib import Path
import json,collections
import numpy as np
from audit_defect_causes_20261112 import ious,match,nms_indices
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/defect_cause_audit_20261112_v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
samples=read(ROOT/'apple_research_v2/datasets/ssda_date_v2/manifest.json')['samples'];gg=read(OUT/'GT_records.json')
invalid=[];duplicate=[];empty=[];sources=collections.defaultdict(set)
for s in samples:
    rows=s['instances'];sources[s['rgb_sha256']].add(s['partition'])
    if not rows:empty.append(s['image_id'])
    seen=set()
    for row in rows:
        c=row['class_id'];x,y,w,h=row['xywh_normalized']
        if not (c in (0,1) and w>0 and h>0 and min(x-w/2,y-h/2)>=0 and max(x+w/2,y+h/2)<=1):invalid.append({'image':s['image_id'],'partition':s['partition'],'line':row['source_line']})
        key=(c,x,y,w,h)
        if key in seen:duplicate.append({'image':s['image_id'],'line':row['source_line']})
        seen.add(key)
score_summary={};scale={};low_nms={}
for part in ('train','dev'):
    rows=[r for r in gg if r['partition']==part];fn=[r for r in rows if r['category']=='low_score_correct_box']
    score_summary[part]={'low_score_FN_count':len(fn),'score_quantiles_0_25_50_75_100':np.quantile([r['max_score_correct_box'] for r in fn],[0,.25,.5,.75,1]).tolist(),'by_class':dict(collections.Counter(r['class'] for r in fn))}
    scale[part]={str(cls):{size:sum(r['class']==cls and r['size_bin']==size for r in rows) for size in ('<16','16-32','32-96','>=96')} for cls in (0,1)}
    counts=collections.Counter()
    for s in (x for x in samples if x['partition']==part):
        rec=read(OUT/'predictions'/f"{Path(s['image_id']).stem}.json");ps=rec['low_predictions'];kept=[ps[i] for i in nms_indices(ps)];mp,_=match(kept,rec['GT'])
        counts.update(TP=len(mp),FP=len(kept)-len(mp),FN=len(rec['GT'])-len(mp))
    low_nms[part]=dict(counts);low_nms[part].update(P=counts['TP']/(counts['TP']+counts['FP']),R=counts['TP']/(counts['TP']+counts['FN']))
review=read(OUT/'visual_review/review_index.json');ambiguous={'FP019','FP033','FP036','FP039','FP042','FP046','FP047'}
notes=[]
for r in review['cases']:
    rid=r['review_id'];cat=r['category']
    if rid.startswith('FP') and cat=='no_near_annotated_GT':
        finding='Diffuse color/texture/shadow or ambiguous feature; cannot resolve normal vs target defect from RGB alone.' if rid in ambiguous else 'Visible localized dark/gray mark or scratch at unmatched prediction; annotation completeness/definition needs adjudication, not confirmed missing GT.'
    elif rid.startswith('FP') and cat=='localization_IoU_0.1_to_0.5':finding='Predicted extent differs from annotation, including narrow core vs broader halo, shifted box or merging adjacent defects; does not by itself prove bad GT.'
    elif cat=='wrong_class_high_overlap':finding='Correct general location but numeric class disagrees; visual appearance alone cannot establish causal biological defect class.'
    elif cat=='duplicate_same_class':finding='Overlapping predictions around one annotated region; confirmed redundant detection at this operating point.'
    elif cat=='low_score_correct_box':finding='Same-class IoU>=.5 candidate exists below .25. Visible appearance ranges from clear cuts/dark spots to diffuse/edge regions. Score/ranking failure; no automatic relabel.'
    elif rid in {'FN042','FN043','FN044','FN045','FN046'}:finding='Broad low-contrast/discolored/near-stem region: extent and visual criterion uncertain; retain original GT for evaluation.'
    elif rid=='FN047':finding='Visible elongated wound/scratch near fruit edge; model fails to produce a correct box even at .001 floor.'
    elif rid=='FN048':finding='Small visible dark spot among similar surface marks; no correct box at .001 floor.'
    else:finding='Visible target extent mismatch; high score response covers only part or merges multiple annotated regions.'
    notes.append({'review_id':rid,'image_id':r['image_id'],'sheet':r['sheet'],'finding':finding,'reviewer':'assistant visual inspection; not independent human expert','GT_changed':False})
report={'label_integrity':{'source_images':len(samples),'included_images':709,'included_GT':1506,'source_invalid_boxes':invalid,'exact_duplicate_annotation_rows':duplicate,'empty_label_images':empty,'cross_train_dev_identical_image_hashes':sum('train' in v and 'dev' in v for v in sources.values()),'known_invalid_image_already_quarantined':True,'all_original_file_hashes_verified_by_inference':True},'low_score_FN':score_summary,'scale_by_class':scale,'diagnostic_only_floor001_NMS07':low_nms,'visual_review':{'coverage':'all 50 post-NMS FP and 51 FN across 17 sheets; train 24 purposefully selected cases across 4 sheets','no_near_annotated_GT_cases':36,'localized_visible_feature_cases':29,'diffuse_or_ambiguous_cases':7,'these_are_visual_descriptions_not_missing_label_counts':True,'confirmed_corrected_labels':0,'notes':notes},'semantic_mapping':{'numeric_labels_unchanged':True,'paper_table_inference':'Original counts class0=500 train/124 val and class1=716 train/172 val exactly match the paper Scratch/Pest columns; supports inferred 0=scratch,1=pest, but no explicit release class-ID dictionary was found.','paper':'https://assets-eu.researchsquare.com/files/rs-4205600/v1_covered_5ca1cd7d-2085-42f9-b29d-e1830ae4d18b.pdf','mapping_author_confirmed':False}}
(OUT/'supplemental_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='visual_review'},ensure_ascii=False,indent=2))
