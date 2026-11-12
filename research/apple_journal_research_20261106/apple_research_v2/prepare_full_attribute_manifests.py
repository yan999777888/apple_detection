"""Build new combined train/development manifests from frozen prepared real labels.
No test reads, no coordinate repairs, no new class interpretation.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def source_receipt(p):
 preserved=ROOT/'migration_original_text'/p.relative_to(ROOT)
 return {'operational_path':str(p),'operational_sha256':sha(p),'original_pre_relocation_preserved_path':str(preserved) if preserved.exists() else None,'original_pre_relocation_bytes_sha256':sha(preserved) if preserved.exists() else sha(p),'note':'Operational manifest hashes differ from original bytes when absolute paths were relocated; image/annotation bytes retain their original hashes.'}
def save(p,d):
 if p.exists(): raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 print(p.name,sha(p),Counter((r['split'],r['label']) for r in d['records']))
def main():
 assert (ROOT/'relocation_complete.json').is_file()
 output=ROOT/'apple_research_v2/prepared/full_attribute_v3'
 mroot=ROOT/'apple_research_v2/prepared/mendeley_v4_original_documented_v2'
 paths=[mroot/'train_manifest.json',mroot/'val_manifest.json']; docs=[read(p) for p in paths]; rows=[]
 assert docs[0]['class_names']==docs[1]['class_names']
 for doc,source_split in zip(docs,['train','val']):
  annotation=mroot/'annotations'/f'{source_split}.json'; ahash=sha(annotation)
  for original in doc['samples']:
   assert original['annotation_sha256']==ahash
   row=dict(original,annotation_path=str(annotation),split='train' if source_split=='train' else 'dev',source_published_split=source_split)
   rows.append(row)
 m={'schema':'attribute_full_manifest_v3','dataset':'Mendeley gfcmdbvw65 v4 visible appearance stages','class_names':docs[0]['class_names'],'label_semantics':docs[0]['label_semantics'],'source_manifests':[source_receipt(p) for p in paths],'source_archive_sha256':docs[0]['source_archive_sha256'],'split_policy':dict(docs[0]['split_policy'],evaluation_role='development_non_independent',independent_group_holdout=False,independent_fruit_or_scene_verified=False,development_source='published val after documented train/val family and exact-hash exclusion; no physiological maturity labels'),'records':rows}
 assert len(rows)==1646 and sum(r['split']=='train' for r in rows)==1532
 save(output/'mendeley_train_dev.json',m)
 sroot=ROOT/'apple_research_v2/datasets/ssda_date_v2'; source=sroot/'manifest.json'; train_source=sroot/'attribute_train_manifest.json'; source_doc=read(source); train_doc=read(train_source)
 lookup={r['image_id']:r for r in source_doc['samples']}; rows=[]
 for original in train_doc['records']:
  image=lookup[original['source_image_id']]; assert image['partition']=='train' and image['training_eligible'] and not any(b['source_out_of_bounds'] for b in image['instances'])
  assert original['annotation_sha256']==image['label_sha256']
  rows.append(dict(original,annotation_path=image['label_path'],source_published_split=image['original_split']))
 for image in source_doc['samples']:
  if image['partition']!='dev': continue
  assert image['training_eligible'] and not any(b['source_out_of_bounds'] for b in image['instances'])
  height,width=image['native_hw']; tokens=image['capture_filename_tokens']
  for index,box in enumerate(image['instances']):
   cx,cy,w,h=box['xywh_normalized']
   rows.append({'image_path':image['rgb_path'],'annotation_path':image['label_path'],'bbox_xyxy':[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height],'label':box['class_id'],'source_image_id':image['image_id'],'source_instance_index':index,'group_id':tokens['date']+'_'+tokens['time'],'group_kind':'filename_same_second_proxy','split':'dev','source_published_split':image['original_split'],'image_sha256':image['rgb_sha256'],'annotation_sha256':image['label_sha256']})
 assert len(rows)==1506 and sum(r['split']=='train' for r in rows)==1276
 d={'schema':'attribute_full_manifest_v3','dataset':'SSDA numeric annotated defect-region categories','class_names':train_doc['class_names'],'label_semantics':train_doc['label_semantics'],'source_manifests':[source_receipt(source),source_receipt(train_source)],'split_policy':dict(train_doc['split_policy'],evaluation_role='development_non_independent',independent_group_holdout=False,validation_records_included=True,development_dates=['20211025','20211027'],group_definition='date+second filename proxy, with dates separated; physical fruit identities unavailable',development_source='full documented date holdout; source multicamera same-second groups kept together; no healthy or whole-fruit grade class'),'records':rows}
 save(output/'ssda_train_dev.json',d)
if __name__=='__main__': main()