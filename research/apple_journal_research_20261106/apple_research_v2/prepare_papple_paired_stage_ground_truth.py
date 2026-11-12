"""Strict author stage / original visible-amodal GT identity and geometry audit.
CPU-only; source references and statistics, no image/mask dataset copies or training.
"""
from pathlib import Path
from collections import defaultdict,Counter
import datetime,hashlib,json,os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106')
BASE=ROOT.parent
SOURCE=ROOT/'apple_research_v2/sources/fuji_labels_audit'
OUT=ROOT/'apple_research_v2/datasets/papple_paired_visible_amodal_stage_v1'
ORIG=BASE/'apple_vision/local_amodal/data/papple/data'
CACHE=BASE/'apple_detection/cache/datasets/local_apple_v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def objsha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def vmap(data):return {r['filename']:(k,r) for k,r in data.get('_via_img_metadata',data).items()}
def regions(row):
 r=row['regions'];return list(r.items()) if isinstance(r,dict) else [(str(i),x) for i,x in enumerate(r)]
def shape_xy(shape):return np.asarray(list(zip(shape['all_points_x'],shape['all_points_y'])),dtype=np.float32)
def bbox(mask):
 xs=np.flatnonzero(mask.any(0));ys=np.flatnonzero(mask.any(1))
 return [int(xs[0]),int(ys[0]),int(xs[-1])+1,int(ys[-1])+1] if len(xs) else [0,0,0,0]
def raster(shape,w,h):
 xy=shape_xy(shape);reasons=[]
 if len(shape['all_points_x'])!=len(shape['all_points_y']) or len(xy)<3:reasons.append('fewer_than_three_points_or_mismatched_coordinate_arrays')
 if not np.isfinite(xy).all():reasons.append('nonfinite_coordinate')
 if len(xy) and np.isfinite(xy).all() and ((xy<0).any() or (xy[:,0]>w).any() or (xy[:,1]>h).any()):reasons.append('coordinates_outside_closed_image_edge_domain')
 canvas=Image.new('L',(w,h),0)
 if len(xy)>=3 and np.isfinite(xy).all():ImageDraw.Draw(canvas).polygon([tuple(map(float,p)) for p in xy],fill=1)
 arr=np.asarray(canvas,dtype=np.uint8).copy()
 if not arr.any():reasons.append('empty_native_raster')
 return arr,reasons,{'point_count':len(xy),'explicit_closed':bool(len(xy)>0 and np.array_equal(xy[0],xy[-1])),'edge_coordinate_equal_width_or_height':bool(len(xy)>0 and ((xy[:,0]==w).any() or (xy[:,1]==h).any()))}
def dist(values):
 x=np.asarray(values,dtype=float)
 if not len(x):return {'count':0}
 return {'count':len(x),'min':float(x.min()),'p25':float(np.quantile(x,.25)),'median':float(np.quantile(x,.5)),'p75':float(np.quantile(x,.75)),'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99)),'max':float(x.max()),'mean':float(x.mean())}
def main():
 OUT.mkdir(parents=True,exist_ok=False);bound={}
 def bind(p,expected=None):
  p=Path(p).resolve();assert p.drive.upper()=='F:';h=sha(p)
  if expected is not None:assert h==expected,str(p)
  bound[str(p)]=h;return h
 candidate_path=SOURCE/'papple_maturity_candidate_manifest.json';candidate=read(candidate_path);bind(candidate_path)
 imagepath=SOURCE/'aligned_image_manifest.json';image_rows=read(imagepath);bind(imagepath)
 current_image={(r['author_split'],r['filename']):r for r in image_rows}
 bind(Path(__file__))
 for n in ['linkage_details.json','component_audit_v2.json','label_consistency_audit.json','identifier_scope_reconciliation.json']:bind(SOURCE/n)
 author={};old={};local={};original_inventory=[];lookup={};geometry={};samples_out=[];zero_counts=Counter()
 for split in ['train','val']:
  ap=SOURCE/(split+'_via_final.json');bind(ap,candidate['source_files_sha256'][split]);author[split]=(ap,vmap(read(ap)))
  lp=CACHE/(split+'_manifest.json');ld=read(lp);bind(lp);local[split]=ld
  old[split]={}
  for br,name in [('visible','via_region_data_instance.json'),('amodal','via_region_data_amodal.json')]:
   path=ORIG/'gt_json'/split/name;bind(path,ld['source']['annotation_sha256'][br]);old[split][br]=(path,vmap(read(path)))
  for frame_index,sample in enumerate(ld['samples']):
   name=sample['image_id'];h,w=sample['original_hw'];assert(h,w)==(1300,1300)
   align=current_image[(split,name)];img=ORIG/'images'/split/name;bind(img,align['sha256']);assert sample['rgb_sha256']==align['sha256']
   cp=CACHE/'masks'/split/(Path(name).stem+'.npz');bind(cp,sample['mask_sha256'])
   vr=old[split]['visible'][1][name];ar=old[split]['amodal'][1][name]
   vregs=regions(vr[1]);aregs=regions(ar[1]);by_v={};by_a={};by_geom=defaultdict(list)
   for i,(key,region) in enumerate(vregs):
    identity=str(region['region_attributes']['apple_ID']).strip();assert identity not in by_v
    by_v[identity]=(i,key,region);by_geom[objsha(region['shape_attributes'])].append(identity)
   for i,(key,region) in enumerate(aregs):
    identity=str(region['region_attributes']['apple_ID']).strip();assert identity not in by_a
    by_a[identity]=(i,key,region)
   assert set(by_v)==set(by_a)=={r['instance_id'] for r in sample['instances']}
   geometry[(split,name)]=by_geom
   if not by_v:zero_counts[split]+=1
   samples_out.append({'split':split,'image_id':name,'image_path':str(img),'image_sha256':align['sha256'],'image_wh':[w,h],'original_GT_instance_count':len(by_v),'author_stage_region_count':len(regions(author[split][1][name][1])),'source_frame_proxy':align['source_frame_proxy'],'original_GT_mask_cache_path':str(cp),'original_GT_mask_cache_sha256':bound[str(cp.resolve())]})
   with np.load(cp,allow_pickle=False) as z:
    assert set(z.files)=={'visible_packed','amodal_packed','original_hw','instance_ids','boxes_visible_xyxy','boxes_amodal_xyxy','labels'}
    assert z['original_hw'].tolist()==[h,w]
    ids=z['instance_ids'].tolist();assert ids==[r['instance_id'] for r in sample['instances']]
    vdecoded=np.unpackbits(z['visible_packed'],axis=-1,count=w,bitorder='little');adecoded=np.unpackbits(z['amodal_packed'],axis=-1,count=w,bitorder='little')
    assert vdecoded.shape==adecoded.shape==(len(ids),h,w)
    for j,meta in enumerate(sample['instances']):
     identity=meta['instance_id'];vi,vkey,vreg=by_v[identity];ai,akey,areg=by_a[identity]
     v,vbad,vshape=raster(vreg['shape_attributes'],w,h);a,abad,ashape=raster(areg['shape_attributes'],w,h)
     assert np.array_equal(v,vdecoded[j]) and np.array_equal(a,adecoded[j]),(split,name,identity,'polygon/cache mismatch')
     va=int(v.sum());aa=int(a.sum());outside=int(np.count_nonzero(v & (1-a)));inside=va-outside;hidden=aa-inside;vb=bbox(v);ab=bbox(a)
     assert va==meta['visible_area_pixels'] and aa==meta['amodal_area_pixels'] and outside==meta['visible_outside_amodal_pixels']
     assert vb==meta['visible_topology']['bbox_xyxy'] and ab==meta['amodal_topology']['bbox_xyxy']
     assert np.array_equal(vb,z['boxes_visible_xyxy'][j]) and np.array_equal(ab,z['boxes_amodal_xyxy'][j])
     ratio=va/aa if aa else None
     assert ratio==meta['visibility_V_area_over_A_area']
     capture=name.split('_')[2];year='2018' if capture[0] in '23' else '2020' if capture[0] in '67' else None
     scoped=f'{year}_01_{int(identity):03d}' if year else None;assert scoped==meta['apple_identity']
     record={'original_instance_key':f'{split}/{name}/apple_ID={identity}','split':split,'image_id':name,'original_apple_ID':identity,'derived_scoped_ID':scoped,'derived_capture_group':meta['capture_group'],'source_frame_proxy':align['source_frame_proxy'],
      'visible_source':{'path':str(old[split]['visible'][0]),'file_sha256':bound[str(old[split]['visible'][0].resolve())],'via_key':vr[0],'region_key':vkey,'region_index':vi,'shape_sha256':objsha(vreg['shape_attributes']),'region_sha256':objsha(vreg),'raster_uint8_sha256':hashlib.sha256(v.tobytes()).hexdigest(),'shape_audit':vshape},
      'amodal_source':{'path':str(old[split]['amodal'][0]),'file_sha256':bound[str(old[split]['amodal'][0].resolve())],'via_key':ar[0],'region_key':akey,'region_index':ai,'shape_sha256':objsha(areg['shape_attributes']),'region_sha256':objsha(areg),'raster_uint8_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'shape_audit':ashape},
      'GT_cache':{'path':str(cp),'file_sha256':bound[str(cp.resolve())],'instance_index':j,'instance_ID_field_value':identity,'packed_bitorder':'little','packed_axis':-1,'count_width':w,'visible_array_key':'visible_packed','amodal_array_key':'amodal_packed','verified_equal_to_original_VIA_raster':True},
      'geometry_status':'invalid_geometry' if vbad or abad else 'valid_source_geometry','invalid_geometry_reasons':{'visible':vbad,'amodal':abad},
      'geometry':{'visible_area_pixels':va,'amodal_area_pixels':aa,'visible_inside_amodal_pixels':inside,'visible_outside_amodal_pixels':outside,'amodal_minus_visible_pixels':hidden,'visible_area_over_amodal_area':ratio,'visible_intersection_amodal_over_amodal_area':inside/aa if aa else None,'visible_outside_amodal_fraction_of_visible':outside/va if va else None,'visible_outside_amodal_fraction_of_amodal':outside/aa if aa else None,'visible_is_subset_of_amodal':outside==0,'visible_area_exceeds_amodal':va>aa,'visible_minus_amodal_area_pixels':va-aa,'visible_bbox_xyxy_halfopen':vb,'amodal_bbox_xyxy_halfopen':ab,'bbox_center_delta_amodal_minus_visible_xy':[(ab[0]+ab[2]-vb[0]-vb[2])/2,(ab[1]+ab[3]-vb[1]-vb[3])/2],'V_not_clipped_to_A':True}}
     original_inventory.append(record);lookup[(split,name,identity)]=record
   if (frame_index+1)%100==0:print(f'ORIGINAL {split} {frame_index+1}/{len(ld["samples"])} raster/cache/identity verified',flush=True)
 records=[];linked_original=set();unknown=[];stats=defaultdict(Counter);scoped_by_split=defaultdict(set);scoped_labels=defaultdict(set)
 for source_index,row in enumerate(candidate['records']):
  split=row['split'];name=row['source_image_id'];ap,ad=author[split];vk,source=ad[name];reg_key,reg=regions(source)[row['source_region_index']]
  # Index locates the declared author record only; no old visible/amodal alignment is inferred from order.
  shape=reg['shape_attributes'];hsh=objsha(shape);label=reg['region_attributes']['Ripeness'];assert hsh==row['source_shape_sha256'] and vk==row['source_via_key']
  assert label==row['label_string_as_provided'] and candidate['class_mapping'][label]==row['label']
  matches=geometry[(split,name)].get(hsh,[]);identity=matches[0] if len(matches)==1 else None
  assert bool(identity)==row['exact_original_modal_geometry'];assert identity==row['matched_original_apple_ID']
  base={'record_index':source_index,'split':split,'split_role':'train' if split=='train' else 'development','image_id':name,'image_path':str(ORIG/'images'/split/name),'image_sha256':row['image_sha256'],'image_wh':row['image_wh'],'label':row['label'],'label_string_as_provided':label,'source_frame_proxy':current_image[(split,name)]['source_frame_proxy'],'author_visible_source':{'path':str(ap),'file_sha256':bound[str(ap.resolve())],'via_key':vk,'region_key':reg_key,'region_index':row['source_region_index'],'shape_sha256':hsh,'region_sha256':objsha(reg)},'author_visible_polygon_bbox_extrema':row['bbox_xyxy'],'diameter_as_provided_not_enabled_for_regression':row['diameter_as_provided_not_regression_target'],'exact_original_visible_geometry_match_count':len(matches),'original_instance_key':None,'original_apple_ID':None,'derived_scoped_ID':None,'original_visible_source':None,'amodal_supervision':None,'geometry':None}
  if identity is None:
   status='unknown_no_exact_visible_geometry_match' if not matches else 'unknown_nonunique_visible_geometry_match';base['pairing_status']=status;unknown.append(base['record_index']);assert not row['exact_original_modal_geometry']
  else:
   original=lookup[(split,name,identity)];assert original['visible_source']['region_index']==row['matched_original_modal_region_index']
   base.update(pairing_status='valid_identity_and_geometry_pair' if original['geometry_status']=='valid_source_geometry' else 'invalid_source_geometry_pair',original_instance_key=original['original_instance_key'],original_apple_ID=identity,derived_scoped_ID=original['derived_scoped_ID'],original_visible_source=original['visible_source'],geometry=original['geometry'],invalid_geometry_reasons=original['invalid_geometry_reasons'])
   if original['geometry_status']=='valid_source_geometry':base['amodal_supervision']={'annotation_kind':'original_author_full_amodal_ground_truth_not_model_prediction','source':original['amodal_source'],'GT_cache':original['GT_cache']}
   linked_original.add(original['original_instance_key']);scoped_by_split[split].add(original['derived_scoped_ID']);scoped_labels[original['derived_scoped_ID']].add(label)
  stats[split][base['pairing_status']]+=1;stats[split]['label_'+label]+=1;records.append(base)
 assert len(records)==12014 and len(unknown)==18
 missing_path=ROOT/'apple_research_v2/datasets/papple_visible_stage_seg_v1/ignore_candidates_provenance.json';missing=read(missing_path);bind(missing_path);missing_out=[]
 for row in missing['candidates']:
  original=lookup[(row['split'],row['filename'],str(row['original_apple_ID']))]
  assert original['visible_source']['region_index']==row['original_region_index'] and original['visible_source']['region_sha256']==objsha(row['original_region'])
  assert original['original_instance_key'] not in linked_original
  missing_out.append({'original_instance_key':original['original_instance_key'],'original_visible_source':original['visible_source'],'original_amodal_source':original['amodal_source'],'stage_label':None,'stage_supervision_enabled':False,'reason':'Explicit previously audited author-stage omission; no label invented','geometry':original['geometry']})
 assert len(missing_out)==14
 scope_audit={}
 for which,rs in [('all_original',original_inventory),('strict_author_pairs',[lookup[(r['split'],r['image_id'],r['original_apple_ID'])] for r in records if r['original_apple_ID'] is not None])]:
  sets={split:{r['derived_scoped_ID'] for r in rs if r['split']==split} for split in ['train','val']};rawsets={split:{r['original_apple_ID'] for r in rs if r['split']==split} for split in ['train','val']}
  scope_audit[which]={'derived_unique_train':len(sets['train']),'derived_unique_val':len(sets['val']),'derived_shared_count':len(sets['train']&sets['val']),'derived_shared_IDs':sorted(sets['train']&sets['val']),'raw_unique_train':len(rawsets['train']),'raw_unique_val':len(rawsets['val']),'raw_shared_count':len(rawsets['train']&rawsets['val'])}
 scope_audit['strict_author_pairs']['derived_IDs_with_both_stage_labels']=sorted(k for k,v in scoped_labels.items() if len(v)>1)
 scope_audit['strict_author_pairs']['derived_IDs_with_both_stage_labels_count']=sum(len(v)>1 for v in scoped_labels.values())
 magnitude={};exceeds=[]
 for which,rs in [('all_original',original_inventory),('strict_author_pairs',[lookup[(r['split'],r['image_id'],r['original_apple_ID'])] for r in records if r['original_apple_ID'] is not None])]:
  magnitude[which]={}
  for split in ['train','val','combined']:
   selected=[r for r in rs if split=='combined' or r['split']==split];g=[r['geometry'] for r in selected];outside=[x for x in g if x['visible_outside_amodal_pixels']>0]
   summary={'instances':len(g),'V_not_subset_A_count':len(outside),'V_area_exceeds_A_count':sum(x['visible_area_exceeds_amodal'] for x in g),'containment_exact_count':sum(x['visible_is_subset_of_amodal'] for x in g),'all_instances':{},'outside_positive_instances':{},'outside_fraction_of_V_bins':{}}
   for name in ['visible_outside_amodal_pixels','visible_outside_amodal_fraction_of_visible','visible_outside_amodal_fraction_of_amodal','visible_area_over_amodal_area']:
    summary['all_instances'][name]=dist([x[name] for x in g]);summary['outside_positive_instances'][name]=dist([x[name] for x in outside])
   for lo,hi,label in [(0,.001,'0_to_0.1percent'),(.001,.01,'0.1_to_1percent'),(.01,.05,'1_to_5percent'),(.05,.1,'5_to_10percent'),(.1,1.0000001,'10_to_100percent')]:summary['outside_fraction_of_V_bins'][label]=sum(lo<x['visible_outside_amodal_fraction_of_visible']<=hi for x in g)
   magnitude[which][split]=summary
 for r in original_inventory:
  if r['geometry']['visible_area_exceeds_amodal']:exceeds.append(r)
 unmatched_original=[{'original_instance_key':r['original_instance_key'],'split':r['split'],'original_apple_ID':r['original_apple_ID'],'derived_scoped_ID':r['derived_scoped_ID'],'visible_source':r['visible_source'],'amodal_source':r['amodal_source'],'reason':'No exact author-visible shape match; no order or bbox-based assignment'} for r in original_inventory if r['original_instance_key'] not in linked_original]
 contract={'schema':'papple_paired_visible_amodal_author_stage_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'audited_supervision_manifest_only; not loaded into training','records':records,'class_names':candidate['class_names'],'class_mapping':candidate['class_mapping'],'label_semantics':candidate['label_semantics'],'pairing_rule':'Exact canonical original shape_attributes hash within same split/image, unique original visible region; then same explicit original apple_ID within same image links original amodal region. Never index/order/bbox-IoU association.','geometry_rule':'Original Pillow ImageDraw rasterization float32 polygons, native1300, checked against hash-bound original GT packed masks. No clipping V to A, no repair, convex hull, mask synthesis, or model prediction.','ratio_rule':'Raw |V|/|A| may exceed1; separate intersection-over-A recorded without replacing raw ratio.','invalid_pair_rule':'Invalid geometry retains source references but amodal_supervision null; V outside A is a separately flagged annotation inconsistency, not automatic removal.','unknown_pair_rule':'18 changed author-visible geometries retain original stage labels and source shape; original identity and amodal supervision null.','mask_storage':'No full mask/image duplication; explicit source file hashes, region IDs and native GT cache references.','split_policy':{'independent_physical_fruit_holdout':False,'train':'original author train','val':'development only','scoped_ID_namespace':'derived local year_01_rawID, original physical namespace not fully verified','crosssplit_identity_audit':scope_audit,'source_test_data_read':False},'counts':{k:dict(v) for k,v in stats.items()},'source_license_for_author_stage':candidate['source_license'],'inferred_quality_grades':False,'training_or_GPU_used':False}
 write(OUT/'paired_stage_manifest.json',contract);write(OUT/'original_GT_inventory.json',{'records':original_inventory,'count':len(original_inventory)});write(OUT/'image_manifest.json',samples_out);write(OUT/'unmatched_original_geometry.json',{'count':len(unmatched_original),'records':unmatched_original});write(OUT/'missing_author_stage_val14.json',{'count':14,'records':missing_out,'not_automatically_ignored_in_existing_YOLO_training':True});write(OUT/'V_area_exceeds_A_cases.json',{'count':len(exceeds),'records':exceeds});write(OUT/'containment_and_ratio_audit.json',magnitude);write(OUT/'scoped_identity_audit.json',scope_audit)
 for filename,h in bound.items():assert sha(filename)==h,'Source changed during audit '+filename
 write(OUT/'source_sha256.json',bound)
 report={'schema':'papple_paired_stage_GT_audit_v1','phase':'complete','original_images':len(samples_out),'original_instances':len(original_inventory),'author_stage_records':len(records),'valid_pairs':sum(r['pairing_status']=='valid_identity_and_geometry_pair' for r in records),'invalid_pairs':sum(r['pairing_status']=='invalid_source_geometry_pair' for r in records),'unknown_pairs':len(unknown),'unknown_by_split':dict(Counter(records[i]['split'] for i in unknown)),'explicit_missing_author_val_instances':14,'unmatched_original_geometry_count':len(unmatched_original),'empty_images':dict(zero_counts),'original_GT_raster_cache_pairs_exactly_verified':len(original_inventory),'original_GT_masks_raster_SHA_recorded':2*len(original_inventory),'V_area_exceeds_A_count':len(exceeds),'source_files_before_after_SHA_verified':len(bound),'source_unchanged':True,'GPU_or_training':False,'manifest_sha256':sha(OUT/'paired_stage_manifest.json'),'script_sha256':sha(__file__),'counts':{k:dict(v) for k,v in stats.items()},'scoped_identity_audit':scope_audit,'containment_and_ratio_audit':magnitude,'full_masks_duplicated':False,'semantic_primary_reference_pending_separate_evidence':True}
 write(OUT/'audit_summary.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['scoped_identity_audit','containment_and_ratio_audit']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
