"""Measure all original GT V-outside-A pixel adjacency without modifying masks."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,cv2
cv2.setNumThreads(1)
R=Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106');O=R/'apple_research_v2/datasets/papple_paired_visible_amodal_stage_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=json.loads((O/'original_GT_inventory.json').read_text(encoding='utf-8'))['records'];groups=defaultdict(list)
for row in rows:groups[row['GT_cache']['path']].append(row)
results=[];counts=Counter();max_distance=0
for gi,(filename,instances) in enumerate(groups.items()):
 assert sha(filename)==instances[0]['GT_cache']['file_sha256']
 with np.load(filename,allow_pickle=False) as z:
  vp=z['visible_packed'];ap=z['amodal_packed'];H,W=map(int,z['original_hw'])
  for row in instances:
   expected=row['geometry']['visible_outside_amodal_pixels'];base={'original_instance_key':row['original_instance_key'],'split':row['split'],'outside_pixels':expected,'distance_exactly_1_pixels':0,'distance_sqrt2_pixels':0,'distance_beyond_sqrt2_pixels':0,'max_outside_distance_pixels':0.}
   if not expected:counts['zero_outside_instances']+=1;results.append(base);continue
   j=row['GT_cache']['instance_index'];l,t,r,b=row['geometry']['visible_bbox_xyxy_halfopen'];l=max(0,l-1);t=max(0,t-1);r=min(W,r+1);b=min(H,b+1);bitl=l//8;bitr=(r+7)//8
   v=np.unpackbits(vp[j,t:b,bitl:bitr],axis=-1,bitorder='little')[:,l-bitl*8:r-bitl*8]
   a=np.unpackbits(ap[j,t:b,bitl:bitr],axis=-1,bitorder='little')[:,l-bitl*8:r-bitl*8]
   outside=v&(1-a);assert int(outside.sum())==expected
   pad=np.pad(a,1);near4=pad[:-2,1:-1]|pad[2:,1:-1]|pad[1:-1,:-2]|pad[1:-1,2:];diag=pad[:-2,:-2]|pad[:-2,2:]|pad[2:,:-2]|pad[2:,2:]
   n1=int((outside&near4).sum());n2=int((outside&(1-near4)&diag).sum());nfar=expected-n1-n2;maxd=1. if not n2 else 2**.5
   if nfar:
    # Rare fallback computes exact native-frame pixel-center Euclidean distance.
    fullv=np.unpackbits(vp[j],axis=-1,count=W,bitorder='little');fulla=np.unpackbits(ap[j],axis=-1,count=W,bitorder='little');dist=cv2.distanceTransform((1-fulla).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);maxd=float(dist[(fullv&(1-fulla)).astype(bool)].max())
   base.update(distance_exactly_1_pixels=n1,distance_sqrt2_pixels=n2,distance_beyond_sqrt2_pixels=nfar,max_outside_distance_pixels=maxd);results.append(base);max_distance=max(max_distance,maxd)
   counts['outside_positive_instances']+=1;counts['outside_pixels']+=expected;counts['distance_exactly_1_pixels']+=n1;counts['distance_sqrt2_pixels']+=n2;counts['distance_beyond_sqrt2_pixels']+=nfar;counts['instances_beyond_one_pixel']+=int(n2+nfar>0)
 if (gi+1)%500==0:print(f'BOUNDARY {gi+1}/{len(groups)} frames',flush=True)
result={'schema':'original_GT_outside_pixel_distance_audit_v1','source_inventory_sha256':sha(O/'original_GT_inventory.json'),'all_instances':len(rows),'counts':dict(counts),'maximum_outside_pixel_distance':max_distance,'distance_convention':'Euclidean distance between native foreground pixel centers. Four-neighbor adjacency proves distance=1; diagonal-only=sqrt2; farther uses exact distance transform.','mask_changes':False,'clipping_or_dilation_applied_to_training_targets':False,'interpretation':'A high frequency of noncontainment does not imply large spatial errors. Distances quantify only observed V-outside-A borders and do not validate hidden shape, physical fruit identity, or fit accuracy.','source_of_uncertainty_is_not_inferred':True,'records':results}
with (O/'outside_pixel_distance_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
