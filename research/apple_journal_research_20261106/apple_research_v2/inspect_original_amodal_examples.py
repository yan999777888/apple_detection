"""Five deterministic annotation-only visual checks; no model or GPU."""
from pathlib import Path
import hashlib,json,os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
cv2.setNumThreads(1)
ROOT=Path(r'F:\研究生毕业设计\模块一定位补偿以及预测\apple_journal_research_20261106');BASE=ROOT.parent;O=ROOT/'apple_research_v2/datasets/papple_paired_visible_amodal_stage_v1';C=BASE/'apple_detection/cache/datasets/local_apple_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for split in ['train','val']:
 d=json.loads((C/(split+'_manifest.json')).read_text(encoding='utf-8-sig'))
 for sample in d['samples']:
  for j,x in enumerate(sample['instances']):rows.append({'split':split,'filename':sample['image_id'],'index':j,'meta':x,'cache_sha256':sample['mask_sha256'],'rgb_sha256':sample['rgb_sha256']})
frac=lambda r:r['meta']['visible_outside_amodal_pixels']/r['meta']['visible_area_pixels']
pos=[r for r in rows if frac(r)>0];median=float(np.median([frac(r) for r in pos]));selected=[];used=set()
def add(reason,ranking):
 for r in ranking:
  key=(r['split'],r['filename'],r['index'])
  if key not in used:selected.append((reason,r));used.add(key);return
add('Heavy occlusion; low V/A',sorted([r for r in rows if r['meta']['visibility_V_area_over_A_area']>.01],key=lambda r:r['meta']['visibility_V_area_over_A_area']))
add('Median positive outside-V fraction',sorted(pos,key=lambda r:abs(frac(r)-median)))
add('Largest outside-V fraction',sorted(rows,key=frac,reverse=True))
add('Largest raw V/A ratio',sorted(rows,key=lambda r:r['meta']['visibility_V_area_over_A_area'],reverse=True))
add('High visibility, exact containment',sorted([r for r in rows if frac(r)==0],key=lambda r:r['meta']['visibility_V_area_over_A_area'],reverse=True))
fontpath=Path(r'C:\Windows\Fonts\arial.ttf');font=ImageFont.truetype(str(fontpath),17) if fontpath.exists() else ImageFont.load_default();small=ImageFont.truetype(str(fontpath),15) if fontpath.exists() else font
W=1500;rh=290;canvas=Image.new('RGB',(W,rh*5+40),'white');draw=ImageDraw.Draw(canvas);draw.text((12,8),'Original author annotations: A = full spherical-projection target; H = A minus V. Audit examples, not model outputs.',font=font,fill='black');cases=[]
for k,(reason,row) in enumerate(selected):
 m=row['meta'];cp=C/'masks'/row['split']/(Path(row['filename']).stem+'.npz');assert sha(cp)==row['cache_sha256']
 with np.load(cp,allow_pickle=False) as z:
  v=np.unpackbits(z['visible_packed'][row['index']],axis=-1,count=1300,bitorder='little');a=np.unpackbits(z['amodal_packed'][row['index']],axis=-1,count=1300,bitorder='little')
 outside=(v&(1-a));hidden=(a&(1-v));both=v&a;assert int(v.sum())==m['visible_area_pixels'] and int(a.sum())==m['amodal_area_pixels']
 vb=m['visible_topology']['bbox_xyxy'];ab=m['amodal_topology']['bbox_xyxy'];box=[max(0,min(vb[0],ab[0])-10),max(0,min(vb[1],ab[1])-10),min(1300,max(vb[2],ab[2])+10),min(1300,max(vb[3],ab[3])+10)];l,t,r,b=box
 imagepath=BASE/'apple_vision/local_amodal/data/papple/data/images'/row['split']/row['filename'];assert sha(imagepath)==row['rgb_sha256']
 with Image.open(imagepath) as image:rgb=image.convert('RGB').crop(box)
 panels=[('RGB context',rgb),('V: visible',Image.fromarray(v[t:b,l:r]*255).convert('RGB')),('A: full amodal',Image.fromarray(a[t:b,l:r]*255).convert('RGB')),('H: A minus V',Image.fromarray(hidden[t:b,l:r]*255).convert('RGB')),('V outside A',Image.fromarray(outside[t:b,l:r]*255).convert('RGB'))]
 y=40+k*rh;title=f"{reason} | {row['split']}/{row['filename']} ID={m['instance_id']} | V/A={m['visibility_V_area_over_A_area']:.5f}, outside={int(outside.sum())}px ({frac(row)*100:.3f}% of V)"
 draw.text((10,y),title,font=small,fill='black')
 for j,(name,panel) in enumerate(panels):
  panel.thumbnail((280,235),Image.Resampling.NEAREST);x=10+j*300;draw.text((x,y+23),name,font=font,fill='black');canvas.paste(panel,(x+(280-panel.width)//2,y+48+(235-panel.height)//2));draw.rectangle((x,y+45,x+280,y+282),outline=(170,170,170))
 distance=cv2.distanceTransform((1-a).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);d=distance[outside.astype(bool)]
 cases.append({'selection_rule':reason,'split':row['split'],'filename':row['filename'],'original_apple_ID':m['instance_id'],'source_cache_path':str(cp),'source_cache_sha256':row['cache_sha256'],'crop_xyxy_for_display_only':box,'visible_area':int(v.sum()),'amodal_area':int(a.sum()),'intersection_area':int(both.sum()),'hidden_A_minus_V_area':int(hidden.sum()),'outside_area':int(outside.sum()),'raw_V_over_A':m['visibility_V_area_over_A_area'],'outside_fraction_of_V':frac(row),'outside_distance_to_A_pixels':{'median':float(np.median(d)) if len(d) else 0,'max':float(d.max()) if len(d) else 0},'raster_visual_check_pending':True})
path=O/'five_original_GT_examples.png';canvas.save(path)
with (O/'five_original_GT_examples.json').open('x',encoding='utf-8') as f:json.dump({'selection':'Deterministic diverse severity examples, not a random prevalence sample','no_model_predictions':True,'no_source_masks_modified':True,'native_geometry_preserved_in_arrays':'Only display crops resized nearest-neighbor','contact_sheet_sha256':sha(path),'cases':cases},f,ensure_ascii=False,indent=2)
print(json.dumps({'output':str(path),'cases':[{k:r[k] for k in ['selection_rule','split','filename','original_apple_ID','raw_V_over_A','outside_fraction_of_V','outside_distance_to_A_pixels']} for r in cases]},ensure_ascii=False))
