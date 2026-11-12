"""Debug evidence sheets, not manuscript figures or relabeled ground truth."""
from pathlib import Path
import json,math,argparse
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/defect_cause_audit_20261112_v1'
p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');args=p.parse_args()
VIS=OUT/('visual_review_train' if args.train else 'visual_review');VIS.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf8'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
gt=read(OUT/'GT_records.json');fp=read(OUT/'FP_records.json')
cases=[]
groups=[('FP',[r for r in fp if r['partition']=='dev' and r['survives_NMS']]),('FN',[r for r in gt if r['partition']=='dev' and not r['TP']])]
if args.train:
    chosen_fp=[];chosen_gt=[]
    for cls in (0,1):
        for source,selected,key in [(fp,chosen_fp,lambda r:-r['score']),(gt,chosen_gt,lambda r:-r['size640'])]:
            eligible=[r for r in source if r['partition']=='train' and r['class']==cls and (source is gt or (r['survives_NMS'] and r['category']=='no_near_annotated_GT'))]
            seen=set()
            for r in sorted(eligible,key=lambda r:(key(r),r['image_id'])):
                if r['image_id'] in seen:continue
                selected.append(r);seen.add(r['image_id'])
                if len(seen)==6:break
    groups=[('TRAIN_FP',chosen_fp),('TRAIN_GT',chosen_gt)]
for prefix,rows in groups:
    for i,r in enumerate(sorted(rows,key=lambda r:(r['category'],r['image_id'],r.get('prediction_index',r.get('GT_index',0))))):
        cases.append(dict(r,review_id=f'{prefix}{i+1:03d}',review_type=prefix))
for start in range(0,len(cases),6):
    canvas=Image.new('RGB',(1200,1020),'white');d=ImageDraw.Draw(canvas)
    for pos,r in enumerate(cases[start:start+6]):
        x0=(pos%2)*600;y0=(pos//2)*340
        image=Image.open(r['image_path']).convert('RGB');b=r['box'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
        size=min(1024,max(256,2.0*max(b[2]-b[0],b[3]-b[1])));left=max(0,min(1024-size,cx-size/2));top=max(0,min(1024-size,cy-size/2))
        crop=(round(left),round(top),round(left+size),round(top+size));scale=280/(crop[2]-crop[0]);plain=image.crop(crop).resize((280,280))
        anno=plain.copy();ad=ImageDraw.Draw(anno);record=read(OUT/'predictions'/f"{Path(r['image_id']).stem}.json")
        def drawbox(box,color,label,width=2):
            p=[(box[0]-crop[0])*scale,(box[1]-crop[1])*scale,(box[2]-crop[0])*scale,(box[3]-crop[1])*scale]
            if p[2]<0 or p[3]<0 or p[0]>280 or p[1]>280:return
            ad.rectangle(p,outline=color,width=width);ad.text((max(0,p[0]),max(0,p[1]-12)),label,font=small,fill=color,stroke_width=1,stroke_fill='black')
        for i,g in enumerate(record['GT']):drawbox(g['box'],'lime',f"G{i} c{g['class']}")
        for i,p in enumerate(record['high_predictions']):
            if i in record['NMS_keep']:drawbox(p[:4],'cyan',f"P{i} c{int(p[5])} {p[4]:.2f}")
        drawbox(b,'magenta',r['review_id'],3)
        canvas.paste(plain,(x0+5,y0+55));canvas.paste(anno,(x0+300,y0+55))
        d.text((x0+5,y0+2),f"{r['review_id']} c{r['class']} {r['category']}",font=font,fill='black')
        d.text((x0+5,y0+23),r['image_id'],font=small,fill='black')
        r.update(sheet=f'sheet_{start//6+1:02d}.jpg',crop_xyxy=crop,visual_assessment='pending; not GT correction')
    canvas.save(VIS/f'sheet_{start//6+1:02d}.jpg',quality=94)
(VIS/'review_index.json').write_text(json.dumps({'legend':'left: original crop; right: green GT, cyan retained prediction, magenta focus.','selection':'train: each class 6 highest-score unmatched image-distinct FP and 6 largest image-distinct GT; diagnostic biased sample, no prevalence claim' if args.train else 'All 50 post-NMS FP and 51 FN, deterministic order.','cases':cases},ensure_ascii=False,indent=2),encoding='utf8')
print('rendered',len(cases),'cases',math.ceil(len(cases)/6),'sheets')
