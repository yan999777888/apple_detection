"""Position-only visual hints on the training split, in original source coordinates."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from afruit_grade_model import ROOT,file_sha
from afruit_grade_train import read,atomic
from afruit_visible_quality import prepare_input

OUT=ROOT/'datasets/afruit_train_spots_v1'
MANIFEST=ROOT/'datasets/afruitdb_apples_v1/manifest.json'
REGIONS=ROOT/'experiments/afruit_grade_crop_alignment_v1/train_regions.json'


def source():
    rr=[r for r in read(MANIFEST)['samples'] if r['split']=='train']
    bysha={r['sha256']:r for r in read(REGIONS)}
    out=[]
    for gid,gr,n in [('burst_0100',1,11),('burst_0100',0,16),('burst_0201',1,12),('burst_0014',0,5)]:
        pool=sorted([r for r in rr if r['group_id']==gid and r['grade']==gr],key=lambda r:r['path'])
        out.extend(pool[round(j*(len(pool)-1)/(n-1))] for j in range(n))
    for i,r in enumerate(out):
        assert r['split']=='train'
        im=cv2.imdecode(np.fromfile(ROOT/r['path'],np.uint8),cv2.IMREAD_COLOR)
        small=prepare_input(im,1280)
        b=bysha[r['sha256']]['crop']; h,w=small.shape[:2]
        l,t,rt,bt=b if b else (0,0,w,h)
        yield i+1,r,im,small,b,small[t:bt,l:rt]


def render(rows):
    out=OUT/'review';out.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),4):
        page=Image.new('RGB',(800,860),'white')
        d=ImageDraw.Draw(page)
        for j,(idx,r,im,small,b,crop) in enumerate(rows[start:start+4]):
            x,y=(j%2)*400,(j//2)*430
            thumb=Image.fromarray(cv2.cvtColor(cv2.resize(crop,(360,360),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
            page.paste(thumb,(x+26,y+18))
            for k in range(11):
                u=k*36
                d.text((x+20+u,y),str(k),fill='black');d.text((x+4,y+12+u),str(k),fill='black')
                d.line((x+26+u,y+15,x+26+u,y+20),fill='red');d.line((x+22,y+18+u,x+30,y+18+u),fill='red')
            d.text((x+10,y+385),f'{idx:02d} {Path(r["path"]).stem} grade {r["grade"]} crop {bool(b)}',fill='black')
        page.save(out/f'{start+1:02d}.jpg',quality=92)


# Coordinates in percent of the displayed actual crop; images without a unique
# detection use the whole photograph. Only confident focal marks are included.
POS={
1:[(70,56,81,65)],2:[(54,50,65,61)],4:[(23,70,31,79)],
5:[(45,60,54,69),(62,30,72,37)],6:[(49,14,58,21),(72,35,78,44)],
7:[(52,17,59,24)],8:[(27,47,39,52),(41,60,47,73)],
9:[(24,62,29,74)],10:[(58,64,64,73)],11:[(83,28,89,38)],
32:[(44,77,52,84)],36:[(36,61,49,73)],38:[(58,56,60,61)],
}
NEG={
1:[(49,45,57,52),(32,39,36,50)],2:[(45,41,51,48)],3:[(43,47,49,56)],4:[(57,44,62,49)],
5:[(40,45,44,55)],6:[(44,60,50,65)],7:[(51,41,55,47)],
8:[(40,44,50,51)],9:[(13,44,27,50)],10:[(71,47,81,53)],11:[(79,44,89,49)],
12:[(53,45,57,50)],13:[(29,40,43,54),(50,52,56,57)],14:[(46,41,53,48),(60,20,66,26)],
15:[(42,40,53,50),(55,27,59,33)],16:[(40,18,48,26),(46,52,52,58)],
17:[(26,32,36,45),(43,50,48,55)],18:[(67,66,78,72),(27,36,35,42)],
19:[(62,39,84,48),(39,45,43,54)],20:[(36,62,51,71),(49,39,55,45)],
21:[(47,39,59,47),(35,34,42,40)],22:[(39,47,48,54),(57,38,62,47)],
23:[(52,43,61,52),(90,48,95,53)],24:[(42,49,49,58),(55,42,58,49)],
25:[(24,47,32,57),(50,48,57,55)],26:[(39,8,45,15),(46,46,53,53)],
27:[(32,20,42,28),(60,61,68,66)],31:[(41,43,50,52)],35:[(34,46,42,56)],
37:[(70,32,81,39)],40:[(29,55,45,62),(21,37,29,43)],
41:[(51,63,66,73),(49,52,60,59)],42:[(30,40,37,47)],43:[(7,38,17,46),(43,41,48,45)],
44:[(20,56,29,63)]}


def export():
    results=[]
    proofs=[]
    for idx,row,im,small,b,croppic in source():
        h,w=im.shape[:2];oh,ow=small.shape[:2]
        l,t,r,bottom=b if b else [0,0,ow,oh]
        def source_box(box):
            u,v,ur,vb=box
            return [round((l+(r-l)*u/100)*w/ow),round((t+(bottom-t)*v/100)*h/oh),
                    round((l+(r-l)*ur/100)*w/ow),round((t+(bottom-t)*vb/100)*h/oh)]
        rects=[list(map(source_box,boxes)) for boxes in [POS.get(idx,[]),NEG.get(idx,[])]]
        assert file_sha(ROOT/row['path'])==row['sha256']
        assert all(0<=x<a<=w and 0<=y<c<=h for boxes in rects for x,y,a,c in boxes)
        item=dict(path=row['path'],grade=row['grade'],group_id=row['group_id'],split='train',
                  source_hw=[h,w],sha256=row['sha256'],view_hw=[oh,ow],crop_xyxy_1280=b,
                  positives_xyxy=rects[0],negatives_xyxy=rects[1])
        results.append(item)
        img=cv2.resize(croppic,(288,288),interpolation=cv2.INTER_AREA)
        for boxes,ref,col in zip(rects,(POS.get(idx,[]),NEG.get(idx,[])),((15,30,255),(20,196,0))):
            for (x,y,q,v),hint in zip(boxes,ref,strict=True):
                out=[(x*ow/w-l)/(r-l),(y*oh/h-t)/(bottom-t),(q*ow/w-l)/(r-l),(v*oh/h-t)/(bottom-t)]
                assert np.max(np.abs(np.asarray(out)-np.array(hint)/100))<0.002
                xx,yy,qq,vv=[round(e*288) for e in out]
                cv2.rectangle(img,(xx,yy),(qq,vv),col,2)
        proofs.append((idx,Path(row['path']).stem,Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))))
    panel=Image.new('RGB',(288*4,318*((len(proofs)+3)//4)),'white')
    d=ImageDraw.Draw(panel)
    for i,(idx,stem,im) in enumerate(proofs):
        x,y=(i%4)*288,(i//4)*318
        panel.paste(im,(x,y));d.text((x+2,y+291),f'{idx:02d} {stem[-6:]}',fill='black')
    panel.save(OUT/'位置复核.jpg',quality=90)
    atomic(OUT/'manifest.json',dict(samples=results,input_manifest_sha256=file_sha(MANIFEST),region_sha256=file_sha(REGIONS),
                scope='training only; AI-reviewed small rectangles. Lesion type and physical severity unverified. Unmarked areas unknown; green boxes are visually ordinary structures, not exhaustive healthy tissue.',
                coordinates='positives_xyxy and negatives_xyxy are on decoded ORIGINAL image axes. crop_xyxy_1280 is on view_hw. No crop means use whole source.'))
    print('cases',len(results),'positive regions',sum(len(x['positives_xyxy']) for x in results),'negative regions',sum(len(x['negatives_xyxy']) for x in results))


if __name__=='__main__':
    cv2.setNumThreads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',action='store_true')
    if parser.parse_args().export:
        export()
        raise SystemExit
    # Stream full-size frames instead of retaining a 4K image for every annotation.
    rows=[]
    for i,r,im,small,b,crop in source():
        rows.append((i,r,None,None,b,crop.copy()))
    render(rows)
