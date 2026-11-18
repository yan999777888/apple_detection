"""Point of the largest mark score and image-ranking quality on AF development fruit crops."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from scipy.stats import rankdata
import train_grade_mark_contrast as run
import train_grade_spatial_repair as common
from matched_grade_loader import load


def run_check():
    net,cut,side=load(run.OUT,"cuda")
    im,_,meta=common.dataset(False,{},run.SUPPORT)
    rows=[];scenes=[]
    with torch.no_grad():
        for i,(v,rr) in enumerate(zip(im,meta,strict=True)):
            if rr["domain"]!="af" or rr["view"]!="auto" or rr["grade"]>1:continue
            field,old=net.encode(run.grade.tensor_rgb([v],"cuda"))
            score,e=net.evidence(field)
            flat=torch.nn.functional.avg_pool2d(score[:,None,2:-2,2:-2],2,1)[0,0]
            y,x=np.unravel_index(flat.argmax().item(),flat.shape)
            xy=((x+3)*8,(y+3)*8)
            rows.append(dict(key=rr["key"],grade=rr["grade"],old=int(old.argmax(1)),s=e.item(),topxy=list(map(int,xy))))
            img=Image.fromarray(v).resize((160,160))
            a,b=[round(k*160/224) for k in xy]
            ImageDraw.Draw(img).rectangle((a-5,b-5,a+5,b+5),outline='yellow',width=2)
            scenes.append(img)
    ordered=sorted(range(len(rows)),key=lambda i:(rows[i]["grade"],-rows[i]["s"]))
    sheet=Image.new("RGB",(160*8,185*((len(rows)+7)//8)),"white");draw=ImageDraw.Draw(sheet)
    for j,i in enumerate(ordered):
        xx,yy=(j%8)*160,(j//8)*185;sheet.paste(scenes[i],(xx,yy))
        draw.text((xx+2,yy+160),f"{rows[i]['grade']} #{i} {rows[i]['s']:.2f}",fill="black")
    sheet.save(run.OUT/"development_peak_review.jpg",quality=90)
    data=[r for r in rows if r["old"]!=2]
    good=[r["s"] for r in data if r["grade"]==0]
    mild=[r["s"] for r in data if r["grade"]==1]
    auc=np.mean([int(x>y)+.5*int(x==y) for x in mild for y in good])
    order=sorted(set(good+mild),reverse=True)
    best=max((sum(x>=cut for x in mild) for cut in [float("inf")]+order if sum(x>=cut for x in good)<=1))
    run.io.atomic(run.OUT/"development_peak_audit.json",dict(rows=rows,eligible_good=len(good),eligible_mild=len(mild),auc=float(auc),
                 max_minor_at_1_false_good_downgrade=best,interpretation="Diagnostic post-fit bound on frozen development scores only; never applied as a deployment threshold."))
    print(len(good),len(mild),auc,best)


if __name__=="__main__":run_check()

