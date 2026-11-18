"""Second-stage comparison for the Woo turntable images, whose backgrounds repeat."""

import cv2
import numpy as np
from PIL import Image


def fruit_view(im):
    rgb=np.asarray(im.convert("RGB").resize((256,176),Image.Resampling.BILINEAR))
    h,s,v=cv2.split(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV))
    red,green,blue=[c.astype(float) for c in cv2.split(rgb)]
    skin=((red>blue+7)&(green>blue-5)&(v>43)&(s>22)&((h<46)|(h>162))).astype("uint8")
    skin[:12]=0;skin[146:]=0;skin[:,:10]=0;skin[:,246:]=0
    skin=cv2.morphologyEx(skin,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,lbl,stats,_=cv2.connectedComponentsWithStats(skin)
    valid=[j for j in range(1,n) if stats[j,4]>=340 and stats[j,2]>=20 and stats[j,3]>=22
           and .35<stats[j,2]/stats[j,3]<2.6 and 15<stats[j,0]+stats[j,2]/2<240]
    if not valid:return None
    j=max(valid,key=lambda k:stats[k,4])
    x,y,w,h,area=stats[j]
    if not .36<area/(w*h)<=1 or area>19000:return None
    mask=(lbl[y:y+h,x:x+w]==j).astype(np.uint8)
    contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(mask,contours,-1,1,cv2.FILLED)
    crop=cv2.resize(rgb[y:y+h,x:x+w],(48,48),interpolation=cv2.INTER_AREA)
    local=cv2.resize(mask,(48,48),interpolation=cv2.INTER_NEAREST)
    lab=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB).astype(np.float32)
    # A descriptor only measures the similarity of the visible surface; two camera
    # positions can show different surfaces of the same physical fruit.
    return dict(lab=lab,mask=local,xyxy=[int(x),int(y),int(x+w),int(y+h)],
                fraction=float(area/(256*176)))


def _orient(v,idx):
    return np.rot90(v if idx<4 else v[:,::-1],idx%4)


def visible_surface(a,b):
    if a is None or b is None:return None
    best=None
    for i in range(8):
        lb=_orient(b["lab"],i);mb=_orient(b["mask"],i)
        mask=(a["mask"]!=0)&(mb!=0)
        overlap=float(mask.sum()/max(1,((a["mask"]!=0)|(mb!=0)).sum()))
        if mask.sum()<750:continue
        lhs=a["lab"][mask];rhs=lb[mask]
        residual=lhs-rhs
        robust=np.median(residual,axis=0)
        intensity=lhs[:,0]-lhs[:,0].mean();other=rhs[:,0]-rhs[:,0].mean()
        corr=float(intensity@other/(np.linalg.norm(intensity)*np.linalg.norm(other)+1.e-6))
        mae=float(np.abs(residual-.6*robust).mean())
        # Low-frequency global lighting alone may agree on two unmarked fruit. The
        # final comparison also includes contrast of spatially local texture.
        def detail(l):
            d=l[:,:,0]-cv2.GaussianBlur(np.ascontiguousarray(l[:,:,0]),(0,0),2.2)
            return d[cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)]
        da=detail(a["lab"]);db=detail(lb)
        texture=float(da@db/(np.linalg.norm(da)*np.linalg.norm(db)+1.e-6))
        key=.4*corr+.45*texture+.15*overlap-mae/150
        if best is None or key>best[0]:best=(key,dict(correlation=corr,texture=texture,absolute_error=mae,
                                                       mask_iou=overlap,orientation=i))
    if best is None:return None
    metrics=best[1]
    # Exclude clearly different visible surfaces from global hashes. Remaining
    # ambiguous near pairs still require manual review and are not auto-admitted.
    distinct=(metrics["texture"]<.30 and (metrics["correlation"]<.78 or metrics["absolute_error"]>9)) or (
              metrics["correlation"]<.3 and metrics["texture"]<.46) or (
              metrics["absolute_error"]>15 and metrics["texture"]<.53)
    definite=(metrics["texture"]>=.85 and metrics["correlation"]>=.95 and
              metrics["absolute_error"]<=5 and metrics["mask_iou"]>=.90)
    return dict(**metrics,relationship="different_visible_surface" if distinct else "near_same_surface" if definite else "uncertain_surface")


def lesion_context(im,box):
    b=np.asarray(box,float);w,h=b[2:]-b[:2];x,y=(b[:2]+b[2:])/2
    # Preserve nearby fruit context. A dark dot by itself has little discriminating power.
    side=max(64.,2.6*max(w,h))
    bounds=(int(x-side/2),int(y-side/2),int(x+side/2),int(y+side/2))
    a=np.asarray(im.crop(bounds).resize((32,32),Image.Resampling.BILINEAR).convert("RGB"))
    lab=cv2.cvtColor(a,cv2.COLOR_RGB2LAB).astype(np.float32)
    return dict(lab=lab,size=float((w*h)**.5))


def copied_lesion(a,b):
    if max(a["size"],b["size"])/max(1.,min(a["size"],b["size"]))>1.45:return None
    x=a["lab"];gray=x[:,:,0]
    d=gray-cv2.GaussianBlur(gray,(0,0),1.5)
    best=None
    for i in range(8):
        y=_orient(b["lab"],i);diff=x-y
        err=float(np.abs(diff-.5*np.mean(diff,axis=(0,1))).mean())
        if err>8.:continue
        gy=np.ascontiguousarray(y[:,:,0]);dy=gy-cv2.GaussianBlur(gy,(0,0),1.5)
        corr=float(np.sum(d*dy)/(np.linalg.norm(d)*np.linalg.norm(dy)+1.e-5))
        centered=gray-gray.mean();other=gy-gy.mean()
        lum=float(np.sum(centered*other)/(np.linalg.norm(centered)*np.linalg.norm(other)+1.e-5))
        if corr>=.78 and lum>=.90 and (best is None or corr>best["texture"]):
            best=dict(orientation=i,texture=corr,luminance=lum,mae=err)
    return best
