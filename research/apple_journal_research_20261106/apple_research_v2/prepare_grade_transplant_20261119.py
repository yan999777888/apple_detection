"""Real-scratch residuals versus matched normal-skin residuals on the same training apple."""
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import afruit_grade_model as grade
import afruit_grade_train as io
import afruit_grade_crop_alignment as align
from afruit_visible_quality import prepare_input
from grade_detail_patch import square


OUT=grade.ROOT/"experiments/grade_pairs_20261119/transfer_sham_native"
REGIONS=grade.ROOT/"datasets/grade_pairs_sources_20261119/regions.json"


def hosts():
    rows=align.split_rows("train")
    regions=io.read(align.OUT/"train_regions.json")
    pool=defaultdict(list)
    for r,box in zip(rows,regions,strict=True):
        if r["grade"]==0 and box["crop"]:pool[r["group_id"]].append((r,box))
    result=[]
    for group,items in sorted(pool.items()):
        items=sorted(items,key=lambda i:i[0]["sha256"])
        for ix in np.linspace(0,len(items)-1,min(8,len(items)),dtype=int):
            r,reg=items[ix];x,y,rgt,bt=reg["crop"]
            raw=prepare_input(align.read_frame(r),1280)[y:bt,x:rgt]
            im=square(raw,320);rgb=im.astype(np.int16)
            # Pick a coloured, uniformly illuminated cheek. It is visually checked before fitting.
            best=None
            for yy in range(92,229,18):
                for xx in range(92,229,18):
                    patch=rgb[yy-18:yy+19,xx-18:xx+19]
                    ch=patch.max(2)-patch.min(2)
                    if np.percentile(ch,5)<42 or np.percentile(patch.max(2),5)<78:continue
                    quality=patch.std((0,1)).mean()
                    if best is None or quality<best[0]:best=(quality,xx,yy)
            if best:result.append(dict(path=r["path"],sha=r["sha256"],group=group,xy=best[1:],rgb=im))
    return result


def donor_pairs():
    rows=[r for r in io.read(REGIONS)["regions"] if r["source"]=="Woo" and r["split"]=="train"]
    normal={r["path"]:r for r in rows if r["subtype"]=="healthy_skin"}
    out=[]
    for r in rows:
        if not r.get("author_polygon_xy") or not r["parent_fruit_xyxy"] or r["path"] not in normal:continue
        ref=normal[r["path"]]
        ox,oy,oright,obot=r["parent_fruit_xyxy"];scale=320/max(oright-ox,obot-oy)
        x,y,right,bottom=r["source_image_xyxy"]
        x0,y0=int(np.floor(x))-4,int(np.floor(y))-4
        w,h=int(np.ceil(right))+4-x0,int(np.ceil(bottom))+4-y0
        img=cv2.cvtColor(align.read_frame(dict(path=r["path"],sha256=r["image_sha256"])),cv2.COLOR_BGR2RGB)
        nx,ny,nr,nb=map(int,ref["source_image_xyxy"])
        sham=img[ny:nb,nx:nr]
        if sham.shape[0]<h or sham.shape[1]<w:sham=np.rot90(sham)
        if sham.shape[0]<h or sham.shape[1]<w:continue
        sy,sx=(sham.shape[0]-h)//2,(sham.shape[1]-w)//2
        sham=np.ascontiguousarray(sham[sy:sy+h,sx:sx+w])
        real=img[y0:y0+h,x0:x0+w];mask=np.zeros((h,w),np.uint8)
        for pts in r["author_polygon_xy"]:
            cv2.fillPoly(mask,[np.rint(np.array(pts).reshape(-1,2)-[x0,y0]).astype(np.int32)],1)
        target=max(1,round(w*scale)),max(1,round(h*scale))
        alpha=cv2.resize(mask.astype(np.float32),target,interpolation=cv2.INTER_AREA)
        if alpha.sum()<8:continue
        # Identical support, inpainting and edge weights for true and sham donor.
        blend=mask.astype(np.float32)*np.minimum(cv2.distanceTransform(mask,cv2.DIST_L2,3)/2.,1.)
        def residual(p):
            base=cv2.inpaint(p,mask,3,cv2.INPAINT_TELEA)
            return cv2.resize((p.astype(np.float32)-base)*blend[:,:,None],target,interpolation=cv2.INTER_AREA)
        out.append(dict(id=r["id"],normal_id=ref["id"],capture=r["capture"],normal_xyxy=ref["source_image_xyxy"],
            src=r["path"],area=float(alpha.sum()),alpha=alpha,positive=residual(real),negative=residual(sham)))
    return out


def edit(host,donor):
    im=host["rgb"];cx,cy=host["xy"];h,w=donor["alpha"].shape
    x,y=cx-w//2,cy-h//2
    independent=donor["alpha"]>0
    def one(signal):
        out=im.copy()
        base=im[y:y+h,x:x+w];add=np.clip(base.astype(np.float32)+signal,0.,255.)
        out[y:y+h,x:x+w]=np.rint(add).astype(np.uint8)
        if not np.array_equal(base[~independent],out[y:y+h,x:x+w][~independent]):raise ValueError("Edit escaped contour")
        return out
    zone=np.zeros((320,320),np.float32);zone[y:y+h,x:x+w]=donor["alpha"]
    site=zone.reshape(80,4,80,4).mean((1,3))>.12
    return one(donor["positive"]),one(donor["negative"]),site


def prepare():
    hs=hosts();ds=donor_pairs();OUT.mkdir(exist_ok=True,parents=True)
    if not hs or not ds:raise ValueError("No traceable transfers")
    canvas=Image.new("RGB",(720,128*((len(hs)+3)//4)),"white")
    d=ImageDraw.Draw(canvas)
    for i,h in enumerate(hs):
        col=(i%4)*180;row=(i//4)*128
        x,y=h["xy"];im=h["rgb"];thumb=Image.fromarray(im).resize((100,100))
        canvas.paste(thumb,(col,row));d.rectangle((col+(x-16)*100//320,row+(y-16)*100//320,col+(x+16)*100//320,row+(y+16)*100//320),outline="lime")
        tile=im[y-23:y+23,x-23:x+23]
        canvas.paste(Image.fromarray(tile).resize((70,70)),(col+102,row+10))
        d.text((col,row+101),str(i)+" "+Path(h["path"]).stem[-6:],fill="black")
    canvas.save(OUT/"training_hosts.jpg")
    gallery=Image.new("RGB",(520,95*len(ds)),"white");dr=ImageDraw.Draw(gallery)
    for i,d in enumerate(ds):
        h=hs[i%len(hs)];pos,sham,_=edit(h,d);cx,cy=h["xy"]
        for j,v in enumerate((h["rgb"],pos,sham)):
            crop=v[cy-17:cy+18,cx-17:cx+18]
            gallery.paste(Image.fromarray(crop).resize((85,85)),(j*100,i*95))
        dr.text((306,i*95+18),d["id"][:8]+" area="+str(round(d["area"],1)),fill="black")
    gallery.save(OUT/"training_positive_sham.jpg")
    io.atomic(OUT/"eligibility.json",dict(hosts=[{k:v for k,v in h.items() if k!="rgb"} for h in hs],
        donors=[{k:v for k,v in d.items() if k not in ("positive","negative","alpha")} for d in ds]))
    print("available",len(hs),len(ds),flush=True)


if __name__=="__main__":
    cv2.setNumThreads(1);prepare()
