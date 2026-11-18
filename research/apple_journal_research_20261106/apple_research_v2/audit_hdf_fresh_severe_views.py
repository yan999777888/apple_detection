"""Reconstruct the HDF fresh-source false-severe images and same-source control views."""
import argparse
import base64
from collections import Counter,defaultdict
from html import escape
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

import afruit_grade_model as grade
from afruit_grade_train import read,atomic
from matched_grade_loader import load

ROOT=grade.ROOT
OUT=ROOT/"experiments/grade_error49_v1/hdf_fresh"
PRED=ROOT/"experiments/grade_matched_review_v1/external/predictions.json"
AUDIT=ROOT/"diagnostics/hdf_peripheral_v1_s20261106/frozen_split_manifest.json"
MODELS={"original":ROOT/"experiments/afruit_grade_release_v1",
        "candidate":ROOT/"experiments/afruit_grade_matched_v1/paired"}
CHANGES=("identity","neutral_background","border_context","mild_blur","boundary_blur","interior_blur",
         "lift_fruit_luminance","reduce_fruit_chroma","horizontal_mirror")
NAMES=("好果","轻瑕","重瑕")


def views():
    rows=sorted((r.copy() for r in read(PRED) if r["panel"]=="HDF" and r["target"]=="fresh"),
                key=lambda r:r["path"])
    ref=read(AUDIT)
    group_counts=Counter((q["split"],q["recommended_proxy_group_id"])
        for q in ref["representatives"] if not q["quarantined"] and q["author_class"]=="fresh")
    ids=Counter()
    if len({r["group"] for r in rows})!=69 or len(rows)!=69: raise ValueError("Unexpected grouping")
    for r in rows:
        prefix=("C","M","H")[r["candidate"]]
        ids[prefix]+=1
        r["id"]=f"{prefix}{ids[prefix]:02}"
        r["proxy_group_size_in_development"]=group_counts["development",r["group"]]
        image=ROOT/r["path"]
        if grade.file_sha(image)!=r["sha256"]: raise ValueError(image.name)
        with Image.open(image) as pil:
            rgb=np.array(ImageOps.exif_transpose(pil).convert("RGB"))
        h,w=rgb.shape[:2]
        r["hw"]=[h,w]
        r["image"]=rgb
        r["square"]=grade.rgb_square(rgb[:,:,::-1].copy())
    assert dict(ids)=={"H":36,"C":28,"M":5}
    return rows


def sheets(rows):
    for prefix in "HCM":
        subset=[r for r in rows if r["id"].startswith(prefix)]
        for offset in range(0,len(subset),12):
            panel=Image.new("RGB",(1080,3*290),"#e4e6e5")
            draw=ImageDraw.Draw(panel)
            for i,r in enumerate(subset[offset:offset+12]):
                x=i%4*270+5;y=i//4*290+3
                panel.paste(Image.fromarray(r["square"]),(x+8,y+25))
                h,w=r["hw"]
                draw.text((x+8,y+3),r["id"]+"  "+Path(r["path"]).stem,fill="black")
                draw.text((x+8,y+251),f"{w}x{h}  old={r['original']} new={r['candidate']}",fill="black")
            panel.save(OUT/f"{prefix}_{offset//12+1}.jpg",quality=93)


def border_white(im):
    # Restrict the diagnostic edit to near-white regions connected to the frame border.
    bright=((im.min(axis=2)>=232)&(np.ptp(im.astype(np.int16),axis=2)<=24)).astype(np.uint8)
    _,labels=cv2.connectedComponents(bright,connectivity=8)
    outer=np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])
    return np.isin(labels,outer[outer>0])


def variants(rgb):
    mask=border_white(rgb)
    recolor=rgb.copy()
    recolor[mask]=[160,160,160]
    # The content of each fruit is preserved; this manipulation changes scale
    # relative to the frame and adds edge-colour background at the same time.
    h,w=rgb.shape[:2]
    p=max(1,round(max(h,w)*.16))
    edge=np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]])
    colour=tuple(np.median(edge,axis=0).astype(int).tolist())
    shrink=cv2.copyMakeBorder(rgb,p,p,p,p,cv2.BORDER_CONSTANT,value=colour)
    blur=cv2.GaussianBlur(rgb,(0,0),max(.5,min(h,w)/90))
    width=max(2,round(min(h,w)*.028))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*width+1,2*width+1))
    edge_band=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,kernel).astype(bool)
    # Separate visible fruit/background boundary smoothing from smoothing the
    # remaining non-white content; masks are approximate and are not lesion labels.
    boundary=rgb.copy()
    boundary[edge_band]=blur[edge_band]
    interior=rgb.copy()
    inner=(~mask)&(~edge_band)
    interior[inner]=blur[inner]
    # Interior-only Lab interventions change appearance, not the biological grade;
    # these tests probe sensitivity and never supply new training labels.
    lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    high=lab.copy()
    high[:,:,0]=np.clip(high[:,:,0]+12,0,100)
    neutral=lab.copy()
    neutral[:,:,1:]*=.5
    toned=[]
    for changed in (high,neutral):
        view=rgb.copy()
        recoloured=np.rint(cv2.cvtColor(changed,cv2.COLOR_LAB2RGB)*255).clip(0,255).astype(np.uint8)
        # Feather the colour adjustment near the interior boundary to avoid a hard seam.
        dist=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
        alpha=np.clip((dist-width)/(width+1),0,1)[...,None]
        view=np.rint(rgb*(1-alpha)+recoloured*alpha).astype(np.uint8)
        toned.append(view)
    return {"identity":rgb,"neutral_background":recolor,"border_context":shrink,"mild_blur":blur,
            "boundary_blur":boundary,"interior_blur":interior,
            "lift_fruit_luminance":toned[0],"reduce_fruit_chroma":toned[1],
            "horizontal_mirror":np.ascontiguousarray(rgb[:,::-1])},float(mask.mean())


@torch.inference_mode()
def probe(rows):
    models={k:load(path,"cpu") for k,path in MODELS.items()}
    all_squares=defaultdict(list)
    for r in rows:
        changed,coverage=variants(r["image"]);r["white_border_fraction"]=coverage
        for key,im in changed.items(): all_squares[key].append(grade.rgb_square(im[:,:,::-1].copy()))
    for key,squares in all_squares.items():
        for name,(model,cutoff,side) in models.items():
            assert side==224
            for start in range(0,len(rows),16):
                b=squares[start:start+16]
                logits=model(grade.tensor_rgb(b,"cpu"))
                probs=logits.softmax(1).tolist()
                pred=grade.decide(logits,cutoff).tolist()
                for r,pr,score in zip(rows[start:start+len(b)],pred,probs,strict=True):
                    r.setdefault("probes",{}).setdefault(name,{})[key]=dict(grade=int(pr),scores=score)
                    if key=="identity" and pr!=r[name]: raise ValueError("Prediction replay mismatch "+r["id"])


def summarize(rows):
    out={}
    for prefix in "HCM":
        rr=[r for r in rows if r["id"].startswith(prefix)]
        out[prefix]=dict(n=len(rr),original_counts=dict(Counter(r["original"] for r in rr)),
            sizes=dict(Counter("x".join(map(str,r["hw"])) for r in rr)),
            probe_windows={k:[sum(r["probes"]["candidate"][k]["grade"]==i for r in rr) for i in range(3)]
                           for k in CHANGES})
    return out


def nearest_views(rows):
    small={r["id"]:cv2.resize(r["image"],(64,64),interpolation=cv2.INTER_AREA).astype(np.float32) for r in rows}
    masks={k:~border_white(im.astype(np.uint8)) for k,im in small.items()}
    for r in rows:
        key=r["id"]
        nearest={}
        for cohort in ("H","C"):
            candidates=[]
            for q in rows:
                target=q["id"]
                if target==key or not target.startswith(cohort):
                    continue
                # Compare RGB in the union of non-white content; retain boundary mismatch.
                area=masks[key]|masks[target]
                mae=float(np.abs(small[key][area]-small[target][area]).mean())
                candidates.append((mae,target))
            cost,neighbor=min(candidates)
            nearest[cohort]=dict(id=neighbor,foreground_union_rgb_mae_0_to_255=cost)
        r["nearest_pictorial_neighbor"]=nearest


def img_uri(array):
    buf=BytesIO()
    im=Image.fromarray(array)
    im.thumbnail((258,258))
    im.save(buf,"PNG")
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")


def render(rows):
    notes=read(OUT/"visual_notes.json")["items"]
    cards=[]
    for r in sorted(rows,key=lambda x:({"H":0,"C":1,"M":2}[x["id"][0]],x["id"])):
        key=r["id"]
        ch=r["probes"]["candidate"]
        probs=ch["identity"]["scores"]
        n=notes.get(key,"同来源对照；不作为训练或选型样本。")
        raw=img_uri(r["image"]);square=img_uri(r["square"])
        cards.append(f'<article><h3>{key}　{escape(Path(r["path"]).name)}</h3><aside>作者：新鲜 · 原模型：{NAMES[r["original"]]} · 新候选：{NAMES[r["candidate"]]}</aside>'
          f'<div class="images"><figure><img src="{raw}"><figcaption>作者图片 {r["hw"][1]}×{r["hw"][0]}</figcaption></figure><figure><img src="{square}"><figcaption>实际 224×224 分级输入</figcaption></figure></div>'
          f'<p>{escape(n)}</p><small>新候选同图试验：外接白背景改灰 {NAMES[ch["neutral_background"]["grade"]]}；果实缩小并补边 {NAMES[ch["border_context"]["grade"]]}；镜像 {NAMES[ch["horizontal_mirror"]["grade"]]}。原始输出分数 好/轻/重：{" / ".join(f"{z:.2f}" for z in probs)}。</small></article>')
    lead='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HDF 新鲜果图像核对</title><style>body{margin:2em auto;max-width:1140px;font:16px system-ui;color:#203428;background:#f6f7f5}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}article{padding:12px;background:white;border:1px solid #becdc5;break-inside:avoid}.images{display:flex}figure{margin:4px;text-align:center;width:48%}img{width:100%;aspect-ratio:1;object-fit:contain}small,aside,figcaption{color:#42574c}h3{margin:.4em 0}</style><h1>HDF 作者标记新鲜的 69 张苹果图</h1><p>H01–H36 是新候选判为重瑕的全部图；C01–C28 是同一来源判好的全部对照，M01–M05 为判轻瑕的其余图。原图和实际输入均放大显示，不额外生成细节。作者没有这些照片的商业等级或逐处病斑真值。只有图像编辑后的判级变化可作为模型敏感性的直接证据；文字观察不等于病理诊断。</p><main>'''
    (OUT/"HDF逐张误判与同源对照.html").write_text(lead+"".join(cards)+"</main></html>",encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("diagnose","render"))
    args=parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    rows=views()
    if args.action=="diagnose":
        sheets(rows);probe(rows);nearest_views(rows)
        atomic(OUT/"per_image_probes.json",[{k:v for k,v in r.items() if k not in ("image","square")} for r in rows])
        atomic(OUT/"summary.json",dict(groups=summarize(rows),label_scope="HDF source fresh, not a per-fruit market grade",
               input_manifest_sha256=grade.file_sha(PRED),models={k:grade.file_sha(v/"weights.pt") for k,v in MODELS.items()},
               experiments="CPU FP32. Edits performed independently on original RGB, no parameters, GT or split learned or changed."))
    else:
        for r,stored in zip(rows,read(OUT/"per_image_probes.json"),strict=True):
            assert stored["sha256"]==r["sha256"] and stored["id"]==r["id"]
            r.update(stored)
        render(rows)
